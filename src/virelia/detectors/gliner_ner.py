"""Tier 2 NER backend on GLiNER — open-source transformer models, behind the
optional ``[gliner]`` extra. Selected with ``defaults.ner_backend: gliner``.

Why a second backend: spaCy's English models miss names written in Indic
scripts entirely, and plain Presidio is what every competitor already wraps.
GLiNER takes entity labels as plain text at inference time, so one Apache-2.0
multilingual model covers every label without a per-language pipeline. The
default is the general ``urchade/gliner_multi-v2.1`` rather than the PII-tuned
``gliner_multi_pii-v1``: Virelia only prompts three labels, and on
benchmarks/bench_ner.py the general model scores the same or better (it is
the only one that catches Hindi addresses).

Same contract as the Presidio backend: nothing imports gliner at module level,
the loader returns None when the extra or the model is unavailable, the model
is loaded once per process, and confidence is capped below Tier 1.

Weights come from the Hugging Face Hub on first load. For an offline machine,
download once and point ``defaults.ner_model`` at the local directory (or set
``HF_HUB_OFFLINE=1`` after a first run).
"""

from __future__ import annotations

import importlib.util
import logging
import re

from virelia.detectors.base import Detector
from virelia.detectors.ner import (
    INDIAN_GIVEN_NAMES,
    NER_ENTITY_TYPES,
    NER_MAX_CONFIDENCE,
)
from virelia.models import Span

logger = logging.getLogger("virelia")

_find_spec = importlib.util.find_spec  # indirection for tests

DEFAULT_GLINER_MODEL = "urchade/gliner_multi-v2.1"

# GLiNER label (natural language, what the model is prompted with) -> virelia
# entity. AADHAAR/PAN/IFSC stay with Tier 1, whose checksums beat a model.
GLINER_TO_VIRELIA: dict[str, str] = {
    "person": "PERSON_NAME",
    "address": "ADDRESS",
    "upi id": "UPI_ID",
}

GLINER_ENTITY_TYPES = frozenset(GLINER_TO_VIRELIA.values())

# Scored below the model's own threshold so policy min_confidence decides.
GLINER_THRESHOLD = 0.3
NAME_LIST_SCORE = 0.65

# GLiNER silently drops input past its window (384 tokens), so long leaves
# are scanned in overlapping windows; a name straddling one boundary is whole
# in the next window. Windows are counted with GLiNER's own word splitter,
# where every punctuation mark is a token — counting whitespace words would
# let a dense CSV-like leaf overflow the window inside a single chunk.
MODEL_WINDOW_TOKENS = 384
CHUNK_TOKENS = 200
OVERLAP_TOKENS = 30

_TOKEN = re.compile(r"\w+(?:[-_]\w+)*|\S")
_NAME_LIST = re.compile(r"\b(?:" + "|".join(INDIAN_GIVEN_NAMES) + r")\b")


def chunk_text(
    text: str, size: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS
) -> list[tuple[int, str]]:
    """Split into windows of ``size`` GLiNER tokens overlapping by ``overlap``;
    each item is (char offset in ``text``, window text)."""
    words = [m.span() for m in _TOKEN.finditer(text)]
    if len(words) <= size:
        return [(0, text)] if words else []
    chunks = []
    step = size - overlap
    for first in range(0, len(words), step):
        last = min(first + size, len(words)) - 1
        start, end = words[first][0], words[last][1]
        chunks.append((start, text[start:end]))
        if last == len(words) - 1:
            break
    return chunks


class GlinerDetector(Detector):
    """Multi-entity like ``NerDetector``: wired into the engine's T2 slot."""

    name = "gliner_ner"
    entity_type = "NER"  # unused — multi-entity
    tier = 2

    def __init__(self, model: object, entities: frozenset[str]) -> None:
        self._model = model
        self._entities = entities
        self._labels = sorted(g for g, v in GLINER_TO_VIRELIA.items() if v in entities)

    def detect(self, text: str) -> list[Span]:
        best: dict[tuple[int, int, str], float] = {}
        for offset, chunk in chunk_text(text):
            for hit in self._model.predict_entities(
                chunk, self._labels, threshold=GLINER_THRESHOLD
            ):
                entity = GLINER_TO_VIRELIA.get(hit["label"])
                if entity is None:
                    continue
                key = (offset + hit["start"], offset + hit["end"], entity)
                best[key] = max(best.get(key, 0.0), hit["score"])
        if "PERSON_NAME" in self._entities:
            # The deny list backstops names the model scores low; it is the
            # same list the Presidio backend uses, so switching backends
            # never loses a name the other one caught.
            for m in _NAME_LIST.finditer(text):
                key = (m.start(), m.end(), "PERSON_NAME")
                best[key] = max(best.get(key, 0.0), NAME_LIST_SCORE)
        return [
            Span(
                start=start,
                end=end,
                entity_type=entity,
                confidence=min(score, NER_MAX_CONFIDENCE),
                tier=self.tier,
                detector=self.name,
            )
            for (start, end, entity), score in sorted(best.items())
        ]


_models: dict[str, object] = {}
_failed: set[str] = set()


def load_gliner_detector(
    entities: frozenset[str], model_name: str | None = None
) -> GlinerDetector | None:
    """None when gliner or the model is unavailable — "tier not installed"."""
    wanted = entities & GLINER_ENTITY_TYPES & NER_ENTITY_TYPES
    if not wanted:
        return None
    model = _get_model(model_name or DEFAULT_GLINER_MODEL)
    if model is None:
        return None
    return GlinerDetector(model, wanted)


def _get_model(model_name: str) -> object | None:
    if model_name in _models or model_name in _failed:
        return _models.get(model_name)
    if _find_spec("gliner") is None:
        _failed.add(model_name)
        return None
    try:
        _models[model_name] = _build_model(model_name)
    except Exception as exc:  # no network, missing weights — never break a scan
        _failed.add(model_name)
        logger.warning("GLiNER NER tier disabled (%s): %s", model_name, exc)
    return _models.get(model_name)


def _build_model(model_name: str) -> object:
    from gliner import GLiNER

    return GLiNER.from_pretrained(model_name)
