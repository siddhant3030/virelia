"""Tier 2 NER detector — Presidio + spaCy behind the optional ``[ner]`` extra.

Nothing here imports presidio at module level, so the engine can always
import this module; ``load_ner_detector`` returns None when the extra (or
its spaCy model) is missing and the scan runs exactly as without it. The
analyzer is built once per process and reused across engines.

T2 confidence is capped at ``NER_MAX_CONFIDENCE`` so a statistical match
can never outrank a checksum-validated Tier 1 finding (Verhoeff/Luhn
matches score 0.85+).
"""

from __future__ import annotations

import importlib.util
import logging
import re

from virelia.detectors.base import Detector
from virelia.models import Span

logger = logging.getLogger("virelia")

_find_spec = importlib.util.find_spec  # indirection for tests

# Presidio entity name -> virelia entity name (the policy vocabulary).
PRESIDIO_TO_VIRELIA: dict[str, str] = {
    "PERSON": "PERSON_NAME",
    "LOCATION": "ADDRESS",
    "IN_AADHAAR_CTX": "AADHAAR",
    "IN_PAN_CTX": "PAN",
    "IN_IFSC_CTX": "IFSC",
    "IN_UPI_ID": "UPI_ID",
}

# Everything this tier can emit; the engine intersects policy coverage with
# this set before deciding to load anything.
NER_ENTITY_TYPES = frozenset(PRESIDIO_TO_VIRELIA.values())

# Entities no other tier finds in free text (T0 catches PERSON_NAME/ADDRESS
# only via column names; T1 has no detector for any of these). With
# defaults.ner: auto, the tier loads only when the policy needs one of them.
NER_ONLY_ENTITY_TYPES = frozenset({"PERSON_NAME", "ADDRESS", "UPI_ID"})

NER_MAX_CONFIDENCE = 0.84  # strictly below any checksum-validated T1 match

_SPACY_MODELS = ("en_core_web_lg", "en_core_web_md", "en_core_web_sm")

# Common Indian given names that spaCy's English models often miss in
# transliterated text. Matched case-sensitively as whole words, so common
# lowercase words never fire.
INDIAN_GIVEN_NAMES = [
    "Aarav", "Aditi", "Akash", "Amit", "Ananya", "Anil", "Anjali", "Arjun",
    "Asha", "Ayesha", "Deepak", "Devi", "Divya", "Farhan", "Fatima",
    "Gaurav", "Geeta", "Gopal", "Harish", "Imran", "Ishita", "Jyoti",
    "Kabir", "Kavita", "Kiran", "Lakshmi", "Lata", "Mahesh", "Manish",
    "Meera", "Mohan", "Mukesh", "Naveen", "Neha", "Nikhil", "Nisha",
    "Pooja", "Prakash", "Priya", "Rahul", "Rajesh", "Rakesh", "Ramesh",
    "Rani", "Ravi", "Rekha", "Rohan", "Sanjay", "Savita", "Seema",
    "Shreya", "Sita", "Sneha", "Suman", "Sunita", "Suresh", "Tanvi",
    "Uma", "Varun", "Vijay", "Vikram", "Vinod", "Yash",
]


class NerDetector(Detector):
    """Wraps a presidio AnalyzerEngine. Unlike T1 detectors it emits spans
    of multiple entity types, so the engine wires it into the dedicated T2
    slot instead of the single-entity ALL_DETECTORS registry."""

    name = "presidio_ner"
    entity_type = "NER"  # unused — multi-entity; see class docstring
    tier = 2

    def __init__(self, analyzer: object, entities: frozenset[str]) -> None:
        self._analyzer = analyzer
        self._presidio_entities = sorted(
            p for p, k in PRESIDIO_TO_VIRELIA.items() if k in entities
        )

    def detect(self, text: str) -> list[Span]:
        results = self._analyzer.analyze(
            text=text, language="en", entities=self._presidio_entities
        )
        spans: list[Span] = []
        for r in results:
            entity = PRESIDIO_TO_VIRELIA.get(r.entity_type)
            if entity is None:
                continue
            spans.append(
                Span(
                    start=r.start,
                    end=r.end,
                    entity_type=entity,
                    confidence=min(r.score, NER_MAX_CONFIDENCE),
                    tier=self.tier,
                    detector=self.name,
                )
            )
        return spans


_analyzer: object | None = None
_analyzer_failed = False


def load_ner_detector(entities: frozenset[str]) -> NerDetector | None:
    """Build a detector for the given virelia entity names, reusing the
    process-wide analyzer. Returns None when presidio or a spaCy model is
    unavailable — callers treat that as "tier not installed"."""
    wanted = entities & NER_ENTITY_TYPES
    if not wanted:
        return None
    analyzer = _get_analyzer()
    if analyzer is None:
        return None
    return NerDetector(analyzer, wanted)


def _get_analyzer() -> object | None:
    global _analyzer, _analyzer_failed
    if _analyzer is not None or _analyzer_failed:
        return _analyzer
    if _find_spec("presidio_analyzer") is None:
        _analyzer_failed = True
        return None
    try:
        _analyzer = _build_analyzer()
    except Exception as exc:  # missing model, version skew — never break a scan
        _analyzer_failed = True
        logger.warning("NER tier disabled: %s", exc)
    return _analyzer


def _build_analyzer() -> object:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    model = next((m for m in _SPACY_MODELS if _find_spec(m) is not None), None)
    if model is None:
        raise RuntimeError(
            "no spaCy English model installed; run: python -m spacy download en_core_web_sm"
        )
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model}],
        }
    ).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    for recognizer in _india_recognizers():
        analyzer.registry.add_recognizer(recognizer)
    return analyzer


def _india_recognizers() -> list:
    from presidio_analyzer import Pattern, PatternRecognizer

    case_sensitive = re.DOTALL | re.MULTILINE  # drop presidio's default IGNORECASE
    # Base scores sit below typical min_confidence; presidio's lemma context
    # enhancer (+0.35) lifts a match only when an India context word appears
    # nearby. This catches e.g. an Aadhaar-shaped number that fails the
    # Verhoeff check (so T1 rejected it) right next to the word "aadhaar".
    return [
        PatternRecognizer(
            supported_entity="IN_AADHAAR_CTX",
            name="in_aadhaar_ctx",
            patterns=[
                Pattern("aadhaar_like", r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)", 0.3)
            ],
            context=["aadhaar", "aadhar", "uidai", "uid"],
        ),
        PatternRecognizer(
            supported_entity="IN_PAN_CTX",
            name="in_pan_ctx",
            patterns=[Pattern("pan_like", r"\b[A-Z]{5}\d{4}[A-Z]\b", 0.3)],
            context=["pan", "income", "tax", "itr"],
            global_regex_flags=case_sensitive,
        ),
        PatternRecognizer(
            supported_entity="IN_IFSC_CTX",
            name="in_ifsc_ctx",
            patterns=[Pattern("ifsc_like", r"\b[A-Z]{4}0[A-Z0-9]{6}\b", 0.3)],
            context=["ifsc", "neft", "rtgs", "imps", "bank", "branch"],
            global_regex_flags=case_sensitive,
        ),
        PatternRecognizer(
            supported_entity="IN_UPI_ID",
            name="in_upi_id",
            # VPA shape overlaps emails; the low base score means it only
            # surfaces with payment context (upi/vpa/gpay/...) nearby.
            patterns=[Pattern("upi_vpa", r"\b[\w.-]{2,}@[A-Za-z]{2,64}\b", 0.2)],
            context=["upi", "vpa", "gpay", "phonepe", "paytm", "bhim", "pay"],
        ),
        PatternRecognizer(
            supported_entity="PERSON",
            name="in_person_names",
            deny_list=INDIAN_GIVEN_NAMES,
            deny_list_score=0.65,
            global_regex_flags=case_sensitive,
        ),
    ]
