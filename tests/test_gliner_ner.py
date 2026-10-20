"""Unit tests for the GLiNER T2 backend. The model is faked throughout — these
run without the [gliner] extra (and without torch)."""

import pytest

from virelia import Engine, parse_policy
from virelia.detectors import gliner_ner, ner
from virelia.models import Span

TEXT = "मिले: लक्ष्मी देवी, गाँव रामपुर"
NAME = "लक्ष्मी देवी"


def hit(text, chunk, label, score):
    start = chunk.index(text)
    return {"start": start, "end": start + len(text), "text": text, "label": label, "score": score}


class FakeGliner:
    """Finds each (text, label, score) in a chunk — but, like the real model,
    only within the first MODEL_WINDOW_TOKENS tokens; the rest is dropped."""

    def __init__(self, targets=()):
        self.targets = list(targets)
        self.calls = []

    def predict_entities(self, text, labels, threshold):
        self.calls.append((text, tuple(labels), threshold))
        tokens = list(gliner_ner._TOKEN.finditer(text))
        if len(tokens) > gliner_ner.MODEL_WINDOW_TOKENS:
            text = text[: tokens[gliner_ner.MODEL_WINDOW_TOKENS].start()]
        return [hit(t, text, label, s) for t, label, s in self.targets if t in text]


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr(gliner_ner, "_models", {})
    monkeypatch.setattr(gliner_ner, "_failed", set())


class TestChunking:
    def test_short_text_is_one_chunk(self):
        assert gliner_ner.chunk_text("a b c") == [(0, "a b c")]

    def test_blank_text_has_no_chunks(self):
        assert gliner_ner.chunk_text("  \n ") == []

    def test_windows_overlap_and_offsets_index_the_original(self):
        text = " ".join(f"w{i}" for i in range(25))
        chunks = gliner_ner.chunk_text(text, size=10, overlap=3)
        for offset, chunk in chunks:
            assert text[offset : offset + len(chunk)] == chunk
        assert chunks[0][1].split()[-3:] == chunks[1][1].split()[:3]
        assert chunks[-1][1].endswith("w24")

    def test_every_word_is_covered(self):
        text = "\n".join(f"w{i}" for i in range(500))
        seen = {w for _, c in gliner_ner.chunk_text(text) for w in c.split()}
        assert seen == set(text.split())


class TestGlinerDetector:
    def test_maps_labels_and_caps_confidence(self):
        model = FakeGliner([(NAME, "person", 0.97), ("रामपुर", "address", 0.5)])
        det = gliner_ner.GlinerDetector(model, frozenset({"PERSON_NAME", "ADDRESS"}))
        spans = det.detect(TEXT)
        assert [TEXT[s.start : s.end] for s in spans] == [NAME, "रामपुर"]
        assert spans[0] == Span(
            spans[0].start, spans[0].end, "PERSON_NAME", ner.NER_MAX_CONFIDENCE, 2, "gliner_ner"
        )
        assert spans[1].confidence == 0.5

    def test_prompts_only_wanted_labels(self):
        model = FakeGliner()
        gliner_ner.GlinerDetector(model, frozenset({"UPI_ID"})).detect("hi")
        assert model.calls == [("hi", ("upi id",), gliner_ner.GLINER_THRESHOLD)]

    def test_long_text_finds_name_past_the_model_window(self):
        text = "filler " * 1000 + "met Bhavna Oraon today"
        model = FakeGliner([("Bhavna Oraon", "person", 0.9)])
        spans = gliner_ner.GlinerDetector(model, frozenset({"PERSON_NAME"})).detect(text)
        assert [text[s.start : s.end] for s in spans] == ["Bhavna Oraon"]
        assert len(model.calls) > 1

    def test_punctuation_dense_text_never_overflows_the_window(self):
        # 150 whitespace "words" but 750 GLiNER tokens, with the name in the
        # middle: whitespace-sized chunks would hand the model one overflowing
        # window and lose it.
        text = "a,b;c " * 100 + "met Bhavna Oraon " + "a,b;c " * 50
        for _, chunk in gliner_ner.chunk_text(text):
            assert len(gliner_ner._TOKEN.findall(chunk)) <= gliner_ner.MODEL_WINDOW_TOKENS
        model = FakeGliner([("Bhavna Oraon", "person", 0.9)])
        spans = gliner_ner.GlinerDetector(model, frozenset({"PERSON_NAME"})).detect(text)
        assert [text[s.start : s.end] for s in spans] == ["Bhavna Oraon"]

    def test_overlap_duplicates_collapse_to_one_span(self):
        words = ["x"] * 190 + ["Bhavna", "Oraon"] + ["y"] * 50
        text = " ".join(words)
        model = FakeGliner([("Bhavna Oraon", "person", 0.9)])
        spans = gliner_ner.GlinerDetector(model, frozenset({"PERSON_NAME"})).detect(text)
        assert len(model.calls) == 2  # the name sits in both windows
        assert len(spans) == 1

    def test_name_list_backstops_the_model(self):
        det = gliner_ner.GlinerDetector(FakeGliner(), frozenset({"PERSON_NAME"}))
        [span] = det.detect("call Lakshmi tomorrow")
        assert (span.entity_type, span.confidence) == ("PERSON_NAME", gliner_ner.NAME_LIST_SCORE)

    def test_name_list_off_when_person_not_wanted(self):
        det = gliner_ner.GlinerDetector(FakeGliner(), frozenset({"ADDRESS"}))
        assert det.detect("call Lakshmi tomorrow") == []


class TestLoader:
    def test_none_without_gliner(self, monkeypatch):
        monkeypatch.setattr(gliner_ner, "_find_spec", lambda name: None)
        assert gliner_ner.load_gliner_detector(frozenset({"PERSON_NAME"})) is None

    def test_none_for_entities_gliner_does_not_cover(self):
        assert gliner_ner.load_gliner_detector(frozenset({"AADHAAR", "PAN"})) is None

    def test_build_failure_is_cached_and_never_raises(self, monkeypatch):
        builds = []

        def boom(name):
            builds.append(name)
            raise OSError("offline")

        monkeypatch.setattr(gliner_ner, "_find_spec", lambda name: object())
        monkeypatch.setattr(gliner_ner, "_build_model", boom)
        for _ in range(2):
            assert gliner_ner.load_gliner_detector(frozenset({"PERSON_NAME"})) is None
        assert builds == [gliner_ner.DEFAULT_GLINER_MODEL]

    def test_model_loaded_once_per_name(self, monkeypatch):
        builds = []
        monkeypatch.setattr(gliner_ner, "_find_spec", lambda name: object())
        monkeypatch.setattr(
            gliner_ner, "_build_model", lambda name: builds.append(name) or FakeGliner()
        )
        for _ in range(2):
            gliner_ner.load_gliner_detector(frozenset({"PERSON_NAME"}), "local/model")
        assert builds == ["local/model"]

    def test_entities_are_policy_addressable(self):
        assert ner.NER_ENTITY_TYPES >= gliner_ner.GLINER_ENTITY_TYPES


class TestEngineBackendSelection:
    def policy(self, **defaults):
        return parse_policy(
            {
                "name": "p",
                "defaults": {"unknown_entity_action": "allow", **defaults},
                "rules": [{"id": "n", "entities": ["PERSON_NAME"], "action": "mask"}],
            }
        )

    def test_gliner_backend_masks_devanagari_name(self, monkeypatch):
        model = FakeGliner([(NAME, "person", 0.9)])
        monkeypatch.setattr(gliner_ner, "_find_spec", lambda name: object())
        monkeypatch.setattr(gliner_ner, "_build_model", lambda name: model)
        engine = Engine(self.policy(ner_backend="gliner"), hmac_salt=b"s")
        result = engine.scan_result("get_notes", {"notes": TEXT})
        assert NAME not in result.payload["notes"]
        assert any(e.entity_type == "PERSON_NAME" and e.tier == 2 for e in result.events)

    def test_ner_model_is_passed_to_the_loader(self, monkeypatch):
        calls = []
        monkeypatch.setattr(gliner_ner, "load_gliner_detector", lambda e, m: calls.append((e, m)))
        policy = self.policy(ner_backend="gliner", ner_model="/models/pii")
        Engine(policy, hmac_salt=b"s").scan_result("t", {"notes": "free text"})
        assert calls == [(frozenset({"PERSON_NAME"}), "/models/pii")]

    def test_default_backend_stays_presidio(self, monkeypatch):
        calls = []
        monkeypatch.setattr(ner, "load_ner_detector", lambda e: calls.append(e))
        monkeypatch.setattr(
            gliner_ner, "load_gliner_detector", lambda *a: pytest.fail("gliner loaded")
        )
        Engine(self.policy(), hmac_salt=b"s").scan_result("t", {"notes": "free text"})
        assert calls == [frozenset({"PERSON_NAME"})]

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(Exception, match="ner_backend"):
            self.policy(ner_backend="spacy-transformers")
