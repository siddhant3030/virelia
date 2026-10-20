"""Policy schema (pydantic v2). Structure-only validation lives here;
entity-name validation against the detector registry happens in the loader
so custom detectors can extend the known set."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from virelia.models import Action
from virelia.pathmatch import compile_path


class StructuralMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = "*"  # fnmatch glob against the tool name
    json_path: str

    @field_validator("json_path")
    @classmethod
    def _valid_path(cls, v: str) -> str:
        compile_path(v)  # raises PathPatternError with a readable message
        return v


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    entities: list[str] = []
    match: StructuralMatch | None = None
    action: Action
    scope: Literal["field", "result"] = "field"
    message: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Rule:
        if not self.entities and self.match is None:
            raise ValueError(f"rule {self.id!r} must specify 'entities' and/or 'match'")
        if self.scope == "result" and self.action is not Action.BLOCK:
            raise ValueError(
                f"rule {self.id!r}: scope 'result' is only valid with action 'block'"
            )
        return self


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Fail closed: entities no rule covers still get masked.
    unknown_entity_action: Action = Action.MASK
    min_confidence: float = 0.4
    # NER tier ([ner] or [gliner] extra). "auto" loads it only when the
    # policy needs entities tiers 0-1 can't find in free text; true always
    # loads it; false never does. A missing extra silently disables it.
    ner: bool | Literal["auto"] = "auto"
    # presidio: spaCy + India-context recognizers ([ner] extra). gliner: an
    # open-source transformer, multilingual by default ([gliner] extra).
    ner_backend: Literal["presidio", "gliner"] = "presidio"
    # gliner only: a Hugging Face model id or a local directory.
    ner_model: str | None = None


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: int = 1
    defaults: Defaults = Defaults()
    rules: list[Rule] = []

    @model_validator(mode="after")
    def _unique_rule_ids(self) -> Policy:
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(f"duplicate rule id {rule.id!r}")
            seen.add(rule.id)
        return self
