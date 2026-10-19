import pytest

from virelia import PolicyError, load_policy, load_preset, parse_policy
from virelia.models import Action


def base(**overrides):
    data = {
        "name": "test",
        "rules": [{"id": "r1", "entities": ["EMAIL"], "action": "mask"}],
    }
    data.update(overrides)
    return data


class TestPresets:
    @pytest.mark.parametrize(
        "name",
        ["ngo-default", "strict", "dev", "personal", "agent", "dpdp", "gdpr", "hipaa-lite"],
    )
    def test_shipped_presets_load(self, name):
        policy = load_preset(name)
        assert policy.name == name

    def test_unknown_preset(self):
        with pytest.raises(PolicyError, match="unknown preset"):
            load_preset("nope")

    def test_ngo_default_fails_closed(self):
        assert load_preset("ngo-default").defaults.unknown_entity_action is Action.MASK


def first_matching_action(policy, entity):
    """First-match-wins, mirroring engine semantics; falls back to the default."""
    for rule in policy.rules:
        if entity in rule.entities:
            return rule.action
    return policy.defaults.unknown_entity_action


class TestRegulationPacks:
    """The dpdp/gdpr/hipaa-lite packs are conservative drafts (issue #17);
    these tests pin their fail-closed defaults and key entity actions."""

    @pytest.mark.parametrize(
        ("name", "default_action"),
        [
            ("dpdp", Action.MASK),
            ("gdpr", Action.MASK),
            ("hipaa-lite", Action.BLOCK),
        ],
    )
    def test_packs_fail_closed(self, name, default_action):
        assert load_preset(name).defaults.unknown_entity_action is default_action

    @pytest.mark.parametrize(
        ("name", "entity", "expected"),
        [
            # dpdp: govt/financial IDs blocked, names/contact/IP masked
            ("dpdp", "AADHAAR", Action.BLOCK),
            ("dpdp", "PAN", Action.BLOCK),
            ("dpdp", "GOVT_ID", Action.BLOCK),
            ("dpdp", "CREDIT_CARD", Action.BLOCK),
            ("dpdp", "BANK_ACCOUNT", Action.BLOCK),
            ("dpdp", "PERSON_NAME", Action.MASK),
            ("dpdp", "EMAIL", Action.MASK),
            ("dpdp", "PHONE", Action.MASK),
            ("dpdp", "IP_ADDRESS", Action.MASK),
            ("dpdp", "ADDRESS", Action.REDACT),
            ("dpdp", "DOB", Action.REDACT),
            ("dpdp", "IFSC", Action.ALLOW),
            # gdpr: all personal-data entities at least masked, IDs blocked
            ("gdpr", "GOVT_ID", Action.BLOCK),
            ("gdpr", "AADHAAR", Action.BLOCK),
            ("gdpr", "CREDIT_CARD", Action.BLOCK),
            ("gdpr", "BANK_ACCOUNT", Action.BLOCK),
            ("gdpr", "PERSON_NAME", Action.MASK),
            ("gdpr", "EMAIL", Action.MASK),
            ("gdpr", "PHONE", Action.MASK),
            ("gdpr", "IP_ADDRESS", Action.MASK),
            ("gdpr", "ADDRESS", Action.REDACT),
            ("gdpr", "DOB", Action.REDACT),
            ("gdpr", "IFSC", Action.ALLOW),
            # hipaa-lite: every detectable Safe Harbor identifier blocked
            ("hipaa-lite", "PERSON_NAME", Action.BLOCK),
            ("hipaa-lite", "PHONE", Action.BLOCK),
            ("hipaa-lite", "EMAIL", Action.BLOCK),
            ("hipaa-lite", "IP_ADDRESS", Action.BLOCK),
            ("hipaa-lite", "BANK_ACCOUNT", Action.BLOCK),
            ("hipaa-lite", "CREDIT_CARD", Action.BLOCK),
            ("hipaa-lite", "GOVT_ID", Action.BLOCK),
            ("hipaa-lite", "AADHAAR", Action.BLOCK),
            ("hipaa-lite", "DOB", Action.BLOCK),
            ("hipaa-lite", "ADDRESS", Action.BLOCK),
            ("hipaa-lite", "IFSC", Action.ALLOW),
        ],
    )
    def test_key_entity_actions(self, name, entity, expected):
        assert first_matching_action(load_preset(name), entity) is expected

    @pytest.mark.parametrize("name", ["dpdp", "gdpr", "hipaa-lite"])
    def test_credentials_blocked_in_every_pack(self, name):
        policy = load_preset(name)
        for secret in ("AWS_ACCESS_KEY", "GITHUB_TOKEN", "JWT"):
            assert first_matching_action(policy, secret) is Action.BLOCK

    def test_hipaa_lite_secrets_withhold_whole_result(self):
        policy = load_preset("hipaa-lite")
        rule = next(r for r in policy.rules if "AWS_ACCESS_KEY" in r.entities)
        assert rule.scope == "result"

    def test_hipaa_lite_lowers_confidence_bar(self):
        assert load_preset("hipaa-lite").defaults.min_confidence == 0.3


class TestValidation:
    def test_minimal_policy(self):
        policy = parse_policy(base())
        assert policy.rules[0].action is Action.MASK
        assert policy.defaults.unknown_entity_action is Action.MASK

    def test_policy_wrapper_key_accepted(self):
        assert parse_policy({"policy": base()}).name == "test"

    def test_bad_action_is_readable(self):
        with pytest.raises(PolicyError) as exc:
            parse_policy(base(rules=[{"id": "r1", "entities": ["EMAIL"], "action": "obliterate"}]))
        message = str(exc.value)
        assert "rules.0.action" in message
        assert "Traceback" not in message

    def test_unknown_entity_rejected(self):
        with pytest.raises(PolicyError, match="AADHAR"):
            parse_policy(base(rules=[{"id": "r1", "entities": ["AADHAR"], "action": "mask"}]))

    def test_extra_entities_extend_known_set(self):
        policy = parse_policy(
            base(rules=[{"id": "r1", "entities": ["CUSTOM"], "action": "mask"}]),
            extra_entities=["CUSTOM"],
        )
        assert policy.rules[0].entities == ["CUSTOM"]

    def test_duplicate_rule_ids_rejected(self):
        with pytest.raises(PolicyError, match="duplicate rule id"):
            parse_policy(
                base(
                    rules=[
                        {"id": "r1", "entities": ["EMAIL"], "action": "mask"},
                        {"id": "r1", "entities": ["PHONE"], "action": "mask"},
                    ]
                )
            )

    def test_rule_needs_entities_or_match(self):
        with pytest.raises(PolicyError, match="entities"):
            parse_policy(base(rules=[{"id": "r1", "action": "mask"}]))

    def test_result_scope_requires_block(self):
        with pytest.raises(PolicyError, match="scope 'result'"):
            rule = {"id": "r1", "entities": ["EMAIL"], "action": "mask", "scope": "result"}
            parse_policy(base(rules=[rule]))

    def test_bad_json_path_rejected(self):
        with pytest.raises(PolicyError, match="json_path"):
            parse_policy(
                base(rules=[{"id": "r1", "match": {"json_path": "rows.x"}, "action": "block"}])
            )

    def test_unknown_top_level_key_rejected(self):
        with pytest.raises(PolicyError):
            parse_policy(base(detectors={"tiers": ["regex"]}))


class TestLoadFromFile:
    def test_load_yaml_file(self, tmp_path):
        f = tmp_path / "p.yaml"
        f.write_text(
            "name: filetest\nrules:\n  - id: r1\n    entities: [EMAIL]\n    action: redact\n"
        )
        policy = load_policy(f)
        assert policy.name == "filetest"
        assert policy.rules[0].action is Action.REDACT

    def test_error_names_the_file(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("name: x\nrules:\n  - id: r1\n    entities: [EMAIL]\n    action: nope\n")
        with pytest.raises(PolicyError, match="bad.yaml"):
            load_policy(f)
