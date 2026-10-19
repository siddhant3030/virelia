"""Framework-neutral Guard: message masking, restore, the tool decorator."""

import asyncio
import json

import pytest
from conftest import VALID_AADHAAR

from virelia import Engine, Guard, Vault, VireliaBlocked, load_preset

EMAIL = "lakshmi@example.org"
GITHUB_TOKEN = "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"


@pytest.fixture
def vault(tmp_path):
    with Vault(tmp_path / "vault.db") as v:
        yield v


@pytest.fixture
def guard(vault):
    g = Guard(load_preset("agent"), vault=vault, hmac_salt=b"test-salt")
    yield g
    g.close()


def dump(value) -> str:
    return json.dumps(value, ensure_ascii=False)


class TestEngineScan:
    def test_scan_matches_shorthands(self, engine):
        payload = {"email": EMAIL}
        assert (
            engine.scan(payload, source="t", direction="result").payload
            == engine.scan_result("t", payload).payload
        )
        assert (
            engine.scan(payload, source="t", direction="request").payload
            == engine.scan_request("t", payload).payload
        )

    def test_source_is_recorded_as_tool(self, engine):
        result = engine.scan(EMAIL, source="llm", direction="request")
        assert {e.tool for e in result.events} == {"llm"}
        assert {e.direction for e in result.events} == {"request"}


class TestMaskMessages:
    def test_string_and_block_content_and_tool_args(self, guard):
        messages = [
            {"role": "user", "content": f"mail {EMAIL}, aadhaar {VALID_AADHAAR}"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "send", "arguments": dump({"to": EMAIL})},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"cc {EMAIL}"},
                    {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
                ],
            },
        ]
        result = guard.mask_messages(messages)
        assert not result.blocked
        assert EMAIL not in dump(result.payload)
        assert VALID_AADHAAR not in dump(result.payload)
        user, assistant, blocks = result.payload
        assert user["content"] == "mail [EMAIL_1], aadhaar [AADHAAR_1]"
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
            "to": "[EMAIL_1]"
        }
        assert blocks["content"][0]["text"] == "cc [EMAIL_1]"
        assert blocks["content"][1] == messages[2]["content"][1]
        assert {m["role"] for m in result.payload} == {"user", "assistant"}

    def test_does_not_mutate_input(self, guard):
        messages = [{"role": "user", "content": EMAIL}]
        guard.mask_messages(messages)
        assert messages == [{"role": "user", "content": EMAIL}]

    def test_restore_round_trip(self, guard):
        messages = [
            {"role": "user", "content": f"reach me at {EMAIL}"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "1", "function": {"name": "f", "arguments": dump({"to": EMAIL})}}
                ],
            },
        ]
        restored = guard.restore_messages(guard.mask_messages(messages).payload)
        assert restored[0] == messages[0]
        args = restored[1]["tool_calls"][0]["function"]["arguments"]
        assert json.loads(args) == {"to": EMAIL}

    def test_restored_json_args_stay_valid_json(self, guard):
        tricky = 'say "hi" to lakshmi@example.org'
        masked = guard.mask_text(tricky).payload
        args = dump({"text": masked})
        restored = guard.restore_messages(
            [{"role": "assistant", "tool_calls": [{"function": {"arguments": args}}]}]
        )
        assert json.loads(restored[0]["tool_calls"][0]["function"]["arguments"]) == {
            "text": tricky
        }

    def test_blocked_message(self, guard):
        result = guard.mask_messages([{"role": "user", "content": f"key {GITHUB_TOKEN}"}])
        assert result.blocked
        assert GITHUB_TOKEN not in dump(result.payload)


class TestScopes:
    def test_tokens_stable_within_scope(self, guard):
        first = guard.mask_text(EMAIL, scope="a").payload
        second = guard.mask_text(f"again {EMAIL}", scope="a").payload
        assert second == f"again {first}"

    def test_scopes_are_isolated(self, guard):
        guard.mask_text("other@example.org", scope="a")
        token_a = guard.mask_text(EMAIL, scope="a").payload
        token_b = guard.mask_text(EMAIL, scope="b").payload
        assert token_a != token_b
        assert guard.restore(token_b, scope="a") != EMAIL
        assert guard.restore(token_b, scope="b") == EMAIL

    def test_without_vault_tokenize_degrades_to_mask(self):
        guard = Guard(load_preset("agent"), hmac_salt=b"test-salt")
        masked = guard.mask_text(EMAIL).payload
        assert EMAIL not in masked
        assert guard.restore(masked) == masked

    def test_accepts_engine(self, vault):
        engine = Engine(load_preset("agent"), hmac_salt=b"test-salt", vault=vault)
        guard = Guard(engine)
        assert guard.restore(guard.mask_text(EMAIL).payload) == EMAIL


class TestToolDecorator:
    def test_tool_receives_real_values_and_result_is_masked(self, guard):
        seen = []

        @guard.tool
        def send(to: str) -> dict:
            seen.append(to)
            return {"sent_to": to}

        token = guard.mask_text(EMAIL).payload
        assert send(token) == {"sent_to": token}
        assert seen == [EMAIL]

    def test_async_tool(self, guard):
        seen = []

        @guard.tool(name="send")
        async def send(to: str) -> str:
            seen.append(to)
            return to

        token = guard.mask_text(EMAIL).payload
        assert asyncio.run(send(to=token)) == token
        assert seen == [EMAIL]

    def test_keeps_signature(self, guard):
        import inspect

        @guard.tool
        def send(to: str, body: str = "") -> str:
            """Send it."""
            return to

        assert list(inspect.signature(send).parameters) == ["to", "body"]
        assert send.__doc__ == "Send it."

    def test_blocked_result_raises(self, guard):
        @guard.tool
        def leak() -> dict:
            return {"token": GITHUB_TOKEN}

        with pytest.raises(VireliaBlocked) as exc:
            leak()
        assert exc.value.rule_id == "secrets-block"
        assert GITHUB_TOKEN not in str(exc.value)

    def test_scan_args_blocks_before_the_tool_runs(self, guard):
        calls = []

        @guard.tool(scan_args=True)
        def post(text: str) -> str:
            calls.append(text)
            return "ok"

        with pytest.raises(VireliaBlocked):
            post(f"my key is {GITHUB_TOKEN}")
        assert calls == []

    def test_scan_args_passes_real_values_through(self, guard):
        seen = []

        @guard.tool(scan_args=True)
        def send(to: str) -> str:
            seen.append(to)
            return "ok"

        send(guard.mask_text(EMAIL).payload)
        assert seen == [EMAIL]
