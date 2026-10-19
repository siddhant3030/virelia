"""LangChain integration: the model sees tokens, tools and users see real
values. Skipped when langchain isn't installed (it's in the dev group)."""

import asyncio
import builtins
import json

import pytest
from conftest import VALID_AADHAAR

pytest.importorskip("langchain")

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from virelia import Guard, Vault, VireliaBlocked, load_preset  # noqa: E402
from virelia.integrations.langchain import (  # noqa: E402
    VireliaMiddleware,
    masking_runnable,
    restoring_runnable,
)

EMAIL = "lakshmi@example.org"
GITHUB_TOKEN = "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"


class RecordingModel(BaseChatModel):
    """Replays scripted replies and records exactly what the provider got."""

    replies: list
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "recording"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(messages)
        return ChatResult(generations=[ChatGeneration(message=reply)])


def sent_to_provider(model: RecordingModel) -> str:
    return json.dumps(
        [[m.model_dump(mode="json") for m in batch] for batch in model.seen],
        ensure_ascii=False,
    )


def tool_call(name, args, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def vault(tmp_path):
    with Vault(tmp_path / "vault.db") as v:
        yield v


@pytest.fixture
def emails():
    sent = []

    @tool
    def send_email(to: str, body: str) -> str:
        """Send an email."""
        sent.append(to)
        return f"sent to {to}"

    return send_email, sent


def build(model, tools, vault, **kwargs):
    middleware = VireliaMiddleware(vault=vault, hmac_salt=b"test-salt", **kwargs)
    return create_agent(
        model,
        tools,
        system_prompt="You help NGO staff.",
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )


def run(agent, text, thread="t1"):
    return agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        {"configurable": {"thread_id": thread}},
    )


class TestAgentMiddleware:
    def test_model_sees_tokens_tool_and_user_see_real_values(self, vault, emails):
        send_email, sent = emails
        model = RecordingModel(
            replies=[
                tool_call("send_email", {"to": "[EMAIL_1]", "body": "hi"}),
                AIMessage(content="Emailed [EMAIL_1]."),
            ]
        )
        agent = build(model, [send_email], vault)
        out = run(agent, f"email {EMAIL}, aadhaar {VALID_AADHAAR}")

        provider_view = sent_to_provider(model)
        assert EMAIL not in provider_view
        assert VALID_AADHAAR not in provider_view
        assert "[EMAIL_1]" in provider_view
        assert sent == [EMAIL]
        assert out["messages"][-1].content == f"Emailed {EMAIL}."
        # Local state keeps real values; only the provider copy was masked.
        assert out["messages"][0].content == f"email {EMAIL}, aadhaar {VALID_AADHAAR}"

    def test_tool_results_are_masked_on_the_next_model_call(self, vault):
        @tool
        def lookup(name: str) -> dict:
            """Look up a beneficiary."""
            return {"name": name, "email": EMAIL}

        model = RecordingModel(
            replies=[tool_call("lookup", {"name": "Lakshmi"}), AIMessage(content="ok")]
        )
        run(build(model, [lookup], vault), "find Lakshmi")
        assert EMAIL not in sent_to_provider(model)
        assert isinstance(model.seen[1][-1], ToolMessage)

    def test_tokens_stable_per_thread_and_isolated_across_threads(self, vault):
        model = RecordingModel(replies=[AIMessage(content="a")] * 3)
        agent = build(model, [], vault)
        run(agent, "first other@example.org", thread="t1")
        run(agent, f"then {EMAIL}", thread="t1")
        run(agent, f"only {EMAIL}", thread="t2")

        second_turn = model.seen[1][-1].content
        other_thread = model.seen[2][-1].content
        assert second_turn == "then [EMAIL_2]"
        assert other_thread == "only [EMAIL_1]"
        # The first turn is re-masked with the same token on the second call.
        assert model.seen[1][1].content == model.seen[0][1].content

    def test_blocked_input_never_reaches_the_model(self, vault):
        model = RecordingModel(replies=[AIMessage(content="should not run")])
        out = run(build(model, [], vault), f"deploy with {GITHUB_TOKEN}")
        assert model.seen == []
        assert "credential" in out["messages"][-1].content
        assert GITHUB_TOKEN not in out["messages"][-1].content

    def test_on_block_raise(self, vault):
        model = RecordingModel(replies=[AIMessage(content="x")])
        agent = build(model, [], vault, on_block="raise")
        with pytest.raises(VireliaBlocked):
            run(agent, f"deploy with {GITHUB_TOKEN}")

    def test_scan_tool_args_refuses_the_call(self, vault):
        posted = []

        @tool
        def post(text: str) -> str:
            """Post publicly."""
            posted.append(text)
            return "posted"

        model = RecordingModel(
            replies=[
                lambda _: tool_call("post", {"text": f"key {GITHUB_TOKEN}"}),
                AIMessage(content="could not post"),
            ]
        )
        out = run(build(model, [post], vault, scan_tool_args=True), "post it")
        assert posted == []
        refusal = next(m for m in out["messages"] if isinstance(m, ToolMessage))
        assert refusal.status == "error"

    def test_scan_tool_args_keeps_real_values_for_allowed_calls(self, vault, emails):
        send_email, sent = emails
        model = RecordingModel(
            replies=[
                tool_call("send_email", {"to": "[EMAIL_1]", "body": "hi"}),
                AIMessage(content="done"),
            ]
        )
        run(build(model, [send_email], vault, scan_tool_args=True), f"email {EMAIL}")
        assert sent == [EMAIL]

    def test_async_path(self, vault, emails):
        send_email, sent = emails
        model = RecordingModel(
            replies=[
                tool_call("send_email", {"to": "[EMAIL_1]", "body": "hi"}),
                AIMessage(content="Emailed [EMAIL_1]."),
            ]
        )
        agent = build(model, [send_email], vault)
        out = asyncio.run(
            agent.ainvoke(
                {"messages": [{"role": "user", "content": f"email {EMAIL}"}]},
                {"configurable": {"thread_id": "a1"}},
            )
        )
        assert EMAIL not in sent_to_provider(model)
        assert sent == [EMAIL]
        assert out["messages"][-1].content == f"Emailed {EMAIL}."

    def test_accepts_guard_and_policy(self, vault):
        guard = Guard(load_preset("agent"), vault=vault, hmac_salt=b"test-salt")
        assert VireliaMiddleware(guard).guard is guard
        assert VireliaMiddleware(load_preset("strict")).guard.engine.policy.name == "strict"


class TestRunnables:
    def test_mask_then_restore_around_a_model(self, vault):
        guard = Guard(load_preset("agent"), vault=vault, hmac_salt=b"test-salt")
        model = RecordingModel(replies=[lambda msgs: AIMessage(content=msgs[-1].content)])
        prompt = ChatPromptTemplate.from_messages([("human", "{q}")])
        chain = prompt | masking_runnable(guard) | model | restoring_runnable(guard)

        reply = chain.invoke({"q": f"echo {EMAIL}"})
        assert EMAIL not in sent_to_provider(model)
        assert reply.content == f"echo {EMAIL}"

    def test_masks_plain_strings_and_messages(self, vault):
        mask = masking_runnable("agent", vault=vault)
        assert EMAIL not in mask.invoke(f"hi {EMAIL}")
        assert EMAIL not in mask.invoke(HumanMessage(content=EMAIL)).content

    def test_blocked_raises(self, vault):
        with pytest.raises(VireliaBlocked):
            masking_runnable("agent", vault=vault).invoke(GITHUB_TOKEN)


def test_clear_error_without_langchain(monkeypatch):
    import virelia.integrations.langchain as integration

    real_import = builtins.__import__

    def no_langchain(name, *args, **kwargs):
        if name.startswith("langchain"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(integration, "_cached", None)
    monkeypatch.setattr(builtins, "__import__", no_langchain)
    with pytest.raises(ImportError, match=r"virelia\[langchain\]"):
        integration.VireliaMiddleware  # noqa: B018
