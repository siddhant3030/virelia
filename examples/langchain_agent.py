"""Watch the trust boundary: what the model sees vs. what tools and users see.

Runs offline with a scripted fake model — no API key needed:

    uv run --extra langchain --extra vault python examples/langchain_agent.py
"""

import tempfile
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from virelia import Vault
from virelia.integrations.langchain import VireliaMiddleware


class ScriptedModel(BaseChatModel):
    """Stands in for a real provider and prints exactly what it receives."""

    replies: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        print("\n--- model provider receives ---")
        for m in messages:
            print(f"  {m.type:>6}: {m.content or m.tool_calls}")
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


@tool
def send_sms(phone: str, body: str) -> str:
    """Send an SMS to a beneficiary."""
    print(f"\n--- tool receives ---\n  send_sms(phone={phone!r})")
    return f"queued SMS to {phone}"


model = ScriptedModel(
    replies=[
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "send_sms",
                    "args": {"phone": "[PHONE_1]", "body": "Camp at 10am"},
                    "id": "1",
                }
            ],
        ),
        AIMessage(content="Sent the reminder to [PHONE_1]."),
    ]
)

with tempfile.TemporaryDirectory() as tmp, Vault(Path(tmp) / "vault.db") as vault:
    agent = create_agent(
        model,
        [send_sms],
        system_prompt="You help NGO field staff.",
        middleware=[VireliaMiddleware(vault=vault)],
        checkpointer=InMemorySaver(),
    )
    request = "Remind 9876543210 (Aadhaar 2345 6789 0124) about the camp"
    out = agent.invoke(
        {"messages": [{"role": "user", "content": request}]},
        {"configurable": {"thread_id": "demo"}},
    )
    print(f"\n--- user reads ---\n  {out['messages'][-1].content}")
