# Virelia in LangChain (and any agent)

Virelia plugs into an agent as middleware. It masks personal data and secrets
in everything sent to the model, then restores the real values for your tools
and your users.

```bash
pip install 'virelia[langchain,vault]'
```

## Agent middleware (`create_agent`)

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from virelia import Vault
from virelia.integrations.langchain import VireliaMiddleware

agent = create_agent(
    "openai:gpt-5-mini",
    tools=[send_email, lookup_beneficiary],
    middleware=[VireliaMiddleware(vault=Vault())],
    checkpointer=InMemorySaver(),
)
agent.invoke(
    {"messages": [{"role": "user", "content": "email lakshmi@example.org"}]},
    {"configurable": {"thread_id": "case-42"}},
)
```

### The trust boundary

```
            local, trusted                      │  third party
                                                │
 user ──▶ agent state (real values) ──mask──▶   │  model provider
            ▲            │                      │      │
            │            ▼                      │      │
         restore ◀── tools (real values)        │      │
            ▲                                   │      │
            └──────────── reply with tokens ◀───┼──────┘
```

The model provider is treated as the only untrusted party. The middleware
guards the model call and nothing else:

1. **Before each call** it masks a *copy* of the request:
   - every message's text;
   - the tool-call arguments on earlier AI messages;
   - the tool results;
   - the system prompt.

   Agent state is never rewritten.
2. **After each call** it restores the tokens in the reply. That covers the
   text, the tool-call arguments and any structured output. The reply goes into
   state only after it is restored.

What follows from this:

- **Tools receive real values.** When the model calls `send_email(to="[EMAIL_1]")`,
  the tool receives `lakshmi@example.org`.
- **Users read real values** in the final reply.
- **Tool results never reach the model unmasked.** They sit in state as
  plaintext and are masked on the next model call.
- **Tokens stay stable within a thread.** Each conversation's `thread_id` gets
  its own token namespace in the vault. Within a thread, the same email is
  `[EMAIL_1]` on every turn. In another thread the numbering starts over, so
  the two threads cannot be linked.

### Options

```python
VireliaMiddleware(
    policy="agent",          # preset name, Policy, Engine, or a Guard
    vault=Vault(),           # enables reversible tokens (virelia[vault])
    scope_by_thread=True,    # token namespace per LangGraph thread_id
    scan_tool_args=False,    # also scan tool arguments under the tool's name
    on_block="respond",      # or "raise" → VireliaBlocked
)
```

- **`policy`**
  - The default `agent` preset:
    - **tokenizes** email, phone, Aadhaar, PAN, government ID and UPI ID;
    - **withholds** card and bank numbers;
    - **refuses the whole message** when it contains a credential (AWS key,
      GitHub token, JWT).
  - Any [YAML policy](policy-schema.md) works. Policy rules match model
    traffic under the source name `llm`. With `match.tool: llm` you can write
    rules that apply only to prompts.
- **`vault`**
  - Without a vault, `tokenize` falls back to `mask`. Nothing leaks, but the
    model sees `********` instead of a token it can refer back to, and nothing
    can be restored.
- **`on_block`**
  - `"respond"` makes the agent reply with the policy message and skips the
    model call entirely.
  - `"raise"` raises `VireliaBlocked` (which carries `reason`, `rule_id` and
    audit `events`) so your app can handle it.
- **`scan_tool_args`**
  - Tools are trusted by default and receive real values. Turn this on when
    some tools are *not* trusted, for example ones that post publicly or call
    third-party APIs.
  - Each call's arguments are then scanned under the tool's name, so a
    `match.tool: post_*` rule can refuse it. A refused call returns an error
    `ToolMessage` to the model, and the tool never runs.
  - The check only refuses or lets the call through. It never rewrites the
    arguments, so a tool that is allowed to run still gets real values.

## LCEL chains

For chains that aren't agents, put a masking step before the model and a
restoring step after it, sharing one `Guard`:

```python
from virelia import Guard, Vault, load_preset
from virelia.integrations.langchain import masking_runnable, restoring_runnable

guard = Guard(load_preset("agent"), vault=Vault())
chain = prompt | masking_runnable(guard) | llm | restoring_runnable(guard)
```

`masking_runnable` accepts any of these inputs:
- a string;
- a message, or a list of messages;
- a `PromptValue`;
- a dict, which is scanned as data.

It raises `VireliaBlocked` when the input is withheld.

## Any other agent: `Guard`

The LangChain integration is a thin layer over `virelia.Guard`, which has no
dependency on any framework:

```python
from virelia import Guard, Vault, load_preset

guard = Guard(load_preset("agent"), vault=Vault())

masked = guard.mask_messages(messages, scope=conversation_id)
if masked.blocked:
    ...  # masked.block_reason says why; nothing was sent
reply = call_your_llm(masked.payload)
reply = guard.restore_messages([reply], scope=conversation_id)[0]

@guard.tool                       # rehydrates args, masks the result
def lookup_beneficiary(phone: str) -> dict: ...
```

`mask_messages` takes OpenAI-style message dicts. Content can be a string or a
list of `{"type": "text"}` blocks, and tool-call arguments are handled too.
Only text is scanned. Roles, ids, images and other non-text blocks pass
through untouched.

To support another framework, you only need to map its message type to and
from `virelia.guard.message_view`. `virelia/integrations/langchain.py` shows
how, in about 150 lines.

## Limitations

- **Streaming.** Tokens are restored once the model call returns. A streamed
  reply therefore shows `[EMAIL_1]` until the final message lands in state.
- **Only `tokenize` can be reversed.** `partial_mask` (`l***@example.org`),
  `mask` and `redact` are one-way by design.
- **Names in free text** need the `[ner]` or `[gliner]` extra plus a `PERSON_NAME` rule. See
  the [README](../README.md#-what-it-detects).
- **One vault connection per thread.** The guard keeps a vault handle open
  for every `thread_id` it has seen, until `middleware.guard.close()` is
  called. A long-running server with many threads should close the guard
  periodically, or set `scope_by_thread=False`.
- **Prompt injection is out of scope.** Virelia stops accidental exposure, not
  an adversarial model or tool. See the [threat model](threat-model.md).
