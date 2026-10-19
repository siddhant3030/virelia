<div align="center">

```
██╗   ██╗ ██╗ ██████╗  ███████╗ ██╗      ██╗  █████╗ 
██║   ██║ ██║ ██╔══██╗ ██╔════╝ ██║      ██║ ██╔══██╗
██║   ██║ ██║ ██████╔╝ █████╗   ██║      ██║ ███████║
╚██╗ ██╔╝ ██║ ██╔══██╗ ██╔══╝   ██║      ██║ ██╔══██║
 ╚████╔╝  ██║ ██║  ██║ ███████╗ ███████╗ ██║ ██║  ██║
  ╚═══╝   ╚═╝ ╚═╝  ╚═╝ ╚══════╝ ╚══════╝ ╚═╝ ╚═╝  ╚═╝
```

### Privacy middleware for AI agents — any agent, any tool, any model.

Mask the personal data and secrets in **your agent's prompts, tool calls and
MCP traffic** before any of it reaches a third-party model — then restore the
real values for your tools and your users. Drop it into a LangChain agent, wrap
any Python function, or put it in front of any MCP server. Fully local,
reversible, no cloud in the loop.

[![CI](https://github.com/siddhant3030/virelia/actions/workflows/ci.yml/badge.svg)](https://github.com/siddhant3030/virelia/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/virelia)](https://pypi.org/project/virelia/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-276%20passing-brightgreen)](tests/)

**prompts · tool calls · MCP traffic** &nbsp;|&nbsp; detect → ask → mask &nbsp;|&nbsp; checksum-validated &nbsp;|&nbsp; hash-only audit &nbsp;|&nbsp; local-first

[Agent Middleware](#-drop-into-any-agent) ·
[Claude Code](#-claude-code-in-60-seconds) ·
[Masking Proxy](#-mask-mcp-tool-outputs-with-the-proxy) ·
[Docker](#-docker) ·
[Detection](#-what-it-detects) ·
[Policies](#-policies-are-yaml-not-code) ·
[Docs](#-documentation) ·
[Roadmap](#-roadmap)

<img src="docs/assets/demo.gif" width="700" alt="virelia scan masking a phone number and blocking an Aadhaar number">

</div>

---

You type your email into a Claude session, or an MCP tool dumps warehouse rows
with phone numbers into the model's context — and that data is now in a
third-party prompt log you don't control. **Virelia** sits at both doors — your
prompts and your tools' results — and swaps personal data for stand-ins before
it leaves your machine, keeping the key locally so you can swap the originals
back. Nothing personal leaves unmasked.

```
                ┌─────────────────────────────────────────────┐
                │             GUARDRAIL ENGINE                │
 MCP client ──▶ │  Policy      Detector        Transformer    │ ──▶ tool /
 (agent)        │  Resolver ─▶ Pipeline     ─▶ (mask/redact/  │     upstream server
            ◀── │  (rules,     T0 structural   partial/block) │ ◀──
                │  defaults)   T1 regex            │          │
                │              T2 NER*             ▼          │
                │              T3 LLM†        Audit events    │
                │                             (hash-only)     │
                └─────────────────────────────────────────────┘
                  * opt-in, behind the [ner] extra   † planned
```

## 🧩 Drop into any agent

```bash
pip install 'virelia[langchain,vault]'
```

```python
from langchain.agents import create_agent
from virelia import Vault
from virelia.integrations.langchain import VireliaMiddleware

agent = create_agent(model, tools, middleware=[VireliaMiddleware(vault=Vault())])
```

The model is the only party that sees stand-ins — your tools and your users
get the real values back:

```
user types        email lakshmi@example.org, aadhaar 2345 6789 0124
model sees        email [EMAIL_1], aadhaar [AADHAAR_1]
model calls       send_email(to="[EMAIL_1]")
tool receives     send_email(to="lakshmi@example.org")
user reads        Done — emailed lakshmi@example.org.
```

Tokens are stable per conversation thread, so the model can still reason
("the same person as before"), and API keys or tokens never reach the
provider at all. Not on LangChain? The framework-neutral `Guard` does the same
for any agent loop:

```python
from virelia import Guard, Vault, load_preset

guard = Guard(load_preset("agent"), vault=Vault())
masked = guard.mask_messages(messages)       # send masked.payload to any LLM API
reply = guard.restore_messages([reply])      # tokens → real values

@guard.tool                                  # any Python callable used as a tool
def send_sms(phone: str, body: str) -> dict: ...
```

Full guide — LCEL chains, thread scoping, blocking behavior:
[docs/langchain.md](docs/langchain.md).

## 🚀 Claude Code in 60 seconds

```bash
uv tool install virelia        # or: pip install virelia
```

Inside Claude Code:

```
/plugin marketplace add siddhant3030/virelia
/plugin install virelia@virelia
```

Now try it — type a prompt with an email in it:

```
> my email is sid@example.org, write me a signature

⛔ virelia blocked this prompt — it contains EMAIL (s***@example.org).

   Masked version you can copy:
   ┌────────────────────────────────────────────────────┐
   │ my email is s***@example.org, write me a signature │
   └────────────────────────────────────────────────────┘
   To send the original anyway, resend the exact same message within 5 minutes.
```

Three guards ship enabled:

| Guard | Where | Default behavior |
|---|---|---|
| 🛑 **Prompt guard** | your messages | block + masked copy + confirm-by-resend |
| ❓ **Tool-input guard** | MCP / Bash / WebFetch calls | native "share this anyway?" dialog |
| 📢 **Tool-output detector** | MCP results | warning + hash-only audit (mask via [`virelia proxy`](#-mask-mcp-tool-outputs-with-the-proxy)) |

Everything is configurable per-guard (`ask` / `mask` / `warn` / `off`) — see
[docs/claude-plugin.md](docs/claude-plugin.md). No `virelia` CLI installed →
the plugin stays silent and never breaks your session.

## 🛡️ Mask MCP tool *outputs* with the proxy

Claude Code hooks can't rewrite a tool result, so virelia ships an MCP gateway:
wrap any server you can't modify, tools are mirrored 1:1, and every result is
scrubbed before the model sees it.

```bash
uv tool install 'virelia[proxy]'       # the proxy extra pulls in fastmcp
```

**1. Move your real server into an upstreams file** — `~/.virelia/upstreams.json`,
the standard `.mcp.json` shape (env vars carry over unchanged):

```json
{
  "mcpServers": {
    "warehouse": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/warehouse-mcp", "warehouse-mcp"],
      "env": { "WAREHOUSE_API_URL": "http://localhost:8002" }
    }
  }
}
```

**2. Point your MCP client at the proxy instead** — Claude Code's `.mcp.json`,
Cursor, or any other client. Keep the same server key: with a single upstream,
tools are mirrored 1:1 under their original names, so nothing else about your
setup changes:

```json
{
  "mcpServers": {
    "warehouse": {
      "command": "virelia",
      "args": ["proxy",
               "--config", "/Users/you/.virelia/upstreams.json",
               "--policy", "ngo-default",
               "--audit", "/Users/you/.local/share/virelia/audit.jsonl"]
    }
  }
}
```

**3. Restart your session.** Every tool result now passes through the policy
before the model sees it:

```
tool returns                              model sees
{"name": "Lakshmi Devi",          →       {"name": "[MASKED:PERSON_NAME]",
 "phone": "+91 98765 43210",      →        "phone": "+** ***** *3210",
 "aadhaar": "2345 6789 0124",     →        "aadhaar": "[BLOCKED:ngo-default/govt-financial-block]",
 "village": "Rampur"}             →        "village": "Rampur"}
```

The model still reasons fine — counts, dates, villages intact. The person
doesn't leak. With multiple upstreams in one config, tools are prefixed
`{server}_{tool}` — mind your policy tool globs.

## 🐳 Docker

The proxy also ships as a container — useful when you'd rather not install
Python tooling on the host, or want the guardrail pinned in CI:

```bash
docker build -t virelia .
docker run -i --rm -v ~/.virelia:/config virelia \
  proxy --config /config/upstreams.json --policy ngo-default
```

Or point your MCP client straight at the container (releases publish
`ghcr.io/siddhant3030/virelia`):

```json
{
  "mcpServers": {
    "warehouse": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-v", "/Users/you/.virelia:/config",
               "ghcr.io/siddhant3030/virelia",
               "proxy", "--config", "/config/upstreams.json", "--policy", "ngo-default"]
    }
  }
}
```

`-i` matters: stdio is the MCP protocol channel. One caveat — upstream servers
defined as *commands* must exist inside the container, so either derive an image
that installs them (`FROM ghcr.io/siddhant3030/virelia` + your server) or use
URL-based upstreams.

## 🔍 What it detects

| Entity | Validation |
|---|---|
| EMAIL | format |
| PHONE (India + international) | digit boundaries |
| CREDIT_CARD | **Luhn checksum** |
| IP_ADDRESS | octet range |
| AADHAAR 🇮🇳 | **Verhoeff checksum** |
| PAN 🇮🇳 | holder-type check |
| IFSC 🇮🇳 | format |
| AWS / GitHub / JWT secrets | anchored formats |
| PERSON_NAME, ADDRESS, DOB, BANK_ACCOUNT, GOVT_ID | column-name heuristics (structured data, incl. Hindi: नाम, फ़ोन, पता, आधार, जन्मतिथि, खाता + romanizations) |
| PERSON_NAME, ADDRESS in *free text* + UPI_ID 🇮🇳 | NER tier — optional `[ner]` extra |

Digits in any script count — ९८७६५४३२१० is still a phone number. Unicode
decimal digits (Devanagari, Bengali, Tamil, Arabic-Indic, …) are normalized to
ASCII before patterns and checksums run, while masked output preserves the
original script: `संपर्क ९८७६५४३२१०` → `संपर्क ******३२१०`.

Checksums kill most false positives (a random 12-digit number is rejected
unless it passes Verhoeff). Try it:

```bash
$ virelia scan "call me at 9876543210 or lakshmi@example.org"
call me at ******3210 or l***@example.org

ENTITY           ACTION        RULE                     TIER CONF
PHONE            partial_mask  contact-partial          1    0.90
EMAIL            partial_mask  contact-partial          1    0.95
```

Names and addresses *inside free text* need the NER tier (Presidio + spaCy,
fully local, India-tuned recognizers — see
[docs/architecture.md](docs/architecture.md#the-ner-tier-ner-extra)):

```bash
uv tool install 'virelia[ner]'        # or: pip install 'virelia[ner]'
python -m spacy download en_core_web_sm
```

Without the extra the engine runs exactly as before — NER confidence is always
capped below a checksum-validated match, and `defaults.ner: false` turns the
tier off per policy.

## 📜 Policies are YAML, not code

```yaml
name: my-org
defaults:
  unknown_entity_action: mask        # fail closed
rules:
  - id: contact-partial
    entities: [EMAIL, PHONE]
    action: partial_mask             # l***@example.org, ******3210
  - id: ids-block
    entities: [AADHAAR, PAN, CREDIT_CARD]
    action: block
    message: "Government/financial IDs don't go to the model. Org policy."
  - id: notes-redact
    match: { tool: "get_beneficiar*", json_path: "$.rows[*].notes" }
    action: redact                   # structural rule — zero text scanning
```

Actions `allow | partial_mask | mask | tokenize | redact | block`,
first-match-wins, `scope: result` refuses a whole payload. Eight shipped
presets: `agent` (middleware default), `personal` (plugin default),
`ngo-default`, `strict`, `dev`, plus
draft regulation packs `dpdp`, `gdpr`, `hipaa-lite`
([docs/policy-packs.md](docs/policy-packs.md)).
Reference: [docs/policy-schema.md](docs/policy-schema.md). `tokenize` swaps a
value for a stable, locally-reversible `[PERSON_NAME_1]` token so the same
person stays the same person across rows — see [docs/vault.md](docs/vault.md)
and `virelia rehydrate`.

## 🧾 Audit you can show your auditor

Every detection is logged with entity type, detector tier, confidence, the
rule that fired, JSON path, offsets — and a **salted HMAC instead of the
value**. The log is safe to show to anyone cleared to see the redacted
output. *"Which categories of personal data left our infrastructure toward a
model provider this quarter?"* is one query. `/virelia:status` summarizes it
inside Claude Code.

## 🗺️ Where virelia sits

Of the four agent-guardrail stages — data collection, model training,
**agent tools & actions**, prompt/response — virelia covers stages 3 and 4
with one engine: inside the agent (LangChain middleware, the `Guard` API) and
at the MCP protocol layer, where structured tool traffic actually exists.
LLM gateways (Portkey, Kong) guard only stage 4, and framework PII filters are
regex lists without checksums, reversible tokens or an audit trail. Built for
sovereignty-sensitive deployments (NGOs under India's DPDP Act): fully
self-hosted, no vendor vault, no SaaS calls, every dependency permissive OSS.

## ⚠️ Honest limitations

- **NER is opt-in** — without the `[ner]` extra, free-text names/addresses
  are caught only via column-name heuristics; a name inside a paragraph gets
  through. With it, detection is English-model based (multilingual is on the
  roadmap) and statistical — don't let either tier create false confidence.
- Prompt guard can't *rewrite* prompts (Claude Code hook limitation) — it
  blocks and hands you the masked copy.
- Tool-output hook can't *unsend* — it warns; true masking needs the proxy.
- Agent middleware restores tokens after the model call returns, so a
  *streamed* reply shows tokens until the final message lands. Partial masks
  (`s***@example.org`) are one-way; only `tokenize` round-trips.
- Checksums shrink false positives, can't zero them; confidence is never 1.0.
- Protects against **accidental** exposure, not malicious servers or prompt
  injection: [docs/threat-model.md](docs/threat-model.md).

## 🐍 Engine API (raw payloads, 3 lines)

```bash
uv add virelia        # or: pip install virelia
```

```python
from virelia import Engine, load_preset

engine = Engine(load_preset("ngo-default"))
result = engine.scan(rows, source="get_beneficiaries", direction="result")
```

`result.payload` is the masked copy, `result.events` the audit trail, and
`result.blocked` says whether policy withheld it entirely.

## 📚 Documentation

| Doc | What's inside |
|---|---|
| [Architecture](docs/architecture.md) | engine internals, scan pipeline, performance posture |
| [LangChain & agents](docs/langchain.md) | agent middleware, `Guard`, LCEL runnables, trust boundary |
| [Claude Code plugin](docs/claude-plugin.md) | the three guards, per-guard config, proxy setup |
| [Policy schema](docs/policy-schema.md) | full YAML reference for rules, actions, matching |
| [Policy packs](docs/policy-packs.md) | draft DPDP / GDPR / HIPAA-lite presets compared |
| [Threat model](docs/threat-model.md) | what virelia defends against — and what it doesn't |
| [Demo script](docs/demo-script.md) | a 5-minute walkthrough you can run live |

## 🛠️ Development

```bash
uv sync && uv run pytest && uv run ruff check .
```

270+ tests including a golden corpus of synthetic, checksum-valid records
(no real personal data anywhere). Contributions welcome — especially
detectors for more ID systems and an India-tuned NER pack:
[CONTRIBUTING.md](CONTRIBUTING.md).

## 🧭 Roadmap

- [x] **NER tier** — Presidio + India-tuned recognizers (`[ner]` extra)
- [x] **Reversible tokenization vault** with rehydration at trusted sinks
- [x] **SQLite/Postgres audit sinks** + `virelia audit` CLI
- [x] **Agent middleware** — LangChain `create_agent` + framework-neutral `Guard`
- [ ] More frameworks — OpenAI Agents SDK, LlamaIndex, Pydantic AI
- [x] **Policy packs** — DPDP, GDPR, HIPAA-lite (draft, pending legal mapping in #15)
- [ ] **Multilingual** detection

## 📄 License

Apache-2.0 — code, presets, docs, corpus, everything. All dependencies are
permissively-licensed OSS and every feature is self-hostable; that's policy,
not accident ([CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md)).

---

<div align="center">

**If virelia guards your agents, a ⭐ helps others find it.**

[Report a bug](https://github.com/siddhant3030/virelia/issues) ·
[Request a feature](https://github.com/siddhant3030/virelia/issues) ·
[Security policy](SECURITY.md)

</div>
