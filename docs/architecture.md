# Architecture

## The big picture

virelia sits between an AI agent and the places its data comes from. Whenever a
payload passes through — a tool result, a prompt, tool arguments — virelia scans
it for personal data and rewrites it according to a policy *before* the model
ever sees it.

At the center is one **guardrail engine**. It knows nothing about MCP, Claude
Code, or any transport — it operates on plain Python values (dicts, lists,
strings, numbers). Thin **adapters** feed it payloads from each surface (an MCP
proxy, Claude Code hooks, your own Python code) and pass its verdict back. This
split is deliberate: the engine is testable in isolation, and adding a new
surface never means touching detection logic.

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

Reading the diagram left to right: a payload comes in, the **policy** decides
what should happen to each kind of entity, the **detector pipeline** finds
candidate personal data (in tiers, cheapest first), the **transformer** rewrites
the payload, and every finding is recorded as an **audit event** that contains a
hash of the matched value — never the value itself.

## A worked example

An MCP tool returns this row from a warehouse:

```json
{"name": "Lakshmi Devi", "phone": "+91 98765 43210", "aadhaar": "234567890124", "village": "Rampur"}
```

Under the `ngo-default` policy, the engine does the following:

1. The field name `aadhaar` matches a **structural detector** — flagged before
   any text is even read.
2. The value `234567890124` also matches the Aadhaar **regex detector**, and
   passes the Verhoeff checksum, so confidence is high. Policy says Aadhaar →
   `block`.
3. `+91 98765 43210` matches the phone detector; policy says phone →
   `partial_mask` (keep the last digits so the row stays recognizable).
4. The field name `name` triggers the column-name detector; policy says person
   names → `mask`.
5. `Rampur` matches nothing — it passes through untouched.

The model receives:

```json
{"name": "[MASKED:PERSON_NAME]", "phone": "+** ***** *3210", "aadhaar": "[BLOCKED:ngo-default/govt-financial-block]", "village": "Rampur"}
```

and the audit log gains one event per finding — entity type, the rule that
fired, confidence, the JSON path, and a salted HMAC of the original text, so an
auditor can later answer "what categories of data were caught?" without the log
itself leaking anything.

## The scan pipeline, step by step

This is `engine.py:_scan`. Two terms used throughout: a **leaf** is a single
scalar value inside the payload (one string or number, however deeply nested),
and a **span** is the specific stretch of characters inside a leaf that a
detector matched.

1. **Walk.** The engine traverses the payload tree and yields every leaf with
   its path (like `$.rows[0].phone`). Dictionary *keys* are never scanned as
   text; booleans and nulls are skipped.
2. **Tier 0 — structural policy rules.** Policy rules with a `match:` block
   name a tool and/or a JSON path (globs allowed, e.g. `get_beneficiar*`,
   `$.rows[*].notes`). A match pins the rule's action with confidence 1.0 — no
   text is read at all. A `block` rule with `scope: result` is the emergency
   brake: it refuses the entire payload and stops the scan immediately.
3. **Tier 0 — structural detectors.** `ColumnNameDetector` flags whole values
   by the field name holding them (`aadhaar`, `father_name`, …) — including
   Hindi/Hinglish field names (`नाम`, `फ़ोन`, `पता`, `naam`, `khata_no`, …).
   Without the NER tier installed, this is the only way free-text names and
   addresses get caught.
4. **Tier 1 — regex detectors.** Each leaf's text is first digit-normalized:
   every Unicode decimal digit (Devanagari ९, Bengali ৯, …) maps 1:1 to its
   ASCII twin, so the mapping is length-preserving and spans found on the
   normalized copy index correctly into the original string — which is what
   the transformer rewrites, keeping the original script outside (and, for
   partial masks, inside) the match. Each compiled pattern then runs over
   the normalized text (numeric leaves are stringified and, if matched,
   treated as wholly sensitive). Entities with checksums — Aadhaar
   (Verhoeff), cards (Luhn) — reject false candidates in `validate()` on the
   normalized ASCII digits: a random 12-digit number does not survive.
5. **Tier 2 — NER (optional).** When the `[ner]` extra is installed and the
   policy needs it, Presidio + spaCy scan string leaves (the same
   digit-normalized copy) for free-text names, locations, and India-context
   patterns. See [The NER tier](#the-ner-tier-ner-extra).
6. **Resolve.** Every span gets exactly one action, by precedence: a pinned
   structural rule wins; otherwise the first policy rule (in file order) listing
   that entity; otherwise the policy's `defaults.unknown_entity_action`. Spans
   below `defaults.min_confidence` are dropped.
7. **Transform.** Within a leaf, overlapping spans are clustered, and the
   cluster takes its most severe action (block > redact > tokenize > mask >
   partial_mask > allow) across the combined extent. `tokenize` swaps the span
   for a stable `[PERSON_NAME_1]` token from the local
   [vault](vault.md) (falling back to `mask` when no vault is configured).
   Replacements apply right-to-left so earlier offsets stay valid. The payload
   is rebuilt from scratch — the caller's object is never mutated.
8. **Audit.** One event per finding: entity type, detector tier, confidence,
   rule id, action, rendered path, character offsets, and a salted HMAC of the
   matched text. Plaintext never enters an event; the only code that touches
   the raw string is `audit.hmac_value()`.

## Why tiers?

Each tier catches what the previous one can't, and costs more than it:

| Tier | What it is | Catches | Cost |
|---|---|---|---|
| T0 structural | field names + JSON-path rules | known data shapes, free-text columns | ~zero (no text scanning) |
| T1 regex + checksum | compiled patterns, Verhoeff/Luhn validation | formatted IDs, emails, phones, secrets | microseconds per leaf |
| T2 NER (`[ner]` extra) | Presidio + spaCy, India-tuned recognizers | names/addresses/UPI IDs inside free text | model inference, opt-in |
| T3 LLM *(planned)* | local model judgment | context-dependent edge cases | highest, opt-in extra |

Research on PII detection consistently shows the failure modes are
complementary — regex lacks semantic understanding; NER has limited type
coverage — which is why virelia layers them instead of betting on either.

## The NER tier (`[ner]` extra)

```bash
uv tool install 'virelia[ner]'        # or: pip install 'virelia[ner]'
python -m spacy download en_core_web_sm  # any en_core_web_* model works
```

The base install is untouched: `detectors/ner.py` never imports presidio at
module level, and when the extra (or its spaCy model) is missing the loader
returns None and the scan runs exactly as before. The analyzer is built once
per process, on the first string leaf scanned, and reused across engines.

What it adds:

- **spaCy NER via Presidio** — PERSON → `PERSON_NAME`, LOCATION → `ADDRESS`,
  inside free text, not just behind a column name.
- **India-context recognizers** — Aadhaar/PAN/IFSC-shaped strings start below
  the confidence bar and get Presidio's context boost only when words like
  "aadhaar", "itr", or "neft" appear nearby. This catches e.g. an Aadhaar with
  a typo'd digit (which T1 rightly rejects on the Verhoeff check) sitting next
  to the word "aadhaar". UPI VPAs (`name@bank`) surface as `UPI_ID` only with
  payment context, so emails don't false-positive.
- **Indic name deny-list** — a small list of common transliterated Indian
  given names spaCy's English models often miss, matched case-sensitively.

Two calibration rules: T2 confidence is capped at 0.84, strictly below any
checksum-validated T1 match, and the policy gates loading via `defaults.ner` —
`auto` (default: load only when the policy acts on entities tiers 0–1 can't
find in free text: `PERSON_NAME`, `ADDRESS`, `UPI_ID`), `true` (always load if
installed), `false` (never). With `auto`, the `personal` and `dev` presets
keep the tier off; `ngo-default` and `strict` use it when installed.

One operational note: Claude Code hook processes are short-lived, so each
invocation pays the spaCy model load. The long-lived proxy (`virelia proxy`)
is the right surface for NER.

## The GLiNER backend (`[gliner]` extra)

```bash
pip install 'virelia[gliner]'   # gliner + torch, permissive licenses
```
```yaml
defaults:
  ner_backend: gliner
  ner_model: urchade/gliner_multi-v2.1   # default; or a local directory
```

[GLiNER](https://github.com/urchade/GLiNER) is an open-source NER model that
takes entity labels as plain text, so one model covers every label without
retraining. Virelia prompts it for `person`, `address` and `upi id` and maps
them to `PERSON_NAME`, `ADDRESS` and `UPI_ID`. Aadhaar, PAN and IFSC stay with
Tier 1, where checksums beat any model. It follows the Presidio backend's
contract: lazy import, one load per process, any failure disables the tier
instead of breaking the scan, and confidence capped at 0.84. The Indian
given-name list runs on this backend too, so switching backends never loses a
name the other one caught.

Two things differ from Presidio:

- **Input window.** GLiNER silently drops everything past ~384 words, so a
  name at the end of a long tool result would be missed. Long leaves are
  scanned in 200-word windows overlapping by 30 words, with offsets carried
  back into the original string.
- **Weights come from the Hugging Face Hub** on first load (about 1.2 GB for
  the default model, plus its mDeBERTa tokenizer). On a machine that must not make that call, download the
  model once and set `ner_model` to the local directory, or set
  `HF_HUB_OFFLINE=1` after a first run.

**What it does and doesn't buy you today.** `benchmarks/bench_ner.py` runs both
backends through the engine on 12 synthetic notes (English, transliterated
Hindi, Hindi, Marathi, one ~750-word note, and decoys):

| | names | addresses | UPI | decoys masked | per note |
|---|---|---|---|---|---|
| Presidio + `en_core_web_sm` | 6/9 | 1/2 | 1/1 | 0/10 | ~9 ms |
| GLiNER `gliner_multi_pii-v1` | 6/9 | 1/2 | 1/1 | 0/10 | ~90 ms |
| GLiNER `gliner_multi-v2.1` | 6/9 | 2/2 | 1/1 | 0/10 | ~110 ms |

All three miss most names written in Devanagari. The general multilingual
model (the default) is the only one that catches the Hindi address, which is
why it is the default over the PII-tuned one. Otherwise GLiNER is not more
accurate here, only ~10× slower; choose it for the open model and label
flexibility, not for Indic name recall. The set is a smoke test, not the published benchmark
(#9/#10). Indic-script names remain the open gap; ai4bharat/IndicNER (MIT) is
the next candidate to evaluate against the same script.

## Performance posture

Detection cost is policy-driven. With `unknown_entity_action: allow` (dev
posture), the engine prunes detectors down to the entities the policy actually
acts on. In every other posture all detectors run — an entity has to be *found*
before it can be default-masked. ("Fail closed": when the policy doesn't know an
entity, the safe default is to mask it, which requires looking for everything.)
Tier 0 is path checks; Tier 1 is compiled regex over leaf strings. The NER
tier is lazy-loaded behind the `[ner]` extra and gated by `defaults.ner`
(the LLM tier will follow the same pattern), so the base path stays
sub-millisecond.

## Adapters (shipped in 0.2)

An adapter is the glue between one surface and the engine. Every adapter has the
same three responsibilities: scan tool arguments before the tool runs, scan the
result before it returns, and surface `result.blocked` as a refusal the caller
understands.

- **`adapters/fastmcp_middleware.py`** — `VireliaMiddleware` for FastMCP 3.x
  servers and proxies (the `on_call_tool` hook; lazy-imported behind the
  `[proxy]` extra). It scans `structured_content` and text content blocks —
  text that looks like JSON is parsed first so structural path rules still
  apply. Blocked results return as `is_error` with the engine's block payload;
  if the engine itself crashes, the result is refused rather than passed
  through unscanned. ⚠️ Note: the FastMCP 1.x bundled inside the official
  `mcp` SDK (`mcp.server.fastmcp`) has **no middleware system** — servers built
  on it need a migration or a tool-wrapper shim.
- **`adapters/proxy.py` / `virelia proxy`** — a standalone gateway for upstream
  MCP servers you can't modify, built on fastmcp's `create_proxy`. With one
  upstream, tools are mirrored 1:1 under their original names; with several,
  fastmcp prefixes them `{server}_{tool}` (remember this when writing policy
  tool globs). In stdio mode stdout *is* the MCP protocol channel, so all
  logging goes to stderr.
- **`hooks/` + the Claude Code plugin** (`plugins/virelia/`) — three guards:
  the prompt guard (block + confirm-by-resend), the tool-input guard (ask or
  mask via `permissionDecision`/`updatedInput`), and the tool-output detector.
  The output guard is warn-only because Claude Code's PostToolUse hooks cannot
  rewrite a result — masking outputs is exactly what the proxy is for. Hooks
  fail open: if the `virelia` CLI is missing, the session is never broken. See
  [claude-plugin.md](claude-plugin.md).

## Open-source guarantee

Every tier, including planned ones, must be self-hostable with OSS components:
Presidio/spaCy for NER, local models (Ollama/vLLM) for the LLM tier. No feature
may require a proprietary service. See [CONTRIBUTING.md](../CONTRIBUTING.md)
for the dependency policy.
