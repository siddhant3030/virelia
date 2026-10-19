# Policy schema reference

Policies are YAML, validated with pydantic (`extra: forbid` — typos fail loudly at
load time, never silently). Load with `load_policy(path)`, `load_preset(name)`, or
`parse_policy(dict)`. A top-level `policy:` wrapper key is tolerated.

```yaml
name: ngo-default          # required
version: 1                 # optional, recorded in audit events via policy name
defaults:
  unknown_entity_action: mask   # action for detected entities no rule covers
  min_confidence: 0.4           # findings below this confidence are dropped
  ner: auto                     # NER tier ([ner] extra): auto | true | false
rules:
  - id: contact-partial         # required, unique
    entities: [EMAIL, PHONE]    # entity types this rule acts on
    action: partial_mask
  - id: aadhaar-block
    entities: [AADHAAR]
    action: block
    message: "Blocked by org policy."   # used in the [BLOCKED] refusal text
  - id: notes-redact
    match:                       # structural rule — fires on path alone, Tier 0
      tool: "get_beneficiar*"    # fnmatch glob on the tool name (default "*")
      json_path: "$.rows[*].notes"
    action: redact
  - id: secrets-kill
    entities: [GITHUB_TOKEN]
    action: block
    scope: result                # withhold the ENTIRE payload (block only)
```

## Semantics

- **First match wins, in file order.** The first rule whose `entities` contains the
  detected entity type decides the action; later rules never see it. Put the most
  specific/strict rules first.
- **Fail closed by default.** A detected entity covered by no rule gets
  `defaults.unknown_entity_action` (default `mask`) with `rule_id: null` in the audit
  event — so you can see exactly what your policy doesn't cover and tighten it.
  Setting it to `allow` also lets the engine skip detectors no rule acts on (dev only).
- A rule must have `entities`, `match`, or both. `match` rules pin their action with
  confidence 1.0 and no text scanning.
- `scope: result` is only valid with `action: block`.
- `defaults.ner` gates the optional NER tier: `auto` (default) loads it only when
  the policy acts on entities tiers 0–1 can't find in free text (`PERSON_NAME`,
  `ADDRESS`, `UPI_ID`); `true` always loads it when installed; `false` never.
  Without the `[ner]` extra the setting is inert.

## Actions

| Action | Effect on the value |
|--------|--------------------|
| `allow` | unchanged (still audited) |
| `partial_mask` | entity-specific template: phones/Aadhaar/cards keep last 4 digits, emails keep first char + domain; entities with no safe template fall back to a full mask |
| `mask` | `[MASKED:ENTITY_TYPE]` |
| `tokenize` | stable, reversible `[PERSON_NAME_1]`-style token from the local [vault](vault.md) — same value, same token. Falls back to `mask` (with a warning) when no vault is configured |
| `redact` | `[REDACTED]` |
| `block` | `[BLOCKED:policy/rule-id]`, or the whole payload refused with `scope: result` |

When overlapping detections disagree, the highest-severity action wins:
`block > redact > tokenize > mask > partial_mask > allow`. `tokenize` sits
above `mask` because a token hides just as many characters while losing
nothing (the value is vaulted); it sits below `redact`/`block` because a
reversible token must never override an explicitly irreversible action.

## json_path grammar

Deliberately minimal — matched against concrete leaf paths, validated at load time:

| Syntax | Meaning |
|--------|---------|
| `$` | root (required prefix) |
| `.key` | literal object key |
| `.*` | any object key |
| `[3]` | literal array index |
| `[*]` | any array index |
| `..key` | recursive descent (any depth), then key |

Examples: `$.rows[*].aadhaar`, `$..phone`, `$.content[0].text`.

## Built-in entity types

Tier 1 (regex, value-based): `EMAIL`, `PHONE`, `AADHAAR` (Verhoeff-gated), `PAN`,
`IFSC`, `CREDIT_CARD` (Luhn-gated), `IP_ADDRESS`, `AWS_ACCESS_KEY`, `GITHUB_TOKEN`,
`JWT`.

Tier 0 (column-name heuristics, additionally): `PERSON_NAME`, `ADDRESS`, `DOB`,
`BANK_ACCOUNT`, `GOVT_ID`.

Custom detectors passed via `Engine(extra_detectors=...)` add their entity types;
declare them to the loader with `load_policy(path, extra_entities=["MY_ENTITY"])`.

## Shipped presets

- **agent** — the agent-middleware default: contact details and IDs tokenized
  (reversible with a vault), card/bank numbers withheld, credentials refuse
  the whole message, unknown → allow. See [langchain.md](langchain.md).
- **ngo-default** — contact info partial-masked, government/financial IDs and
  credentials blocked, names masked, addresses/DOB redacted, unknown → mask.
- **strict** — credentials withhold the whole result; all known PII blocked;
  unknown → block.
- **dev** — only credentials and Aadhaar masked; everything else passes. Not for
  production.
- **dpdp**, **gdpr**, **hipaa-lite** — draft regulation packs (conservative
  defaults, not legal advice); compared entity-by-entity in
  [policy-packs.md](policy-packs.md).
