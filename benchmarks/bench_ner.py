"""Compare NER backends end to end on synthetic free text.

    uv run --extra ner --extra gliner python benchmarks/bench_ner.py
    uv run --extra gliner python benchmarks/bench_ner.py --model knowledgator/gliner-pii-base-v1.0

Each backend runs through the real Engine (so the deny list, confidence cap and
min_confidence all apply). A value counts as caught when it no longer appears
in the scanned output; a decoy counts as damaged when it was masked away.
Backends whose extra isn't installed are skipped.

Next candidate: ai4bharat/IndicNER (MIT) — token classification, so it needs a
BIO-to-span adapter rather than GLiNER's label prompts.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from virelia import Engine, parse_policy

CASES = Path(__file__).with_name("ner_free_text.json")
FILLER_REPEAT = 30  # ~750 words, about twice GLiNER's input window


def load_cases() -> list[dict]:
    data = json.loads(CASES.read_text())
    filler = data["filler"] * FILLER_REPEAT
    return [{**c, "text": c["text"].replace("@@FILLER@@", filler)} for c in data["cases"]]


def engine_for(backend: str, model: str | None) -> Engine:
    policy = parse_policy(
        {
            "name": f"bench-{backend}",
            "defaults": {
                "unknown_entity_action": "allow",
                "ner": True,
                "ner_backend": backend,
                "ner_model": model,
            },
            "rules": [
                {"id": "ner", "entities": ["PERSON_NAME", "ADDRESS", "UPI_ID"], "action": "mask"}
            ],
        }
    )
    return Engine(policy, hmac_salt=b"bench")


def run(backend: str, model: str | None, cases: list[dict]) -> None:
    engine = engine_for(backend, model)
    t0 = time.perf_counter()
    engine.scan_result("warmup", {"notes": "warm up the model"})
    load_s = time.perf_counter() - t0
    if engine._ner is None:
        print(f"\n{backend}: not installed, skipped")
        return

    caught, total, damaged, decoys = (defaultdict(int) for _ in range(4))
    misses: list[str] = []
    elapsed = 0.0
    for case in cases:
        t0 = time.perf_counter()
        out = engine.scan_result("bench", {"notes": case["text"]}).payload["notes"]
        elapsed += time.perf_counter() - t0
        for entity, values in case["catch"].items():
            for value in values:
                total[entity] += 1
                if value in out:
                    misses.append(f"{case['lang']:>8} {entity}: {value}")
                else:
                    caught[entity] += 1
        for decoy in case["keep"]:
            decoys[case["lang"]] += 1
            damaged[case["lang"]] += decoy not in out

    label = backend + (f" ({model})" if model else "")
    print(f"\n{label}  load {load_s:.1f}s, {1000 * elapsed / len(cases):.0f} ms/case")
    for entity in sorted(total):
        print(f"  recall {entity:<12} {caught[entity]}/{total[entity]}")
    print(f"  decoys masked  {sum(damaged.values())}/{sum(decoys.values())}")
    for miss in misses:
        print(f"  missed {miss}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["presidio", "gliner", "all"], default="all")
    parser.add_argument("--model", help="GLiNER model id or local path")
    args = parser.parse_args()
    cases = load_cases()
    backends = ["presidio", "gliner"] if args.backend == "all" else [args.backend]
    for backend in backends:
        run(backend, args.model if backend == "gliner" else None, cases)


if __name__ == "__main__":
    main()
