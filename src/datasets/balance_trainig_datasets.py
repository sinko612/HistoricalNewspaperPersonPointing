#!/usr/bin/env python3
"""
Balance Molmo JSONL training rows by task and positive/negative label.
The script can optionally deduplicate rows and either downsample or oversample selected tasks.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def classify_task(prompt: str) -> str:
    if "Determine whether" in prompt:
        return "is_person_present"
    if "Find ALL photographs of people" in prompt:
        return "point_all_person_photos"
    if "Find the photograph of the person named" in prompt:
        return "point_person_by_name"
    return "unknown"


def classify_label(answer: str) -> str:
    return "neg" if answer.strip().lower().startswith("there are none") else "pos"


def task_and_label(row: dict) -> tuple[str, str]:
    prompt = row["messages"][0]["content"]
    answer = row["messages"][1]["content"]
    return classify_task(prompt), classify_label(answer)


def parse_task_list(value: str) -> set[str]:
    return {x.strip() for x in value.split(",") if x.strip()}


def counts(rows):
    c = Counter(task_and_label(r) for r in rows)
    return {f"{task}/{label}": n for (task, label), n in sorted(c.items())}


def main():
    ap = argparse.ArgumentParser(
        description="Balance PeopleGator Molmo JSONL by positive/negative labels inside selected tasks."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--dedupe",
        action="store_true",
        help="Remove exact duplicate image+prompt+answer rows before balancing.",
    )
    ap.add_argument(
        "--mode",
        choices=["downsample_majority", "oversample_minority"],
        default="downsample_majority",
        help="Validation/eval should normally use downsample_majority. Train can use either.",
    )
    ap.add_argument(
        "--balance-tasks",
        default="is_person_present,point_person_by_name",
        help="Comma-separated tasks to balance. Default balances the two name-conditioned tasks.",
    )
    ap.add_argument(
        "--keep-other-tasks",
        action="store_true",
        help="Keep tasks not listed in --balance-tasks unchanged, e.g. point_all_person_photos.",
    )
    args = ap.parse_args()

    balance_tasks = parse_task_list(args.balance_tasks)
    rng = random.Random(args.seed)

    rows = []
    seen = set()
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if args.dedupe:
            key = (
                row.get("image_rel_path"),
                row["messages"][0]["content"],
                row["messages"][1]["content"],
            )
            if key in seen:
                continue
            seen.add(key)
        rows.append(row)

    buckets = defaultdict(list)
    other_rows = []
    for row in rows:
        task, label = task_and_label(row)
        if task in balance_tasks:
            buckets[(task, label)].append(row)
        elif args.keep_other_tasks:
            other_rows.append(row)

    out = []
    for task in sorted(balance_tasks):
        pos = buckets[(task, "pos")]
        neg = buckets[(task, "neg")]
        if not pos or not neg:
            raise SystemExit(
                f"Cannot balance task {task}: pos={len(pos)}, neg={len(neg)}. "
                "Remove it from --balance-tasks or add --keep-other-tasks."
            )

        if args.mode == "downsample_majority":
            keep = min(len(pos), len(neg))
            out.extend(pos if len(pos) == keep else rng.sample(pos, keep))
            out.extend(neg if len(neg) == keep else rng.sample(neg, keep))
        else:
            keep = max(len(pos), len(neg))
            out.extend(
                pos if len(pos) == keep else [rng.choice(pos) for _ in range(keep)]
            )
            out.extend(
                neg if len(neg) == keep else [rng.choice(neg) for _ in range(keep)]
            )

    out.extend(other_rows)
    rng.shuffle(out)

    with open(args.output, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input_rows_after_optional_dedupe": len(rows),
                "output_rows": len(out),
                "balanced_tasks": sorted(balance_tasks),
                "kept_other_tasks": bool(args.keep_other_tasks),
                "input_counts": counts(rows),
                "output_counts": counts(out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
