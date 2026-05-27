#!/usr/bin/env python3
"""
Vypočíta základné štatistiky Molmo JSONL datasetov podľa úloh a splitov.
Skript rozlišuje pozitívne a negatívne príklady a vypisuje výsledky ako Markdown tabuľky.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TASK_ORDER = [
    "point_person_by_name",
    "is_person_present",
    "point_all_person_photos",
]


def assistant_answer(obj: dict[str, Any]) -> str:
    for msg in obj.get("messages", []):
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


def is_positive(obj: dict[str, Any]) -> bool:
    if "_expected_present" in obj:
        return bool(obj["_expected_present"])
    return assistant_answer(obj) != "There are none."


def read_stats(path: Path) -> dict[str, Counter]:
    stats: dict[str, Counter] = defaultdict(Counter)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Neplatný JSON na riadku {line_no} v súbore {path}: {e}"
                ) from e

            task = obj.get("_task", "<missing_task>")
            stats[task]["total"] += 1
            if is_positive(obj):
                stats[task]["positive"] += 1
            else:
                stats[task]["negative"] += 1
    return stats


def all_tasks(
    train_stats: dict[str, Counter], val_stats: dict[str, Counter]
) -> list[str]:
    known = [t for t in TASK_ORDER if t in train_stats or t in val_stats]
    extra = sorted((set(train_stats) | set(val_stats)) - set(TASK_ORDER))
    return known + extra


def print_markdown_table(
    train_stats: dict[str, Counter], val_stats: dict[str, Counter]
) -> None:
    tasks = all_tasks(train_stats, val_stats)
    print("| Úloha | Train | Val | Pozitívne spolu | Negatívne spolu | Spolu |")
    print("|---|---:|---:|---:|---:|---:|")
    totals = Counter()
    for task in tasks:
        train_total = train_stats[task]["total"]
        val_total = val_stats[task]["total"]
        pos = train_stats[task]["positive"] + val_stats[task]["positive"]
        neg = train_stats[task]["negative"] + val_stats[task]["negative"]
        total = train_total + val_total
        print(f"| `{task}` | {train_total} | {val_total} | {pos} | {neg} | {total} |")
        totals["train"] += train_total
        totals["val"] += val_total
        totals["positive"] += pos
        totals["negative"] += neg
        totals["total"] += total
    print(
        f"| **Spolu** | **{totals['train']}** | **{totals['val']}** | **{totals['positive']}** | **{totals['negative']}** | **{totals['total']}** |"
    )


def print_split_detail(
    train_stats: dict[str, Counter], val_stats: dict[str, Counter]
) -> None:
    tasks = all_tasks(train_stats, val_stats)
    print("\nDetail podľa splitu:")
    print("| Úloha | Train + | Train - | Val + | Val - |")
    print("|---|---:|---:|---:|---:|")
    totals = Counter()
    for task in tasks:
        tr_pos = train_stats[task]["positive"]
        tr_neg = train_stats[task]["negative"]
        va_pos = val_stats[task]["positive"]
        va_neg = val_stats[task]["negative"]
        print(f"| `{task}` | {tr_pos} | {tr_neg} | {va_pos} | {va_neg} |")
        totals["tr_pos"] += tr_pos
        totals["tr_neg"] += tr_neg
        totals["va_pos"] += va_pos
        totals["va_neg"] += va_neg
    print(
        f"| **Spolu** | **{totals['tr_pos']}** | **{totals['tr_neg']}** | **{totals['va_pos']}** | **{totals['va_neg']}** |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        required=True,
        type=Path,
        help="Cesta k molmo_peoplegator_dev_train.jsonl",
    )
    parser.add_argument(
        "--val",
        required=True,
        type=Path,
        help="Cesta k molmo_peoplegator_dev_val.jsonl",
    )
    args = parser.parse_args()

    train_stats = read_stats(args.train)
    val_stats = read_stats(args.val)
    print_markdown_table(train_stats, val_stats)
    print_split_detail(train_stats, val_stats)


if __name__ == "__main__":
    main()
