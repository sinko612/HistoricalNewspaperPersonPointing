#!/usr/bin/env python3
"""
Create smaller sampled variants of the PeopleGator Molmo JSONL dataset.
The script keeps page-level grouping stable and balances negative rows for selected task types.
"""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

NAME_RE = re.compile(r'person named "([^"]+)"')


def task_of(prompt: str) -> str:
    if "Determine whether" in prompt:
        return "is_person_present"
    if "Find the photograph of the person named" in prompt:
        return "point_person_by_name"
    if "Find ALL photographs of people" in prompt:
        return "point_all_person_photos"
    return "unknown"


def label_of(answer: str) -> str:
    return "neg" if answer.strip().lower().startswith("there are none") else "pos"


def target_name(prompt: str):
    m = NAME_RE.search(prompt)
    return m.group(1) if m else None


def row_key(row):
    return (
        row.get("image_rel_path", ""),
        row["messages"][0]["content"],
        row["messages"][1]["content"],
    )


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def count_rows(rows):
    c = Counter()
    for r in rows:
        p = r["messages"][0]["content"]
        a = r["messages"][1]["content"]
        c[(task_of(p), label_of(a))] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--drop-neg-name-without-positive", action="store_true")
    ap.add_argument("--drop-point-all", action="store_true")
    ap.add_argument(
        "--positive-ratio",
        type=float,
        default=1.0,
        help="Target positive:negative ratio for each name-conditioned task. Use 2.0 to push positives.",
    )
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    rows = load_rows(Path(args.input))
    print("Original rows:", len(rows), dict(count_rows(rows)))

    if args.dedupe:
        out = []
        seen = set()
        for r in rows:
            k = row_key(r)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        rows = out
        print("After dedupe:", len(rows), dict(count_rows(rows)))

    if args.drop_point_all:
        rows = [
            r
            for r in rows
            if task_of(r["messages"][0]["content"]) != "point_all_person_photos"
        ]
        print("After dropping point_all:", len(rows), dict(count_rows(rows)))

    if args.drop_neg_name_without_positive:
        positive_names = set()
        for r in rows:
            p = r["messages"][0]["content"]
            a = r["messages"][1]["content"]
            n = target_name(p)
            if n and label_of(a) == "pos":
                positive_names.add(n)

        kept = []
        dropped = 0
        dropped_names = Counter()
        for r in rows:
            p = r["messages"][0]["content"]
            a = r["messages"][1]["content"]
            n = target_name(p)
            if n and label_of(a) == "neg" and n not in positive_names:
                dropped += 1
                dropped_names[n] += 1
                continue
            kept.append(r)
        rows = kept
        print("Dropped neg rows whose name has no positive in this split:", dropped)
        print("Top dropped names:", dropped_names.most_common(20))
        print("After dropping harmful negatives:", len(rows), dict(count_rows(rows)))

    by_task_label = defaultdict(list)
    other = []
    for r in rows:
        t = task_of(r["messages"][0]["content"])
        l = label_of(r["messages"][1]["content"])
        if t in {"is_person_present", "point_person_by_name"}:
            by_task_label[(t, l)].append(r)
        else:
            other.append(r)

    final = list(other)
    for t in ["is_person_present", "point_person_by_name"]:
        pos = by_task_label[(t, "pos")]
        neg = by_task_label[(t, "neg")]
        target_pos = int(math.ceil(args.positive_ratio * len(neg))) if neg else len(pos)
        final.extend(neg)
        if len(pos) >= target_pos:
            sampled_pos = rnd.sample(pos, target_pos)
        else:
            sampled_pos = list(pos)
            sampled_pos.extend(rnd.choice(pos) for _ in range(target_pos - len(pos)))
        final.extend(sampled_pos)
        print(
            f"Task {t}: kept neg={len(neg)}, pos_original={len(pos)}, pos_written={len(sampled_pos)}"
        )

    rnd.shuffle(final)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Final rows:", len(final), dict(count_rows(final)))
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()
