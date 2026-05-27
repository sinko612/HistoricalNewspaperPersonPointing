#!/usr/bin/env python3
"""
Overuje základné štatistiky a prekryvy medzi PeopleGator dev a test anotáciami.
Skript kontroluje počty knižníc, dokumentov, strán, mien a duplicitných JSON riadkov.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def unique_count(rows, key_fn):
    return len({key_fn(r) for r in rows})


def overlap_count(dev, test, key_fn):
    return len({key_fn(r) for r in dev} & {key_fn(r) for r in test})


def duplicate_report(rows):
    encoded = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows]
    counts = Counter(encoded)
    unique_exact = len(counts)
    duplicate_groups = sum(1 for c in counts.values() if c > 1)
    extra_duplicate_lines = sum(c - 1 for c in counts.values() if c > 1)
    return unique_exact, duplicate_groups, extra_duplicate_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()

    dev = load_jsonl(args.dev)
    test = load_jsonl(args.test)
    both = dev + test

    metrics = [
        (
            "Počet záznamov",
            len(dev),
            len(test),
            len(both),
        ),
        (
            "Počet knižníc",
            unique_count(dev, lambda r: r["library"]),
            unique_count(test, lambda r: r["library"]),
            unique_count(both, lambda r: r["library"]),
        ),
        (
            "Počet dokumentov",
            unique_count(dev, lambda r: (r["library"], r["document"])),
            unique_count(test, lambda r: (r["library"], r["document"])),
            unique_count(both, lambda r: (r["library"], r["document"])),
        ),
        (
            "Počet strán",
            unique_count(dev, lambda r: (r["library"], r["document"], r["page"])),
            unique_count(test, lambda r: (r["library"], r["document"], r["page"])),
            unique_count(both, lambda r: (r["library"], r["document"], r["page"])),
        ),
        (
            "Počet unikátnych hodnôt person_name",
            unique_count(dev, lambda r: r["person_name"]),
            unique_count(test, lambda r: r["person_name"]),
            unique_count(both, lambda r: r["person_name"]),
        ),
        (
            "Počet unikátnych hodnôt crop_name",
            unique_count(dev, lambda r: r["crop_name"]),
            unique_count(test, lambda r: r["crop_name"]),
            unique_count(both, lambda r: r["crop_name"]),
        ),
    ]

    print("| Metrika | Dev | Test | Spolu |")
    print("|---|---:|---:|---:|")
    for name, dev_count, test_count, total_count in metrics:
        print(f"| {name} | {dev_count} | {test_count} | {total_count} |")

    print("\nPrekryvy dev ∩ test:")
    print(f"knižnice: {overlap_count(dev, test, lambda r: r['library'])}")
    print(
        f"dokumenty: {overlap_count(dev, test, lambda r: (r['library'], r['document']))}"
    )
    print(
        f"strany: {overlap_count(dev, test, lambda r: (r['library'], r['document'], r['page']))}"
    )
    print(f"person_name: {overlap_count(dev, test, lambda r: r['person_name'])}")
    print(f"crop_name: {overlap_count(dev, test, lambda r: r['crop_name'])}")

    print("\nDuplicitné úplné JSON riadky:")
    for split_name, rows in [("dev", dev), ("test", test)]:
        unique_exact, duplicate_groups, extra_duplicates = duplicate_report(rows)
        print(
            f"{split_name}: {len(rows)} riadkov, "
            f"{unique_exact} unikátnych úplných JSON záznamov, "
            f"{duplicate_groups} duplicitných skupín, "
            f"{extra_duplicates} nadbytočných duplicitných riadkov"
        )


if __name__ == "__main__":
    main()
