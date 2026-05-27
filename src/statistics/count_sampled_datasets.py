#!/usr/bin/env python3
"""
Vypočíta porovnateľné štatistiky pre pôvodný a samplované Molmo datasety.
Skript vypisuje tabuľku, LaTeX riadky a kontrolné počty podľa úloh.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

DEFAULT_FILES = [
    "molmo_peoplegator_dev_train.jsonl",
    "molmo_peoplegator_dev_train_2sampled.jsonl",
    "molmo_peoplegator_dev_train_3sampled.jsonl",
]


def variant_name(path: Path) -> str:
    name = path.name

    if name == "molmo_peoplegator_dev_train.jsonl":
        return "balanced"
    if "2sampled" in name:
        return "sampled_2"
    if "3sampled" in name:
        return "sampled_3"

    return path.stem


def is_positive(value) -> bool:
    return value in (1, "1", True, "true", "True")


def is_negative(value) -> bool:
    return value in (0, "0", False, "false", "False")


def format_int(value: int) -> str:
    return f"{value:,}".replace(",", r"\,")


def analyze_file(path: Path) -> dict:
    stats = {
        "variant": variant_name(path),
        "total": 0,
        "point_name_pos": 0,
        "point_name_neg": 0,
        "present_pos": 0,
        "present_neg": 0,
        "point_all": 0,
        "unique_pages": set(),
        "unique_names": set(),
        "task_counts": defaultdict(int),
        "warnings": [],
    }

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: neplatný JSON: {e}") from e

            stats["total"] += 1

            task = row.get("_task")
            expected_present = row.get("_expected_present")

            stats["task_counts"][task] += 1

            page_key = (
                row.get("_page_key")
                or row.get("image_rel_path_from_data_root")
                or row.get("image_rel_path")
            )
            if page_key:
                stats["unique_pages"].add(page_key)

            target_name = row.get("_target_person_name")
            if target_name:
                stats["unique_names"].add(str(target_name).strip())

            if task == "point_person_by_name":
                if is_positive(expected_present):
                    stats["point_name_pos"] += 1
                elif is_negative(expected_present):
                    stats["point_name_neg"] += 1
                else:
                    stats["warnings"].append(
                        f"{path}:{line_no}: neznáma hodnota _expected_present={expected_present!r}"
                    )

            elif task == "is_person_present":
                if is_positive(expected_present):
                    stats["present_pos"] += 1
                elif is_negative(expected_present):
                    stats["present_neg"] += 1
                else:
                    stats["warnings"].append(
                        f"{path}:{line_no}: neznáma hodnota _expected_present={expected_present!r}"
                    )

            elif task == "point_all_person_photos":
                stats["point_all"] += 1

            else:
                stats["warnings"].append(f"{path}:{line_no}: neznámy task {task!r}")

    stats["unique_pages_count"] = len(stats["unique_pages"])
    stats["unique_names_count"] = len(stats["unique_names"])

    return stats


def print_markdown_table(results: list[dict]) -> None:
    headers = [
        "Variant",
        "Spolu",
        "point_person_by_name poz.",
        "point_person_by_name neg.",
        "is_person_present poz.",
        "is_person_present neg.",
        "point_all",
        "Unik. strany",
        "Unik. mená",
    ]

    rows = []
    for r in results:
        rows.append(
            [
                r["variant"],
                r["total"],
                r["point_name_pos"],
                r["point_name_neg"],
                r["present_pos"],
                r["present_neg"],
                r["point_all"],
                r["unique_pages_count"],
                r["unique_names_count"],
            ]
        )

    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")


def print_latex_rows(results: list[dict]) -> None:
    print("\nLaTeX riadky do tabuľky:\n")

    for r in results:
        print(
            f"\\texttt{{{r['variant']}}} & "
            f"{format_int(r['total'])} & "
            f"{format_int(r['point_name_pos'])} & "
            f"{format_int(r['point_name_neg'])} & "
            f"{format_int(r['present_pos'])} & "
            f"{format_int(r['present_neg'])} & "
            f"{format_int(r['point_all'])} & "
            f"{format_int(r['unique_pages_count'])} & "
            f"{format_int(r['unique_names_count'])} \\\\"
        )


def print_checks(results: list[dict]) -> None:
    print("\nKontrola taskov:\n")

    for r in results:
        print(f"{r['variant']}:")
        for task, count in sorted(r["task_counts"].items()):
            print(f"  {task}: {count}")

        if r["warnings"]:
            print("  VAROVANIA:")
            for warning in r["warnings"][:10]:
                print(f"    {warning}")
            if len(r["warnings"]) > 10:
                print(f"    ... ďalších {len(r['warnings']) - 10} varovaní")

        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vyráta štatistiky pre základný a samplované Molmo JSONL datasety."
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=DEFAULT_FILES,
        help="Cesty k JSONL súborom. Ak nie sú zadané, použijú sa predvolené názvy.",
    )
    parser.add_argument(
        "--no-checks",
        action="store_true",
        help="Nevypíše kontrolný rozpis taskov.",
    )

    args = parser.parse_args()

    results = []
    for file_name in args.files:
        path = Path(file_name)
        if not path.exists():
            raise FileNotFoundError(f"Súbor neexistuje: {path}")
        results.append(analyze_file(path))

    print_markdown_table(results)
    print_latex_rows(results)

    if not args.no_checks:
        print_checks(results)


if __name__ == "__main__":
    main()
