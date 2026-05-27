#!/usr/bin/env python3
"""
Build name-conditioned Molmo instruction rows from PeopleGator face annotations.
The script creates positive and absent-name examples for person pointing and presence verification tasks.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".jp2", ".j2k"]

PROMPT_POINT_PERSON_BY_NAME = """
You are given a scanned historical newspaper page.
Find the photograph of the person named "{name}".
Point to the CENTER of that person's FACE.
Do not point to text, ornaments, drawings, or non-human figures.
Return EXACTLY one point in this format:
<point x=".." y=".." alt="{name}">{name}</point>
Coordinates must be 0-100 relative to the FULL page image.
If the person does not appear in a photograph on the page, return EXACTLY:
There are none.
""".strip()

PROMPT_IS_PERSON_PRESENT = """
You are given a scanned historical newspaper page.
Determine whether the person named "{name}" appears in a photograph on the page.
If yes, point to the CENTER of that person's FACE.
Do not point to text, ornaments, drawings, or non-human figures.
Return EXACTLY one of the following:

<point x=".." y=".." alt="{name}">{name}</point>

or

There are none.

Coordinates must be 0-100 relative to the FULL page image.
""".strip()


@dataclass(frozen=True)
class Ann:
    library: str
    document: str
    page_id: str
    page_filename: str
    person_name: str
    x1: float
    y1: float
    w: float
    h: float
    page_w: Optional[float]
    page_h: Optional[float]
    keypoints: tuple[tuple[float, float], ...]

    @property
    def x2(self) -> float:
        return self.x1 + self.w

    @property
    def y2(self) -> float:
        return self.y1 + self.h


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build Molmo fine-tuning JSONL from a PeopleGator JSONL for two name-aware tasks: "
            "point_person_by_name and is_person_present. Rows are compatible with train_molmo_peoplegator_lora.py."
        )
    )
    ap.add_argument(
        "--peoplegator-jsonl",
        required=True,
        help="Input PeopleGator JSONL, e.g. people_gator__corresponding_faces__2026-02-11.dev.jsonl",
    )
    ap.add_argument(
        "--project-root",
        required=True,
        help=(
            "Path to data root, directly to datasets/digiknihovna_data. "
            "The script uses it to verify/relativize image paths."
        ),
    )
    ap.add_argument("--out-train", required=True, help="Output train JSONL.")
    ap.add_argument(
        "--out-val",
        required=True,
        help="Output validation JSONL. Can be empty when --val-ratio 0.",
    )
    ap.add_argument(
        "--tasks",
        default="point_person_by_name,is_person_present",
        help="Comma-separated subset of: point_person_by_name,is_person_present",
    )
    ap.add_argument(
        "--negative-names-per-page-point",
        type=int,
        default=1,
        help=(
            "How many absent-person 'There are none.' examples to add per page for point_person_by_name. "
            "Use 0 for positive-only point training."
        ),
    )
    ap.add_argument(
        "--negative-names-per-page-present",
        type=int,
        default=1,
        help=(
            "How many absent-person 'There are none.' examples to add per page for is_person_present. "
            "Use 0 for positive-only presence training."
        ),
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Validation split ratio by PAGE, not by row. Default: 0.10.",
    )
    ap.add_argument(
        "--val-pages",
        type=int,
        default=None,
        help="Optional exact number of validation pages. Overrides --val-ratio.",
    )
    ap.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional smoke-test limit after shuffling pages.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--allow-missing-images",
        action="store_true",
        help=(
            "Write rows even if images cannot be found under --project-root. "
            "Coordinates are then computed from page_width/page_height in the JSONL. "
            "Training will still need the images to exist later."
        ),
    )
    ap.add_argument(
        "--include-metadata",
        action="store_true",
        help="Keep helper fields like _task and _target_person_name in the output JSONL.",
    )
    return ap.parse_args()


def normalize_text(text: object) -> str:
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_page_id(page: object) -> str:
    return Path(str(page)).stem


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        p = Path(p).expanduser()
        key = p.as_posix()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def data_roots(project_root: Path) -> list[Path]:
    project_root = project_root.expanduser().resolve()
    roots = [
        project_root / "datasets" / "digiknihovna_data",
        project_root / "digiknihovna_data",
        project_root,
    ]
    return unique_paths([p for p in roots if p.exists()])


def _candidate_files_for_page(
    directory: Path, page_filename: str, page_id: str
) -> list[Path]:
    page_filename = Path(page_filename).name
    candidates = []
    if page_filename:
        candidates.append(directory / page_filename)

    stem = Path(page_filename).stem if page_filename else page_id
    for ext in IMG_EXTS:
        candidates.append(directory / f"{stem}{ext}")

    return unique_paths(candidates)


def resolve_existing_image(
    project_root: Path, library: str, document: str, page_id: str, page_filename: str
) -> Optional[Path]:
    for root in data_roots(project_root):
        doc_dir = root / library / document
        if not doc_dir.exists():
            continue

        for directory in [doc_dir / "detections", doc_dir]:
            if not directory.exists():
                continue
            for cand in _candidate_files_for_page(directory, page_filename, page_id):
                if cand.exists() and cand.is_file():
                    return cand.resolve()

        stem = Path(page_filename).stem if page_filename else page_id
        for ext in IMG_EXTS:
            hits = list(doc_dir.rglob(f"{stem}{ext}"))
            if hits:
                return hits[0].resolve()
        for h in doc_dir.rglob(f"{stem}.*"):
            if h.is_file() and h.suffix.lower() in IMG_EXTS:
                return h.resolve()

    return None


def rel_paths_for_output(
    image_path: Optional[Path],
    project_root: Path,
    library: str,
    document: str,
    page_filename: str,
) -> tuple[str, Optional[str]]:
    project_root = project_root.expanduser().resolve()

    if image_path is not None:
        image_path = image_path.resolve()

        try:
            rel_project = image_path.relative_to(project_root).as_posix()
        except ValueError:
            rel_project = image_path.as_posix()

        rel_data = None
        for root in data_roots(project_root):
            try:
                candidate = image_path.relative_to(root.resolve()).as_posix()
                parts = Path(candidate).parts
                if len(parts) >= 3 and parts[0] == library and parts[1] == document:
                    rel_data = candidate
                    break
                if rel_data is None:
                    rel_data = candidate
            except ValueError:
                pass

        return rel_project, rel_data

    page_filename = Path(page_filename).name
    rel_data = f"{library}/{document}/detections/{page_filename}"
    rel_project = f"datasets/digiknihovna_data/{rel_data}"
    return rel_project, rel_data


def get_image_size(image_path: Optional[Path]) -> Optional[tuple[int, int]]:
    if image_path is None:
        return None

    try:
        from PIL import Image
    except ImportError:
        raise SystemExit(
            "Pillow is required for image size checks. Install with: pip install pillow"
        )

    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as exc:
        print(f"WARNING: could not read image {image_path}: {exc}", file=sys.stderr)
        return None


def parse_keypoints_from_flat(obj: dict) -> tuple[tuple[float, float], ...]:
    out: list[tuple[float, float]] = []
    for kp in obj.get("page_keypoints") or []:
        if isinstance(kp, (list, tuple)) and len(kp) >= 2:
            try:
                out.append((round(float(kp[0]), 3), round(float(kp[1]), 3)))
            except (TypeError, ValueError):
                pass
    return tuple(out)


def parse_keypoints_from_nested(face_info: dict) -> tuple[tuple[float, float], ...]:
    ids = sorted(
        int(m.group(1))
        for key in face_info.keys()
        for m in [re.match(r"kp_(\d+)_x$", str(key))]
        if m is not None
    )
    out: list[tuple[float, float]] = []
    for kp_id in ids:
        kx = face_info.get(f"kp_{kp_id}_x")
        ky = face_info.get(f"kp_{kp_id}_y")
        if kx is None or ky is None:
            continue
        try:
            out.append((round(float(kx), 3), round(float(ky), 3)))
        except (TypeError, ValueError):
            pass
    return tuple(out)


def _float_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_annotation(obj: dict) -> Optional[Ann]:
    if "document_info" in obj and "face_info" in obj:
        di = obj.get("document_info") or {}
        fi = obj.get("face_info") or {}

        library = normalize_text(di.get("library") or obj.get("library"))
        document = normalize_text(di.get("document") or obj.get("document"))
        page = normalize_text(di.get("page") or obj.get("page"))
        person_name = normalize_text(obj.get("person_name"))

        x1 = _float_or_none(fi.get("x") or fi.get("page_left"))
        y1 = _float_or_none(fi.get("y") or fi.get("page_top"))
        w = _float_or_none(fi.get("width"))
        h = _float_or_none(fi.get("height"))

        page_w = _float_or_none(di.get("page_width") or obj.get("page_width"))
        page_h = _float_or_none(di.get("page_height") or obj.get("page_height"))
        keypoints = parse_keypoints_from_nested(fi)
    else:
        library = normalize_text(obj.get("library"))
        document = normalize_text(obj.get("document"))
        page = normalize_text(obj.get("page"))
        person_name = normalize_text(obj.get("person_name"))

        x1 = _float_or_none(obj.get("page_left"))
        y1 = _float_or_none(obj.get("page_top"))
        w = _float_or_none(obj.get("width"))
        h = _float_or_none(obj.get("height"))

        page_w = _float_or_none(obj.get("page_width"))
        page_h = _float_or_none(obj.get("page_height"))
        keypoints = parse_keypoints_from_flat(obj)

    if not library or not document or not page or not person_name:
        return None
    if x1 is None or y1 is None or w is None or h is None:
        return None
    if w <= 0 or h <= 0:
        return None

    page_filename = Path(page).name
    if not Path(page_filename).suffix:
        page_filename = f"{normalize_page_id(page)}.jpg"

    return Ann(
        library=library,
        document=document,
        page_id=normalize_page_id(page),
        page_filename=page_filename,
        person_name=person_name,
        x1=x1,
        y1=y1,
        w=w,
        h=h,
        page_w=page_w,
        page_h=page_h,
        keypoints=keypoints,
    )


def load_grouped_annotations(
    jsonl_path: Path,
) -> tuple[dict[tuple[str, str, str], list[Ann]], Counter]:
    grouped: dict[tuple[str, str, str], list[Ann]] = defaultdict(list)
    stats: Counter = Counter()
    seen: set[tuple] = set()

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stats["raw_lines"] += 1
            line = line.strip()
            if not line:
                stats["blank_lines"] += 1
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json_lines"] += 1
                print(f"WARNING: bad JSON at line {line_no}", file=sys.stderr)
                continue

            ann = parse_annotation(obj)
            if ann is None:
                stats["invalid_annotations"] += 1
                continue

            dedupe_key = (
                ann.library,
                ann.document,
                ann.page_id,
                ann.person_name,
                round(ann.x1, 3),
                round(ann.y1, 3),
                round(ann.w, 3),
                round(ann.h, 3),
                ann.keypoints,
            )
            if dedupe_key in seen:
                stats["duplicate_annotations_skipped"] += 1
                continue
            seen.add(dedupe_key)

            grouped[(ann.library, ann.document, ann.page_id)].append(ann)
            stats["valid_annotations"] += 1

    return dict(grouped), stats


def ann_center_px(ann: Ann) -> tuple[float, float]:
    if ann.keypoints:
        x = sum(kp[0] for kp in ann.keypoints) / len(ann.keypoints)
        y = sum(kp[1] for kp in ann.keypoints) / len(ann.keypoints)
        return x, y
    return ann.x1 + ann.w / 2.0, ann.y1 + ann.h / 2.0


def point_answer(name: str, ann: Ann, image_w: int, image_h: int) -> str:
    x, y = ann_center_px(ann)
    x100 = max(0.0, min(100.0, x / float(image_w) * 100.0))
    y100 = max(0.0, min(100.0, y / float(image_h) * 100.0))

    safe_name = html.escape(name, quote=True)
    return f'<point x="{x100:.2f}" y="{y100:.2f}" alt="{safe_name}">{safe_name}</point>'


def sort_anns_top_left(anns: list[Ann]) -> list[Ann]:
    return sorted(
        anns, key=lambda a: (ann_center_px(a)[1], ann_center_px(a)[0], a.person_name)
    )


def sample_negative_names(
    name_pool: list[str],
    names_on_page: set[str],
    k: int,
    rng: random.Random,
) -> list[str]:
    if k <= 0:
        return []
    candidates = [name for name in name_pool if name not in names_on_page]
    rng.shuffle(candidates)
    return candidates[: min(k, len(candidates))]


def build_row(
    image_rel_path: str,
    image_rel_path_from_data_root: Optional[str],
    prompt: str,
    answer: str,
    include_metadata: bool,
    metadata: dict,
) -> dict:
    row = {
        "image_rel_path": image_rel_path,
        "image_rel_path_from_project_root": image_rel_path,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }
    if image_rel_path_from_data_root is not None:
        row["image_rel_path_from_data_root"] = image_rel_path_from_data_root

    if include_metadata:
        row.update(metadata)

    return row


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    peoplegator_jsonl = Path(args.peoplegator_jsonl).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()

    tasks = {x.strip() for x in args.tasks.split(",") if x.strip()}
    allowed_tasks = {"point_person_by_name", "is_person_present"}
    unknown = tasks - allowed_tasks
    if unknown:
        raise SystemExit(
            f"Unknown task(s): {sorted(unknown)}. Allowed: {sorted(allowed_tasks)}"
        )
    if not tasks:
        raise SystemExit("No tasks selected.")

    grouped, load_stats = load_grouped_annotations(peoplegator_jsonl)
    if not grouped:
        raise SystemExit("No valid PeopleGator annotations were loaded.")

    name_pool = sorted(
        {
            ann.person_name
            for anns in grouped.values()
            for ann in anns
            if ann.person_name
        }
    )
    page_keys = sorted(grouped.keys())
    rng.shuffle(page_keys)

    if args.max_pages is not None:
        page_keys = page_keys[: max(0, args.max_pages)]

    page_to_rows: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    stats: Counter = Counter(load_stats)
    image_size_mismatch_examples: list[
        tuple[str, tuple[int, int], tuple[Optional[float], Optional[float]]]
    ] = []

    for page_key in page_keys:
        anns = grouped[page_key]
        library, document, page_id = page_key
        page_filename = anns[0].page_filename

        image_path = resolve_existing_image(
            project_root, library, document, page_id, page_filename
        )
        image_size = get_image_size(image_path)

        if image_size is None:
            if not args.allow_missing_images:
                stats["pages_skipped_missing_or_unreadable_image"] += 1
                continue
            page_w = anns[0].page_w
            page_h = anns[0].page_h
            if page_w is None or page_h is None:
                stats["pages_skipped_missing_image_and_missing_page_size"] += 1
                continue
            image_size = (int(round(page_w)), int(round(page_h)))
            stats["pages_kept_without_image_check"] += 1
        else:
            stats["pages_with_readable_image"] += 1
            ann_page_w = anns[0].page_w
            ann_page_h = anns[0].page_h
            if ann_page_w and ann_page_h:
                if (
                    abs(image_size[0] - ann_page_w) > 2
                    or abs(image_size[1] - ann_page_h) > 2
                ):
                    stats["pages_with_image_size_different_from_json_size"] += 1
                    if len(image_size_mismatch_examples) < 5:
                        image_size_mismatch_examples.append(
                            (
                                f"{library}/{document}/{page_filename}",
                                image_size,
                                (ann_page_w, ann_page_h),
                            )
                        )

        image_w, image_h = image_size
        image_rel_path, image_rel_data = rel_paths_for_output(
            image_path, project_root, library, document, page_filename
        )

        anns_by_name: dict[str, list[Ann]] = defaultdict(list)
        for ann in anns:
            anns_by_name[ann.person_name].append(ann)

        names_on_page = set(anns_by_name.keys())

        for target_name in sorted(anns_by_name.keys()):
            target_anns = sort_anns_top_left(anns_by_name[target_name])
            target_ann = target_anns[0]
            answer = point_answer(target_name, target_ann, image_w, image_h)

            if "point_person_by_name" in tasks:
                prompt = PROMPT_POINT_PERSON_BY_NAME.format(name=target_name)
                page_to_rows[page_key].append(
                    build_row(
                        image_rel_path,
                        image_rel_data,
                        prompt,
                        answer,
                        args.include_metadata,
                        {
                            "_task": "point_person_by_name",
                            "_target_person_name": target_name,
                            "_expected_present": 1,
                            "_page_key": f"{library}/{document}/{page_id}",
                            "_source": "peoplegator",
                        },
                    )
                )
                stats["examples_point_person_by_name_positive"] += 1

            if "is_person_present" in tasks:
                prompt = PROMPT_IS_PERSON_PRESENT.format(name=target_name)
                page_to_rows[page_key].append(
                    build_row(
                        image_rel_path,
                        image_rel_data,
                        prompt,
                        answer,
                        args.include_metadata,
                        {
                            "_task": "is_person_present",
                            "_target_person_name": target_name,
                            "_expected_present": 1,
                            "_page_key": f"{library}/{document}/{page_id}",
                            "_source": "peoplegator",
                        },
                    )
                )
                stats["examples_is_person_present_positive"] += 1

        if "point_person_by_name" in tasks:
            neg_names = sample_negative_names(
                name_pool,
                names_on_page,
                args.negative_names_per_page_point,
                rng,
            )
            for neg_name in neg_names:
                prompt = PROMPT_POINT_PERSON_BY_NAME.format(name=neg_name)
                page_to_rows[page_key].append(
                    build_row(
                        image_rel_path,
                        image_rel_data,
                        prompt,
                        "There are none.",
                        args.include_metadata,
                        {
                            "_task": "point_person_by_name",
                            "_target_person_name": neg_name,
                            "_expected_present": 0,
                            "_page_key": f"{library}/{document}/{page_id}",
                            "_source": "peoplegator_negative_name_sample",
                        },
                    )
                )
                stats["examples_point_person_by_name_negative"] += 1

        if "is_person_present" in tasks:
            neg_names = sample_negative_names(
                name_pool,
                names_on_page,
                args.negative_names_per_page_present,
                rng,
            )
            for neg_name in neg_names:
                prompt = PROMPT_IS_PERSON_PRESENT.format(name=neg_name)
                page_to_rows[page_key].append(
                    build_row(
                        image_rel_path,
                        image_rel_data,
                        prompt,
                        "There are none.",
                        args.include_metadata,
                        {
                            "_task": "is_person_present",
                            "_target_person_name": neg_name,
                            "_expected_present": 0,
                            "_page_key": f"{library}/{document}/{page_id}",
                            "_source": "peoplegator_negative_name_sample",
                        },
                    )
                )
                stats["examples_is_person_present_negative"] += 1

    usable_page_keys = [k for k in page_keys if page_to_rows.get(k)]
    rng.shuffle(usable_page_keys)

    if args.val_pages is not None:
        n_val_pages = max(0, min(args.val_pages, len(usable_page_keys)))
    else:
        val_ratio = max(0.0, min(1.0, args.val_ratio))
        n_val_pages = int(round(len(usable_page_keys) * val_ratio))
        if val_ratio > 0 and n_val_pages == 0 and len(usable_page_keys) > 1:
            n_val_pages = 1

    val_keys = set(usable_page_keys[:n_val_pages])

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for key in usable_page_keys:
        rows = page_to_rows[key]
        if key in val_keys:
            val_rows.extend(rows)
        else:
            train_rows.extend(rows)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    write_jsonl(Path(args.out_train), train_rows)
    write_jsonl(Path(args.out_val), val_rows)

    print("Input:", peoplegator_jsonl)
    print("Project root:", project_root)
    print("Data roots considered:", [p.as_posix() for p in data_roots(project_root)])
    print("Selected tasks:", sorted(tasks))
    print("Unique names in pool:", len(name_pool))
    print("Pages loaded:", len(grouped))
    print("Pages used:", len(usable_page_keys))
    print("Validation pages:", len(val_keys))
    print("Train rows:", len(train_rows), "->", args.out_train)
    print("Val rows:", len(val_rows), "->", args.out_val)

    print("\nStats:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")

    if image_size_mismatch_examples:
        print(
            "\nWARNING: Some image sizes differ from page_width/page_height in JSONL. First examples:"
        )
        for rel, actual, json_size in image_size_mismatch_examples:
            print(f"  {rel}: actual={actual}, json={json_size}")

    if not train_rows:
        raise SystemExit(
            "No train rows were written. Check --project-root and image availability."
        )


if __name__ == "__main__":
    main()
