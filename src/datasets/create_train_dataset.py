#!/usr/bin/env python3
"""
Build the main Molmo training and validation JSONL files from PeopleGator annotations.
The script prepares name-conditioned tasks and page-level all-person pointing examples with identity-based splitting.
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

PROMPT_POINT_ALL_PERSON_PHOTOS = """
You are given a scanned historical newspaper page.
Find ALL photographs of people on the page.
For each photograph, point to the CENTER of one visible FACE.
Do not point to text, ornaments, drawings, or non-human figures.
Return ONLY points in this format:
<points x1=".." y1=".." x2=".." y2=".." ... alt="person photographs">person photographs</points>
Order the points from top to bottom, then left to right.
Coordinates must be 0-100 relative to the FULL page image.
If there are no photographs of people on the page, return EXACTLY:
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
    person_name: Optional[str]
    x1: float
    y1: float
    w: float
    h: float
    page_w: Optional[float]
    page_h: Optional[float]
    keypoints: tuple[tuple[float, float], ...]
    source: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.library, self.document, self.page_id)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build Molmo train/validation JSONL from PeopleGator dev annotations. "
            "The test split and all detections are used only as safety context for name pools, "
            "negative sampling, and the point_all_person_photos task; no positive test rows are trained."
        )
    )
    ap.add_argument(
        "--peoplegator-jsonl",
        required=True,
        help="DEV corresponding_faces JSONL used as the only source of name-positive training examples.",
    )
    ap.add_argument(
        "--peoplegator-test-jsonl",
        default=None,
        help="TEST corresponding_faces JSONL used only for global identity/page-presence checks.",
    )
    ap.add_argument(
        "--all-detections-jsonl",
        default=None,
        help="people_gator__detections.jsonl used for point_all_person_photos GT points.",
    )
    ap.add_argument(
        "--project-root",
        required=True,
        help="Path to ~/diplomka, ~/diplomka/datasets, or ~/diplomka/datasets/digiknihovna_data.",
    )
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument(
        "--out-stats", default=None, help="Optional JSON stats report path."
    )
    ap.add_argument(
        "--tasks",
        default="point_person_by_name,is_person_present,point_all_person_photos",
        help="Comma-separated subset of: point_person_by_name,is_person_present,point_all_person_photos",
    )
    ap.add_argument(
        "--negative-names-per-positive-point",
        type=int,
        default=2,
        help="Absent-name negatives for PROMPT_POINT_PERSON_BY_NAME per DEV positive annotation row.",
    )
    ap.add_argument(
        "--negative-names-per-positive-present",
        type=int,
        default=2,
        help="Absent-name negatives for PROMPT_IS_PERSON_PRESENT per DEV positive annotation row.",
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Validation ratio by identity/person name, not by page or row.",
    )
    ap.add_argument(
        "--val-identities",
        type=int,
        default=None,
        help="Exact number of validation identities; overrides --val-ratio.",
    )
    ap.add_argument(
        "--max-dev-rows",
        type=int,
        default=None,
        help="Optional smoke-test limit on DEV positive annotation rows after shuffle.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-missing-images", action="store_true")
    ap.add_argument("--include-metadata", action="store_true")
    ap.add_argument(
        "--all-person-split-policy",
        choices=["train", "val_if_any_val_identity", "skip_if_mixed"],
        default="val_if_any_val_identity",
        help=(
            "Where to place page-level point_all_person_photos rows. "
            "val_if_any_val_identity keeps pages containing validation identities out of train for this task."
        ),
    )
    ap.add_argument(
        "--max-point-all-detections",
        type=int,
        default=8,
        help=(
            "Maximum number of detections/faces allowed for one point_all_person_photos row. "
            "Pages with more detections are skipped so the <points ...> answer stays short enough for training. "
            "Use 0 to disable this filter."
        ),
    )
    return ap.parse_args()


def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", "" if text is None else str(text)).strip()


def normalize_name(text: object) -> Optional[str]:
    name = normalize_text(text)
    return name or None


def normalize_page_id(page: object) -> str:
    return Path(str(page)).stem


def _float_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
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


def parse_annotation(obj: dict, *, source: str, require_name: bool) -> Optional[Ann]:
    if "document_info" in obj and "face_info" in obj:
        di = obj.get("document_info") or {}
        fi = obj.get("face_info") or {}
        library = normalize_text(di.get("library") or obj.get("library"))
        document = normalize_text(di.get("document") or obj.get("document"))
        page = normalize_text(di.get("page") or obj.get("page"))
        person_name = normalize_name(obj.get("person_name"))
        x1 = _float_or_none(
            fi.get("x") if fi.get("x") is not None else fi.get("page_left")
        )
        y1 = _float_or_none(
            fi.get("y") if fi.get("y") is not None else fi.get("page_top")
        )
        w = _float_or_none(fi.get("width"))
        h = _float_or_none(fi.get("height"))
        page_w = _float_or_none(di.get("page_width") or obj.get("page_width"))
        page_h = _float_or_none(di.get("page_height") or obj.get("page_height"))
        keypoints = parse_keypoints_from_nested(fi)
    else:
        library = normalize_text(obj.get("library"))
        document = normalize_text(obj.get("document"))
        page = normalize_text(obj.get("page"))
        person_name = normalize_name(obj.get("person_name"))
        x1 = _float_or_none(obj.get("page_left"))
        y1 = _float_or_none(obj.get("page_top"))
        w = _float_or_none(obj.get("width"))
        h = _float_or_none(obj.get("height"))
        page_w = _float_or_none(obj.get("page_width"))
        page_h = _float_or_none(obj.get("page_height"))
        keypoints = parse_keypoints_from_flat(obj)

    if not library or not document or not page:
        return None
    if require_name and not person_name:
        return None
    if x1 is None or y1 is None or w is None or h is None or w <= 0 or h <= 0:
        return None

    page_filename = Path(page).name
    if not Path(page_filename).suffix:
        page_filename = f"{normalize_page_id(page)}.jpg"

    return Ann(
        library,
        document,
        normalize_page_id(page),
        page_filename,
        person_name,
        x1,
        y1,
        w,
        h,
        page_w,
        page_h,
        keypoints,
        source,
    )


def load_annotations(
    jsonl_path: Path, *, source: str, require_name: bool, dedupe: bool = True
) -> tuple[list[Ann], Counter]:
    rows: list[Ann] = []
    stats: Counter = Counter()
    seen: set[tuple] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stats[f"{source}_raw_lines"] += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats[f"{source}_bad_json_lines"] += 1
                print(
                    f"WARNING: bad JSON in {jsonl_path} at line {line_no}",
                    file=sys.stderr,
                )
                continue
            ann = parse_annotation(obj, source=source, require_name=require_name)
            if ann is None:
                stats[f"{source}_invalid_annotations"] += 1
                continue
            dkey = (
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
            if dedupe and dkey in seen:
                stats[f"{source}_duplicate_annotations_skipped"] += 1
                continue
            seen.add(dkey)
            rows.append(ann)
            stats[f"{source}_valid_annotations"] += 1
    return rows, stats


def unique_paths(paths: list[Path]) -> list[Path]:
    out, seen = [], set()
    for p in paths:
        key = p.expanduser().as_posix()
        if key not in seen:
            out.append(p.expanduser())
            seen.add(key)
    return out


def data_roots(project_root: Path) -> list[Path]:
    root = project_root.expanduser().resolve()
    return unique_paths(
        [
            p
            for p in [
                root / "datasets" / "digiknihovna_data",
                root / "digiknihovna_data",
                root,
            ]
            if p.exists()
        ]
    )


def resolve_existing_image(project_root: Path, ann: Ann) -> Optional[Path]:
    for root in data_roots(project_root):
        doc_dir = root / ann.library / ann.document
        if not doc_dir.exists():
            continue
        stem = Path(ann.page_filename).stem or ann.page_id
        candidates = [
            doc_dir / "detections" / ann.page_filename,
            doc_dir / ann.page_filename,
        ]
        for ext in IMG_EXTS:
            candidates.extend(
                [doc_dir / "detections" / f"{stem}{ext}", doc_dir / f"{stem}{ext}"]
            )
        for cand in unique_paths(candidates):
            if cand.exists() and cand.is_file():
                return cand.resolve()
        for h in doc_dir.rglob(f"{stem}.*"):
            if h.is_file() and h.suffix.lower() in IMG_EXTS:
                return h.resolve()
    return None


def get_image_size(image_path: Optional[Path]) -> Optional[tuple[int, int]]:
    if image_path is None:
        return None
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.size
    except Exception as exc:
        print(f"WARNING: could not read image {image_path}: {exc}", file=sys.stderr)
        return None


def rel_paths_for_output(
    image_path: Optional[Path], project_root: Path, ann: Ann
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
                if (
                    len(parts) >= 3
                    and parts[0] == ann.library
                    and parts[1] == ann.document
                ):
                    rel_data = candidate
                    break
                if rel_data is None:
                    rel_data = candidate
            except ValueError:
                pass
        return rel_project, rel_data
    rel_data = f"{ann.library}/{ann.document}/detections/{ann.page_filename}"
    return f"datasets/digiknihovna_data/{rel_data}", rel_data


def ann_center_px(ann: Ann) -> tuple[float, float]:
    if ann.keypoints:
        return (
            sum(x for x, _ in ann.keypoints) / len(ann.keypoints),
            sum(y for _, y in ann.keypoints) / len(ann.keypoints),
        )
    return ann.x1 + ann.w / 2.0, ann.y1 + ann.h / 2.0


def one_point_answer(name: str, ann: Ann, image_w: int, image_h: int) -> str:
    x, y = ann_center_px(ann)
    x100 = max(0.0, min(100.0, x / float(image_w) * 100.0))
    y100 = max(0.0, min(100.0, y / float(image_h) * 100.0))
    safe = html.escape(name, quote=True)
    return f'<point x="{x100:.2f}" y="{y100:.2f}" alt="{safe}">{safe}</point>'


def all_points_answer(anns: list[Ann], image_w: int, image_h: int) -> str:
    if not anns:
        return "There are none."
    attrs: list[str] = []
    for i, ann in enumerate(
        sorted(anns, key=lambda a: (ann_center_px(a)[1], ann_center_px(a)[0])), start=1
    ):
        x, y = ann_center_px(ann)
        x100 = max(0.0, min(100.0, x / float(image_w) * 100.0))
        y100 = max(0.0, min(100.0, y / float(image_h) * 100.0))
        attrs.append(f'x{i}="{x100:.2f}" y{i}="{y100:.2f}"')
    return f'<points {" ".join(attrs)} alt="person photographs">person photographs</points>'


def build_row(
    image_rel_path: str,
    image_rel_data: Optional[str],
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
    if image_rel_data is not None:
        row["image_rel_path_from_data_root"] = image_rel_data
    if include_metadata:
        row.update(metadata)
    return row


def sample_negative_names(
    pool: list[str],
    names_present_on_page: set[str],
    forbidden_split_names: set[str],
    k: int,
    rng: random.Random,
) -> list[str]:
    if k <= 0:
        return []
    candidates = [
        n
        for n in pool
        if n not in names_present_on_page and n not in forbidden_split_names
    ]
    rng.shuffle(candidates)
    return candidates[: min(k, len(candidates))]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    project_root = Path(args.project_root).expanduser().resolve()
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    allowed = {"point_person_by_name", "is_person_present", "point_all_person_photos"}
    unknown = tasks - allowed
    if unknown:
        raise SystemExit(
            f"Unknown task(s): {sorted(unknown)}. Allowed: {sorted(allowed)}"
        )

    dev_path = Path(args.peoplegator_jsonl).expanduser().resolve()
    dev_anns, stats = load_annotations(
        dev_path, source="dev", require_name=True, dedupe=False
    )
    if args.max_dev_rows is not None:
        rng.shuffle(dev_anns)
        dev_anns = dev_anns[: max(0, args.max_dev_rows)]
        stats["dev_rows_after_max_dev_rows"] = len(dev_anns)
    if not dev_anns:
        raise SystemExit("No valid DEV annotations were loaded.")

    named_context_anns = list(dev_anns)
    if args.peoplegator_test_jsonl:
        test_anns, test_stats = load_annotations(
            Path(args.peoplegator_test_jsonl).expanduser().resolve(),
            source="test",
            require_name=True,
            dedupe=True,
        )
        named_context_anns.extend(test_anns)
        stats.update(test_stats)
    else:
        test_anns = []

    all_detection_anns: list[Ann] = []
    if args.all_detections_jsonl:
        all_detection_anns, det_stats = load_annotations(
            Path(args.all_detections_jsonl).expanduser().resolve(),
            source="detections",
            require_name=False,
            dedupe=True,
        )
        stats.update(det_stats)

    names_present_by_page: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for ann in named_context_anns:
        if ann.person_name:
            names_present_by_page[ann.key].add(ann.person_name)

    dev_names = sorted({ann.person_name for ann in dev_anns if ann.person_name})
    global_names = sorted(
        {ann.person_name for ann in named_context_anns if ann.person_name}
    )
    if not global_names:
        raise SystemExit("No identities available for sampling.")

    rng.shuffle(dev_names)
    if args.val_identities is not None:
        n_val = max(0, min(args.val_identities, len(dev_names)))
    else:
        val_ratio = max(0.0, min(1.0, args.val_ratio))
        n_val = int(round(len(dev_names) * val_ratio))
        if val_ratio > 0 and n_val == 0 and len(dev_names) > 1:
            n_val = 1
    val_names = set(dev_names[:n_val])
    train_names = set(dev_names[n_val:])

    detections_by_page: dict[tuple[str, str, str], list[Ann]] = defaultdict(list)
    if all_detection_anns:
        for ann in all_detection_anns:
            detections_by_page[ann.key].append(ann)
    else:
        for ann in named_context_anns:
            detections_by_page[ann.key].append(ann)
            stats["point_all_person_photos_used_named_annotations_fallback"] += 1

    rows_by_split: dict[str, list[dict]] = {"train": [], "val": []}
    image_cache: dict[
        tuple[str, str, str], tuple[tuple[int, int], str, Optional[str], Ann]
    ] = {}
    skipped_pages_for_image = set()

    def page_image_info(
        ann: Ann,
    ) -> Optional[tuple[tuple[int, int], str, Optional[str]]]:
        if ann.key in image_cache:
            image_size, rel_project, rel_data, _ = image_cache[ann.key]
            return image_size, rel_project, rel_data
        image_path = resolve_existing_image(project_root, ann)
        image_size = get_image_size(image_path)
        if image_size is None:
            if not args.allow_missing_images:
                skipped_pages_for_image.add(ann.key)
                return None
            if ann.page_w is None or ann.page_h is None:
                skipped_pages_for_image.add(ann.key)
                return None
            image_size = (int(round(ann.page_w)), int(round(ann.page_h)))
            stats["pages_kept_without_image_check"] += 1
        else:
            stats["pages_with_readable_image"] += 1
        rel_project, rel_data = rel_paths_for_output(image_path, project_root, ann)
        image_cache[ann.key] = (image_size, rel_project, rel_data, ann)
        return image_size, rel_project, rel_data

    for idx, ann in enumerate(dev_anns):
        assert ann.person_name is not None
        info = page_image_info(ann)
        if info is None:
            stats["dev_positive_rows_skipped_missing_or_unreadable_image"] += 1
            continue
        (image_w, image_h), rel_project, rel_data = info
        split = "val" if ann.person_name in val_names else "train"
        answer = one_point_answer(ann.person_name, ann, image_w, image_h)
        page_names = names_present_by_page.get(ann.key, set())
        forbidden = train_names if split == "val" else val_names
        split_pool = sorted(val_names if split == "val" else train_names)
        split_pool = sorted(set(split_pool) | (set(global_names) - set(dev_names)))

        if "point_person_by_name" in tasks:
            rows_by_split[split].append(
                build_row(
                    rel_project,
                    rel_data,
                    PROMPT_POINT_PERSON_BY_NAME.format(name=ann.person_name),
                    answer,
                    args.include_metadata,
                    {
                        "_task": "point_person_by_name",
                        "_target_person_name": ann.person_name,
                        "_expected_present": 1,
                        "_split_by_identity": split,
                        "_source": "dev_positive_annotation_row",
                        "_dev_row_index": idx,
                        "_page_key": f"{ann.library}/{ann.document}/{ann.page_id}",
                    },
                )
            )
            stats[f"{split}_point_person_by_name_positive"] += 1
            for neg_name in sample_negative_names(
                split_pool,
                page_names,
                forbidden,
                args.negative_names_per_positive_point,
                rng,
            ):
                rows_by_split[split].append(
                    build_row(
                        rel_project,
                        rel_data,
                        PROMPT_POINT_PERSON_BY_NAME.format(name=neg_name),
                        "There are none.",
                        args.include_metadata,
                        {
                            "_task": "point_person_by_name",
                            "_target_person_name": neg_name,
                            "_expected_present": 0,
                            "_split_by_identity": split,
                            "_source": "absent_name_negative_checked_against_dev_test_page_names",
                            "_positive_dev_row_index_anchor": idx,
                            "_page_key": f"{ann.library}/{ann.document}/{ann.page_id}",
                        },
                    )
                )
                stats[f"{split}_point_person_by_name_negative"] += 1

        if "is_person_present" in tasks:
            rows_by_split[split].append(
                build_row(
                    rel_project,
                    rel_data,
                    PROMPT_IS_PERSON_PRESENT.format(name=ann.person_name),
                    answer,
                    args.include_metadata,
                    {
                        "_task": "is_person_present",
                        "_target_person_name": ann.person_name,
                        "_expected_present": 1,
                        "_split_by_identity": split,
                        "_source": "dev_positive_annotation_row",
                        "_dev_row_index": idx,
                        "_page_key": f"{ann.library}/{ann.document}/{ann.page_id}",
                    },
                )
            )
            stats[f"{split}_is_person_present_positive"] += 1
            for neg_name in sample_negative_names(
                split_pool,
                page_names,
                forbidden,
                args.negative_names_per_positive_present,
                rng,
            ):
                rows_by_split[split].append(
                    build_row(
                        rel_project,
                        rel_data,
                        PROMPT_IS_PERSON_PRESENT.format(name=neg_name),
                        "There are none.",
                        args.include_metadata,
                        {
                            "_task": "is_person_present",
                            "_target_person_name": neg_name,
                            "_expected_present": 0,
                            "_split_by_identity": split,
                            "_source": "absent_name_negative_checked_against_dev_test_page_names",
                            "_positive_dev_row_index_anchor": idx,
                            "_page_key": f"{ann.library}/{ann.document}/{ann.page_id}",
                        },
                    )
                )
                stats[f"{split}_is_person_present_negative"] += 1

    if "point_all_person_photos" in tasks:
        dev_page_representatives: dict[tuple[str, str, str], Ann] = {}
        for ann in dev_anns:
            dev_page_representatives.setdefault(ann.key, ann)
        for page_key, rep_ann in sorted(dev_page_representatives.items()):
            info = page_image_info(rep_ann)
            if info is None:
                stats["all_person_rows_skipped_missing_or_unreadable_image"] += 1
                continue
            (image_w, image_h), rel_project, rel_data = info
            page_names = names_present_by_page.get(page_key, set())
            has_val = bool(page_names & val_names)
            has_train = bool(page_names & train_names)
            if args.all_person_split_policy == "train":
                split = "train"
            elif (
                args.all_person_split_policy == "skip_if_mixed"
                and has_val
                and has_train
            ):
                stats["point_all_person_photos_skipped_mixed_identity_page"] += 1
                continue
            else:
                split = "val" if has_val else "train"
            dets = detections_by_page.get(page_key, [])
            n_dets = len(dets)
            if (
                args.max_point_all_detections > 0
                and n_dets > args.max_point_all_detections
            ):
                stats["point_all_person_photos_skipped_over_max_detections"] += 1
                stats[
                    f"{split}_point_all_person_photos_skipped_over_max_detections"
                ] += 1
                continue

            answer = all_points_answer(dets, image_w, image_h)
            rows_by_split[split].append(
                build_row(
                    rel_project,
                    rel_data,
                    PROMPT_POINT_ALL_PERSON_PHOTOS,
                    answer,
                    args.include_metadata,
                    {
                        "_task": "point_all_person_photos",
                        "_expected_present": int(bool(dets)),
                        "_split_by_identity": split,
                        "_source": (
                            "all_detections_on_unique_dev_page"
                            if all_detection_anns
                            else "named_annotations_fallback_on_unique_dev_page"
                        ),
                        "_n_gt_faces": n_dets,
                        "_max_point_all_detections": args.max_point_all_detections,
                        "_page_key": f"{rep_ann.library}/{rep_ann.document}/{rep_ann.page_id}",
                        "_page_has_val_identity": int(has_val),
                        "_page_has_train_identity": int(has_train),
                    },
                )
            )
            stats[f"{split}_point_all_person_photos"] += 1

    rng.shuffle(rows_by_split["train"])
    rng.shuffle(rows_by_split["val"])
    write_jsonl(Path(args.out_train), rows_by_split["train"])
    write_jsonl(Path(args.out_val), rows_by_split["val"])

    report = {
        "input_dev": dev_path.as_posix(),
        "input_test_context": (
            str(Path(args.peoplegator_test_jsonl).expanduser().resolve())
            if args.peoplegator_test_jsonl
            else None
        ),
        "input_all_detections": (
            str(Path(args.all_detections_jsonl).expanduser().resolve())
            if args.all_detections_jsonl
            else None
        ),
        "project_root": project_root.as_posix(),
        "data_roots_considered": [p.as_posix() for p in data_roots(project_root)],
        "tasks": sorted(tasks),
        "max_point_all_detections": args.max_point_all_detections,
        "dev_positive_annotation_rows_loaded": len(dev_anns),
        "unique_dev_pages": len({ann.key for ann in dev_anns}),
        "unique_dev_identities": len(dev_names),
        "unique_global_identities_dev_plus_test": len(global_names),
        "train_identities": len(train_names),
        "val_identities": len(val_names),
        "train_rows": len(rows_by_split["train"]),
        "val_rows": len(rows_by_split["val"]),
        "skipped_pages_for_image": len(skipped_pages_for_image),
        "stats": dict(stats),
    }
    if args.out_stats:
        stats_path = Path(args.out_stats).expanduser()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not rows_by_split["train"]:
        raise SystemExit(
            "No train rows were written. Check --project-root, image availability, and split settings."
        )


if __name__ == "__main__":
    main()
