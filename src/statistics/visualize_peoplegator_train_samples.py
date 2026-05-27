#!/usr/bin/env python3
"""
Visualize sampled Molmo training rows as annotated contact sheets.
The script overlays model-format points, PeopleGator detections, and short prompt/answer summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

TASK_ORDER = [
    "point_person_by_name",
    "is_person_present",
    "point_all_person_photos",
]

POINT_RE = re.compile(r'x="([0-9]+(?:\.[0-9]+)?)"\s+y="([0-9]+(?:\.[0-9]+)?)"')
POINTS_BLOCK_RE = re.compile(r"<points\s+([^>]+)>", re.IGNORECASE | re.DOTALL)
INDEXED_POINT_RE = re.compile(
    r'x(\d+)="([0-9]+(?:\.[0-9]+)?)"\s+y\1="([0-9]+(?:\.[0-9]+)?)"'
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Visualize random PeopleGator train samples with Molmo target points and detection boxes."
    )
    ap.add_argument(
        "--train-jsonl", required=True, help="Path to the train dataset JSONL."
    )
    ap.add_argument(
        "--detections-jsonl",
        required=True,
        help="Path to people_gator__detections.jsonl.",
    )
    ap.add_argument(
        "--project-root",
        required=True,
        help="Project root (e.g. ../.. when running from src/datasets).",
    )
    ap.add_argument(
        "--out-dir", required=True, help="Directory where visualizations will be saved."
    )
    ap.add_argument(
        "--samples-per-task",
        type=int,
        default=5,
        help="How many random samples per task.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--include-none-responses",
        action="store_true",
        help="By default, samples with assistant response 'There are none.' / none / empty are skipped. Use this to include them.",
    )
    ap.add_argument(
        "--draw-all-detections",
        action="store_true",
        help="Draw all detections on the page (default). If omitted, only matched detections are emphasized and all detections are still lightly shown.",
    )
    ap.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Skip samples whose images are missing instead of failing.",
    )
    return ap.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def page_key_from_detection(obj: dict[str, Any]) -> str:
    page = Path(str(obj["page"])).stem
    return f"{obj['library']}/{obj['document']}/{page}"


def load_detections(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            grouped[page_key_from_detection(obj)].append(obj)
    return grouped


def extract_assistant_response(row: dict[str, Any]) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


def has_non_none_response(row: dict[str, Any]) -> bool:
    response = extract_assistant_response(row).strip()
    if not response:
        return False
    return response.lower() not in {
        "none",
        "null",
        "there are none",
        "there are none.",
    }


def extract_user_prompt(row: dict[str, Any]) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") == "user":
            return str(msg.get("content", "")).strip()
    return ""


def parse_points_from_response(text: str) -> list[tuple[float, float]]:
    text = (text or "").strip()
    if not text or text.lower() == "there are none.":
        return []

    points: list[tuple[float, float]] = []

    for x, y in POINT_RE.findall(text):
        points.append((float(x), float(y)))
    if points:
        return points

    m = POINTS_BLOCK_RE.search(text)
    if not m:
        return []
    attrs = m.group(1)
    indexed = []
    for idx, x, y in INDEXED_POINT_RE.findall(attrs):
        indexed.append((int(idx), float(x), float(y)))
    indexed.sort(key=lambda t: t[0])
    return [(x, y) for _, x, y in indexed]


def molmo_to_pixel(
    points_0_100: list[tuple[float, float]], img_w: int, img_h: int
) -> list[tuple[float, float]]:
    return [(x / 100.0 * img_w, y / 100.0 * img_h) for x, y in points_0_100]


def bbox_contains_point(det: dict[str, Any], pt: tuple[float, float]) -> bool:
    x, y = pt
    left = float(det["page_left"])
    top = float(det["page_top"])
    right = left + float(det["width"])
    bottom = top + float(det["height"])
    return left <= x <= right and top <= y <= bottom


def bbox_center(det: dict[str, Any]) -> tuple[float, float]:
    return (
        float(det["page_left"]) + float(det["width"]) / 2.0,
        float(det["page_top"]) + float(det["height"]) / 2.0,
    )


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def match_points_to_detections(
    points_px: list[tuple[float, float]], detections: list[dict[str, Any]]
) -> list[int]:
    matched: list[int] = []
    used: set[int] = set()
    for pt in points_px:
        containing = [
            i
            for i, d in enumerate(detections)
            if i not in used and bbox_contains_point(d, pt)
        ]
        if containing:
            best = min(
                containing, key=lambda i: distance(pt, bbox_center(detections[i]))
            )
            matched.append(best)
            used.add(best)
            continue
        if detections:
            candidates = [i for i in range(len(detections)) if i not in used]
            if not candidates:
                continue
            best = min(
                candidates, key=lambda i: distance(pt, bbox_center(detections[i]))
            )
            matched.append(best)
            used.add(best)
    return matched


def draw_cross(
    draw: ImageDraw.ImageDraw, x: float, y: float, size: int = 10, width: int = 3
):
    draw.line((x - size, y, x + size, y), fill="red", width=width)
    draw.line((x, y - size, x, y + size), fill="red", width=width)
    r = max(3, size // 3)
    draw.ellipse(
        (x - r, y - r, x + r, y + r),
        outline="white",
        width=max(1, width - 1),
        fill="red",
    )


def annotation_panel(
    img: Image.Image,
    row: dict[str, Any],
    detections: list[dict[str, Any]],
    font: ImageFont.ImageFont,
) -> Image.Image:
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    response = extract_assistant_response(row)
    prompt = extract_user_prompt(row)
    task = row.get("_task", "unknown")
    points_molmo = parse_points_from_response(response)
    points_px = molmo_to_pixel(points_molmo, img.width, img.height)
    matched_idxs = match_points_to_detections(points_px, detections)
    matched_set = set(matched_idxs)

    for i, det in enumerate(detections):
        left = float(det["page_left"])
        top = float(det["page_top"])
        right = left + float(det["width"])
        bottom = top + float(det["height"])
        color = "lime" if i in matched_set else "yellow"
        width = 5 if i in matched_set else 2
        draw.rectangle((left, top, right, bottom), outline=color, width=width)
        if i in matched_set:
            label = f"det {i}"
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            draw.rectangle((left, max(0, top - th - 4), left + tw + 6, top), fill=color)
            draw.text((left + 3, max(0, top - th - 2)), label, fill="black", font=font)

    for j, (x, y) in enumerate(points_px, start=1):
        draw_cross(draw, x, y, size=12, width=4)
        label = str(j)
        draw.text((x + 10, y + 8), label, fill="red", font=font)

    footer_lines = [
        f"task: {task}",
        f"page_key: {row.get('_page_key', '')}",
        f"target_name: {row.get('_target_person_name', '-')}",
        f"response: {response if len(response) <= 140 else response[:137] + '...'}",
        f"n_points: {len(points_px)} | n_detections: {len(detections)} | matched_boxes: {len(matched_set)}",
    ]

    prompt_wrapped = textwrap.wrap(prompt.replace("\n", " "), width=85)
    footer_lines.append("prompt: " + (prompt_wrapped[0] if prompt_wrapped else ""))
    for extra in prompt_wrapped[1:3]:
        footer_lines.append("        " + extra)

    line_height = 18
    footer_h = 12 + line_height * len(footer_lines)
    canvas = Image.new("RGB", (img.width, img.height + footer_h), "white")
    canvas.paste(img, (0, 0))
    cdraw = ImageDraw.Draw(canvas)
    y0 = img.height + 6
    for line in footer_lines:
        cdraw.text((8, y0), line, fill="black", font=font)
        y0 += line_height
    return canvas


def make_contact_sheet(
    images: list[Image.Image], out_path: Path, font: ImageFont.ImageFont, title: str
):
    if not images:
        return

    thumb_max_w = 520
    thumb_max_h = 760
    processed = []
    for img in images:
        copy = img.copy()
        copy.thumbnail((thumb_max_w, thumb_max_h))
        processed.append(copy)

    cols = 2
    rows = math.ceil(len(processed) / cols)
    cell_w = max(im.width for im in processed) + 20
    cell_h = max(im.height for im in processed) + 20
    title_h = 40
    sheet = Image.new(
        "RGB", (cols * cell_w + 20, rows * cell_h + title_h + 20), "#f0f0f0"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill="black", font=font)

    for idx, im in enumerate(processed):
        r = idx // cols
        c = idx % cols
        x = 10 + c * cell_w
        y = title_h + 10 + r * cell_h
        sheet.paste(im, (x, y))
        draw.rectangle((x, y, x + im.width, y + im.height), outline="#555555", width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(Path(args.train_jsonl))
    detections_by_page = load_detections(Path(args.detections_jsonl))

    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped_none = 0
    for row in rows:
        if not args.include_none_responses and not has_non_none_response(row):
            skipped_none += 1
            continue
        rows_by_task[row.get("_task", "unknown")].append(row)

    if skipped_none:
        print(
            f"[INFO] Skipped {skipped_none} samples with none/empty responses. Use --include-none-responses to draw them too."
        )

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    summary = []

    for task in TASK_ORDER:
        task_rows = rows_by_task.get(task, [])
        if not task_rows:
            print(f"[WARN] No rows for task: {task}")
            continue

        sample_n = min(args.samples_per_task, len(task_rows))
        sampled = rng.sample(task_rows, sample_n)
        rendered: list[Image.Image] = []
        saved_count = 0

        task_dir = out_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in enumerate(sampled, start=1):
            rel_path = row.get("image_rel_path_from_project_root") or row.get(
                "image_rel_path"
            )
            if not rel_path:
                print(f"[WARN] Missing image path in row for task {task}")
                continue
            img_path = (project_root / rel_path).resolve()
            if not img_path.exists():
                msg = f"[WARN] Missing image: {img_path}"
                if args.allow_missing_images:
                    print(msg)
                    continue
                raise FileNotFoundError(msg)

            page_key = row.get("_page_key", "")
            dets = detections_by_page.get(page_key, [])

            with Image.open(img_path) as im:
                panel = annotation_panel(im.copy(), row, dets, font)

            indiv_path = task_dir / f"{idx:02d}_{Path(img_path).stem}.png"
            panel.save(indiv_path)
            rendered.append(panel)
            saved_count += 1

        if rendered:
            sheet_path = out_dir / f"{task}_samples.png"
            make_contact_sheet(
                rendered,
                sheet_path,
                font,
                f"{task} | {saved_count} random train samples",
            )
            summary.append((task, saved_count, str(sheet_path)))
        else:
            summary.append((task, 0, "no images saved"))

    print("\nDone. Outputs:")
    for task, count, path in summary:
        print(f"- {task}: {count} samples -> {path}")


if __name__ == "__main__":
    main()
