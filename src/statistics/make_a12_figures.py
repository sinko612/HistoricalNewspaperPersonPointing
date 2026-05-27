#!/usr/bin/env python3
"""
Vytvára kvalitatívne obrázky pre adaptér A.12 z evaluačných výstupov.
Skript kreslí GT rámčeky, predikované body a textový panel s promptom a odpoveďou modelu.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

FIXED_ROW_CASES = [
    (
        "a12_name_pos_correct",
        4890,
        "point_person_by_name",
        "Prítomná osoba, správna lokalizácia",
    ),
    (
        "a12_name_pos_wrong",
        2309,
        "point_person_by_name",
        "Prítomná osoba, chybná lokalizácia",
    ),
    (
        "a12_name_neg_correct",
        6,
        "point_person_by_name",
        "Neprítomná osoba, správne odmietnutie",
    ),
    (
        "a12_name_neg_wrong",
        212,
        "point_person_by_name",
        "Neprítomná osoba, chybný vygenerovaný bod",
    ),
    (
        "a12_pointall_correct",
        291,
        "point_all_person_photos",
        "Všetky fotografie osôb, úspešná množinová predikcia",
    ),
    (
        "a12_pointall_wrong_overgen",
        1422,
        "point_all_person_photos",
        "Všetky fotografie osôb, nadgenerovanie bodov",
    ),
    (
        "a12_pointall_wrong_localization",
        2490,
        "point_all_person_photos",
        "Všetky fotografie osôb, presný počet, ale chybná poloha",
    ),
]

AUTO_CASES = [
    (
        "a12_present_pos_correct",
        "is_person_present",
        "pos_correct",
        "Prítomná osoba, správne potvrdenie",
    ),
    (
        "a12_present_pos_wrong",
        "is_person_present",
        "pos_wrong",
        "Prítomná osoba, potvrdenie s chybnou lokalizáciou",
    ),
    (
        "a12_present_neg_correct",
        "is_person_present",
        "neg_correct",
        "Neprítomná osoba, správne odmietnutie",
    ),
    (
        "a12_present_neg_wrong",
        "is_person_present",
        "neg_wrong",
        "Neprítomná osoba, chybný vygenerovaný bod",
    ),
]

SENSITIVE_NAME_RE = re.compile(
    r"Hitler|Stalin|Mussolini|Gottwald|Kopecký|Göring|Goebbels", re.IGNORECASE
)


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    page_w: float | None = None
    page_h: float | None = None
    score: float | None = None
    label: str | None = None

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path or not path.exists():
        return []

    def _iter() -> Iterable[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    return _iter()


def first_present(rec: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in rec and rec[key] not in (None, ""):
            return rec[key]
    return None


def page_key_from_record(rec: dict[str, Any]) -> str | None:
    return first_present(
        rec,
        ["page_id", "page_uuid", "image_id", "id", "page", "pageId", "img_id"],
    )


def label_from_record(rec: dict[str, Any]) -> str | None:
    return first_present(
        rec,
        [
            "target_person_name",
            "person_name",
            "name",
            "entity",
            "person",
            "label",
            "text",
        ],
    )


def extract_box(rec: dict[str, Any]) -> Box | None:
    page_w = first_present(
        rec,
        [
            "page_width",
            "page_image_width",
            "image_width",
            "width_orig",
            "page_w",
            "page_width_px",
        ],
    )
    page_h = first_present(
        rec,
        [
            "page_height",
            "page_image_height",
            "image_height",
            "height_orig",
            "page_h",
            "page_height_px",
        ],
    )

    left = first_present(rec, ["page_left", "x", "left", "x1"])
    top = first_present(rec, ["page_top", "y", "top", "y1"])
    width = first_present(rec, ["width", "w", "bbox_width"])
    height = first_present(rec, ["height", "h", "bbox_height"])
    if (
        left is not None
        and top is not None
        and width is not None
        and height is not None
    ):
        return Box(
            float(left),
            float(top),
            float(left) + float(width),
            float(top) + float(height),
            float(page_w) if page_w else None,
            float(page_h) if page_h else None,
            (
                float(rec["score"])
                if "score" in rec and rec["score"] is not None
                else None
            ),
            label_from_record(rec),
        )

    for key in ["bbox", "box", "face_bbox", "detection_bbox", "bounds", "bounding_box"]:
        val = rec.get(key)
        if isinstance(val, dict):
            lx = first_present(val, ["x", "left", "x1"])
            ty = first_present(val, ["y", "top", "y1"])
            rx = first_present(val, ["x2", "right"])
            by = first_present(val, ["y2", "bottom"])
            ww = first_present(val, ["w", "width"])
            hh = first_present(val, ["h", "height"])
            if lx is not None and ty is not None and rx is not None and by is not None:
                return Box(
                    float(lx),
                    float(ty),
                    float(rx),
                    float(by),
                    None,
                    None,
                    None,
                    label_from_record(rec),
                )
            if lx is not None and ty is not None and ww is not None and hh is not None:
                return Box(
                    float(lx),
                    float(ty),
                    float(lx) + float(ww),
                    float(ty) + float(hh),
                    None,
                    None,
                    None,
                    label_from_record(rec),
                )
        if isinstance(val, (list, tuple)) and len(val) >= 4:
            a, b, c, d = map(float, val[:4])
            if c > a and d > b and (c - a) > 3 and (d - b) > 3:
                return Box(a, b, c, d, None, None, None, label_from_record(rec))
            return Box(a, b, a + c, b + d, None, None, None, label_from_record(rec))
    return None


def load_box_index(path: Path) -> dict[str, list[Box]]:
    index: dict[str, list[Box]] = {}
    if not path or not path.exists():
        return index
    for rec in read_jsonl(path):
        page = page_key_from_record(rec)
        box = extract_box(rec)
        if page and box:
            index.setdefault(str(page), []).append(box)
    return index


def iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def deduplicate_boxes(boxes: list[Box], threshold: float = 0.97) -> list[Box]:
    out: list[Box] = []
    for box in sorted(boxes, key=lambda b: (b.score is None, -(b.score or 0.0))):
        if all(iou(box, prev) < threshold for prev in out):
            out.append(box)
    return out


def find_target_box(
    row: pd.Series, detections: list[Box], faces: list[Box]
) -> Box | None:
    tx = row.get("nearest_target_center_x")
    ty = row.get("nearest_target_center_y")
    if pd.isna(tx) or pd.isna(ty):
        return None
    target_name = str(row.get("target_person_name", ""))

    labelled = [
        b
        for b in faces
        if b.label and target_name and target_name.lower() in str(b.label).lower()
    ]
    candidates = labelled or faces or detections
    if not candidates:
        return None

    def dist2(box: Box) -> float:
        cx, cy = box.center
        return (cx - float(tx)) ** 2 + (cy - float(ty)) ** 2

    best = min(candidates, key=dist2)
    return best


def parse_model_points(
    answer: Any, image_size: tuple[int, int]
) -> list[tuple[float, float]]:
    if answer is None or (isinstance(answer, float) and math.isnan(answer)):
        return []
    text = str(answer)
    if "none" in text.lower() or "no person" in text.lower():
        return []
    w, h = image_size
    points: list[tuple[float, float]] = []

    for mx, my in re.findall(r'<point[^>]*\bx="([0-9.]+)"[^>]*\by="([0-9.]+)"', text):
        points.append((float(mx) / 100.0 * w, float(my) / 100.0 * h))

    xs = {int(i): float(v) for i, v in re.findall(r'\bx(\d+)="([0-9.]+)"', text)}
    ys = {int(i): float(v) for i, v in re.findall(r'\by(\d+)="([0-9.]+)"', text)}
    for i in sorted(set(xs) & set(ys)):
        points.append((xs[i] / 100.0 * w, ys[i] / 100.0 * h))
    return points


def parse_eval_cached_points(
    row: pd.Series, image_size: tuple[int, int]
) -> list[tuple[float, float]]:
    val = row.get("pred_points_json")
    if val is not None and not (isinstance(val, float) and math.isnan(val)):
        text = str(val).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            try:
                data = json.loads(text)
                points: list[tuple[float, float]] = []
                w, h = image_size
                for item in data if isinstance(data, list) else []:
                    if not isinstance(item, dict):
                        continue
                    if item.get("x_px") is not None and item.get("y_px") is not None:
                        points.append((float(item["x_px"]), float(item["y_px"])))
                    elif item.get("x100") is not None and item.get("y100") is not None:
                        points.append(
                            (
                                float(item["x100"]) / 100.0 * w,
                                float(item["y100"]) / 100.0 * h,
                            )
                        )
                if points:
                    return points
            except Exception:
                pass
    return parse_model_points(row.get("answer"), image_size)


def group_duplicate_points(
    points: list[tuple[float, float]], eps: float = 1e-3
) -> list[tuple[tuple[float, float], int]]:

    groups: list[tuple[tuple[float, float], int]] = []
    for x, y in points:
        for idx, ((gx, gy), count) in enumerate(groups):
            if abs(x - gx) <= eps and abs(y - gy) <= eps:
                groups[idx] = ((gx, gy), count + 1)
                break
        else:
            groups.append(((x, y), 1))
    return groups


def draw_duplicate_count_label(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    count: int,
    text_font: ImageFont.ImageFont,
) -> None:

    if count <= 1:
        return

    label = f"×{count}"

    tx, ty = x + 28, y + 10
    bbox = draw.textbbox((tx, ty), label, font=text_font)
    pad_x, pad_y = 14, 8

    rect = (
        bbox[0] - pad_x,
        bbox[1] - pad_y,
        bbox[2] + pad_x,
        bbox[3] + pad_y,
    )
    draw.rectangle(rect, fill=(255, 255, 255), outline=(210, 0, 0), width=5)
    draw.text(
        (tx, ty),
        label,
        fill=(210, 0, 0),
        font=text_font,
        stroke_width=1,
        stroke_fill=(255, 255, 255),
    )


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:

    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        (
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
        ),
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    ]
    for path in candidates:
        try:
            if Path(path).exists() or not Path(path).is_absolute():
                return ImageFont.truetype(str(path), size=size)
        except Exception:
            pass

    print(
        f"WARNING: Scalable TTF font not found; using Pillow default font at requested size {size}."
    )
    return ImageFont.load_default()


def transform_box(
    box: Box, img_size: tuple[int, int], out_scale: float
) -> tuple[float, float, float, float]:
    img_w, img_h = img_size
    sx = img_w / box.page_w if box.page_w and box.page_w > 0 else 1.0
    sy = img_h / box.page_h if box.page_h and box.page_h > 0 else 1.0
    return (
        box.x1 * sx * out_scale,
        box.y1 * sy * out_scale,
        box.x2 * sx * out_scale,
        box.y2 * sy * out_scale,
    )


def draw_cross(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    radius: int,
    color: tuple[int, int, int],
    width: int,
) -> None:
    draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=width)
    draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=width)


def draw_point(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    radius: int,
    fill: tuple[int, int, int],
    outline=(255, 255, 255),
) -> None:
    draw.ellipse(
        (x - radius - 3, y - radius - 3, x + radius + 3, y + radius + 3), fill=outline
    )
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def shorten_answer(answer: Any, max_len: int = 180) -> str:
    if answer is None or (isinstance(answer, float) and math.isnan(answer)):
        return ""
    text = re.sub(r"\s+", " ", str(answer)).strip()
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def build_prompt_label(row: pd.Series) -> str:
    mode = row.get("eval_mode")
    name = row.get("target_person_name")
    if mode == "point_person_by_name":
        return f"Prompt: point_person_by_name; meno: {name}"
    if mode == "is_person_present":
        return f"Prompt: is_person_present; meno: {name}"
    return "Prompt: point_all_person_photos"


def draw_example(
    row: pd.Series,
    detections: list[Box],
    faces: list[Box],
    output_path: Path,
    title: str,
    max_width: int = 1800,
) -> dict[str, Any]:
    image_path = Path(str(row["image_path"]))
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        orig_w, orig_h = im.size
        scale = min(1.0, max_width / float(orig_w))
        out_w, out_h = int(orig_w * scale), int(orig_h * scale)
        page = (
            im.resize((out_w, out_h), Image.Resampling.LANCZOS)
            if scale != 1.0
            else im.copy()
        )

    answer_f = font(78, bold=False)
    duplicate_f = font(72, bold=True)
    margin = 32
    top_h = 0
    bottom_h = 620
    canvas = Image.new(
        "RGB", (out_w + 2 * margin, out_h + top_h + bottom_h + 2 * margin), "white"
    )
    canvas.paste(page, (margin, top_h))
    draw = ImageDraw.Draw(canvas)

    mode = row.get("eval_mode")
    reference_boxes: list[Box] = []
    if mode == "point_all_person_photos":
        reference_boxes = deduplicate_boxes(detections)
    elif bool(row.get("expected_present")):
        target = find_target_box(row, detections, faces)
        if target:
            reference_boxes = [target]

    for box in reference_boxes:
        x1, y1, x2, y2 = transform_box(box, (orig_w, orig_h), scale)
        x1 += margin
        x2 += margin
        y1 += top_h
        y2 += top_h
        draw.rectangle(
            (x1, y1, x2, y2), outline=(0, 155, 80), width=max(7, int(8 * scale))
        )
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        draw_point(draw, cx, cy, max(12, int(16 * scale)), fill=(0, 170, 80))

    if (
        mode != "point_all_person_photos"
        and bool(row.get("expected_present"))
        and not reference_boxes
    ):
        tx = row.get("nearest_target_center_x")
        ty = row.get("nearest_target_center_y")
        if not pd.isna(tx) and not pd.isna(ty):
            draw_point(
                draw,
                margin + float(tx) * scale,
                top_h + float(ty) * scale,
                max(15, int(20 * scale)),
                fill=(0, 170, 80),
            )

    pred_points: list[tuple[float, float]] = []
    if mode == "point_all_person_photos":
        pred_points = parse_eval_cached_points(row, (orig_w, orig_h))
    else:
        px = row.get("pred_x_px")
        py = row.get("pred_y_px")
        if not pd.isna(px) and not pd.isna(py):
            pred_points = [(float(px), float(py))]

    for (x, y), duplicate_count in group_duplicate_points(pred_points):
        cx = margin + x * scale
        cy = top_h + y * scale
        pr = max(12, int(16 * scale))
        halo = max(5, int(6 * scale))
        draw.ellipse(
            (cx - pr - halo, cy - pr - halo, cx + pr + halo, cy + pr + halo),
            fill=(255, 255, 255),
        )
        draw.ellipse(
            (cx - pr, cy - pr, cx + pr, cy + pr),
            fill=(210, 0, 0),
            outline=(110, 0, 0),
            width=max(4, int(5 * scale)),
        )
        draw_duplicate_count_label(draw, cx, cy, duplicate_count, duplicate_f)

    strip_y = top_h + out_h + margin
    draw.line(
        (margin, strip_y - 12, margin + out_w, strip_y - 12), fill=(0, 0, 0), width=2
    )
    answer = shorten_answer(row.get("answer"), max_len=320)
    gt = "prítomná" if bool(row.get("expected_present")) else "neprítomná"
    if mode == "point_all_person_photos":
        gt = f"ref. bodov: {int(row.get('n_gt_faces_without_duplicates', 0))}; pred. bodov: {int(row.get('n_pred_points', len(pred_points)))}"
    text = f"GT: {gt}. Odpoveď modelu: {answer}"
    wrapped = textwrap.wrap(text, width=max(32, int(out_w / 45)))
    y = strip_y
    line_h = 96
    for line in wrapped[:6]:
        draw.text((margin, y), line, fill=(0, 0, 0), font=answer_f)
        y += line_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92, optimize=True)
    return {
        "file": str(output_path),
        "n_reference_boxes_drawn": len(reference_boxes),
        "n_prediction_points_drawn": len(pred_points),
    }


def extract_results_from_zip(eval_outputs_zip: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(eval_outputs_zip) as outer:
        candidates = [
            n for n in outer.namelist() if "safe_pos2" in n and n.endswith(".zip")
        ]
        if not candidates:
            raise FileNotFoundError(
                "A12 safe_pos2 nested zip was not found in eval outputs zip."
            )
        nested_name = candidates[0]
        nested_bytes = outer.read(nested_name)
    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
        results_candidates = [
            n for n in inner.namelist() if n.endswith("checkpoint-1250/results.csv")
        ]
        if not results_candidates:
            raise FileNotFoundError(
                "checkpoint-1250/results.csv was not found in A12 nested zip."
            )
        inner.extract(results_candidates[0], output_dir)
        return output_dir / results_candidates[0]


def load_results(args: argparse.Namespace) -> pd.DataFrame:
    if args.results_csv:
        csv_path = Path(args.results_csv)
    else:
        csv_path = extract_results_from_zip(
            Path(args.eval_outputs_zip), Path(args.work_dir)
        )
    df = pd.read_csv(csv_path).reset_index().rename(columns={"index": "row_id"})
    return df


def auto_select(df: pd.DataFrame, mode: str, case: str) -> pd.Series:
    sub = df[df["eval_mode"] == mode].copy()
    if case == "pos_correct":
        sub = sub[
            (sub.expected_present == True)
            & (sub.success == True)
            & (sub.predicted_none == False)
        ]
        sub = sub.sort_values("dist_px_to_nearest_target", ascending=True)
    elif case == "pos_wrong":
        sub = sub[
            (sub.expected_present == True)
            & (sub.success == False)
            & (sub.predicted_none == False)
        ]
        sub = sub.sort_values("dist_px_to_nearest_target", ascending=False)
    elif case == "neg_correct":
        sub = sub[
            (sub.expected_present == False)
            & (sub.success == True)
            & (sub.predicted_none == True)
        ]
    elif case == "neg_wrong":
        sub = sub[
            (sub.expected_present == False)
            & (sub.success == False)
            & (sub.predicted_none == False)
        ]
        sub = sub[~sub.target_person_name.fillna("").str.contains(SENSITIVE_NAME_RE)]
    else:
        raise ValueError(case)
    if sub.empty:
        raise RuntimeError(f"No candidate found for {mode}/{case}")
    return sub.iloc[0]


def row_page_keys(row: pd.Series) -> list[str]:
    keys: list[str] = []
    for col in ["page", "page_id", "image_name"]:
        val = row.get(col)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        text = str(val)
        keys.append(text)
        if not text.endswith(".jpg"):
            keys.append(text + ".jpg")
        keys.append(Path(text).name)
    out: list[str] = []
    for key in keys:
        if key and key not in out:
            out.append(key)
    return out


def boxes_for_row(
    row: pd.Series, det_index: dict[str, list[Box]], face_index: dict[str, list[Box]]
) -> tuple[list[Box], list[Box]]:
    dets: list[Box] = []
    faces: list[Box] = []
    for key in row_page_keys(row):
        dets.extend(det_index.get(key, []))
        faces.extend(face_index.get(key, []))
    return deduplicate_boxes(dets), deduplicate_boxes(faces)


def maybe_relocate_images(df: pd.DataFrame, image_root: Path | None) -> pd.DataFrame:
    if image_root is None or not image_root.exists():
        return df
    missing = [not Path(str(p)).exists() for p in df["image_path"].fillna("")]
    if not any(missing):
        return df
    jpg_index = {p.stem: p for p in image_root.rglob("*.jpg")}
    df = df.copy()
    for idx, row in df.iterrows():
        p = Path(str(row.get("image_path", "")))
        if p.exists():
            continue
        page_id = str(row.get("page_id", ""))
        if page_id in jpg_index:
            df.at[idx, "image_path"] = str(jpg_index[page_id])
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Direct path to A12 checkpoint-1250 results.csv",
    )
    parser.add_argument(
        "--eval-outputs-zip",
        default="all_molmo_eval_outputs.zip",
        help="Zip containing nested eval outputs",
    )
    parser.add_argument("--detections-jsonl", default="people_gator__detections.jsonl")
    parser.add_argument(
        "--faces-jsonl",
        default="people_gator__corresponding_faces__2026-02-11.test.jsonl",
    )
    parser.add_argument("--output-dir", default="figures/a12_qualitative")
    parser.add_argument("--work-dir", default="_a12_eval_extract")
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional root used to find PAGE_ID.jpg when stored absolute paths are invalid",
    )
    parser.add_argument("--max-width", type=int, default=1800)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    df = load_results(args)
    df = maybe_relocate_images(df, Path(args.image_root) if args.image_root else None)

    det_index = load_box_index(Path(args.detections_jsonl))
    face_index = load_box_index(Path(args.faces_jsonl))

    selected_rows: list[dict[str, Any]] = []

    for filename, row_id, mode, title in FIXED_ROW_CASES:
        match = df[df.row_id == row_id]
        if match.empty:
            raise RuntimeError(f"Fixed row_id {row_id} for {filename} not found.")
        row = match.iloc[0]
        dets, faces = boxes_for_row(row, det_index, face_index)
        result = draw_example(
            row,
            dets,
            faces,
            out_dir / f"{filename}.jpg",
            title,
            max_width=args.max_width,
        )
        selected_rows.append(
            {
                **row.to_dict(),
                "figure_name": filename,
                "figure_file": result["file"],
                "case_title": title,
                **result,
            }
        )

    for filename, mode, case, title in AUTO_CASES:
        row = auto_select(df, mode, case)
        dets, faces = boxes_for_row(row, det_index, face_index)
        result = draw_example(
            row,
            dets,
            faces,
            out_dir / f"{filename}.jpg",
            title,
            max_width=args.max_width,
        )
        selected_rows.append(
            {
                **row.to_dict(),
                "figure_name": filename,
                "figure_file": result["file"],
                "case_title": title,
                **result,
            }
        )

    summary_csv = out_dir / "a12_qualitative_selected_examples.csv"
    pd.DataFrame(selected_rows).to_csv(summary_csv, index=False)
    print(f"Saved {len(selected_rows)} figures to {out_dir}")
    print(f"Saved selected-example summary to {summary_csv}")


if __name__ == "__main__":
    main()
