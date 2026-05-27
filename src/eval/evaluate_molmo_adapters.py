#!/usr/bin/env python3
"""
Evaluate a Molmo model with one or more LoRA adapters on PeopleGator-style tasks.
The script runs generation, parses point outputs, compares them with annotations, and writes detailed metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".jp2", ".j2k"}

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

POINT_TAG_RE = re.compile(
    r'<point\b[^>]*x="([0-9]+(?:\.[0-9]+)?)"[^>]*y="([0-9]+(?:\.[0-9]+)?)"[^>]*>(.*?)</point>',
    re.IGNORECASE | re.DOTALL,
)
POINTS_TAG_RE = re.compile(
    r"<points\b([^>]*)>(.*?)</points>",
    re.IGNORECASE | re.DOTALL,
)
ATTR_RE = re.compile(r'([A-Za-z0-9_]+)="(.*?)"')
ANY_XY_RE = re.compile(
    r'x(?:\d+)?="([0-9]+(?:\.[0-9]+)?)".*?y(?:\d+)?="([0-9]+(?:\.[0-9]+)?)"',
    re.IGNORECASE | re.DOTALL,
)
NONE_RE = re.compile(r"^\s*there\s+are\s+none\.?\s*$", re.IGNORECASE)

CENTER_THRESH_PX = [10, 20, 30, 50, 100]


def parse_args():
    """Parse command-line arguments controlling inputs, tasks, model loading, and output paths."""
    ap = argparse.ArgumentParser(
        description="Evaluate Molmo + LoRA on PeopleGator name-conditioned and all-person-photo tasks."
    )
    ap.add_argument("--model-id", default="allenai/Molmo-7B-D-0924")
    ap.add_argument(
        "--adapter-dir",
        required=True,
        help="Path to the trained LoRA adapter directory, e.g. runs/... or runs/.../checkpoint-1708",
    )
    ap.add_argument(
        "--annotations-jsonl",
        required=True,
        help="Raw PeopleGator test/dev JSONL with corresponding/named face annotations. For point_all_person_photos this file defines the unique page set to evaluate.",
    )
    ap.add_argument(
        "--detections-jsonl",
        default="",
        help="Raw PeopleGator detections JSONL with all face detections. Required for point_all_person_photos; if omitted, a sibling people_gator__detections.jsonl is used when present.",
    )
    ap.add_argument(
        "--data-root", required=True, help="Path to datasets/digiknihovna_data"
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--eval-mode",
        choices=[
            "point_person_by_name",
            "is_person_present",
            "point_all_person_photos",
            "both",
            "all",
        ],
        default="all",
        help="both = the two name-conditioned modes; all = both name-conditioned modes plus point_all_person_photos",
    )

    ap.add_argument(
        "--negative-names-per-page",
        type=int,
        default=1,
        help="Absent-person queries sampled per page for each selected task",
    )
    ap.add_argument("--negative-random-seed", type=int, default=42)
    ap.add_argument(
        "--max-pages", type=int, default=20, help="0 means all annotated pages"
    )

    ap.add_argument("--resize-long-side", type=int, default=512)
    ap.add_argument("--max-crops", type=int, default=2)
    ap.add_argument("--sequence-length", type=int, default=768)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument(
        "--max-point-all-detections",
        type=int,
        default=8,
        help="For point_all_person_photos evaluate only pages with at most this many RAW face detections, matching the training data construction.",
    )
    ap.add_argument(
        "--dedupe-iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used for the without-duplicates point_all_person_photos statistics. The run always reports both strict/raw and deduplicated statistics.",
    )

    ap.add_argument(
        "--device-map",
        default="auto",
        choices=["", "auto", "balanced", "balanced_low_0", "sequential", "block"],
        help="Use block for a manual whole-transformer-block map that is safer with Molmo + LoRA on multiple GPUs.",
    )
    ap.add_argument("--max-memory", default="", help="Example: 0:22GiB,1:22GiB,2:22GiB")
    ap.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Use bf16/fp16 instead of 4-bit loading",
    )
    ap.add_argument("--force-cpu", action="store_true")

    ap.add_argument("--print-raw", action="store_true")
    ap.add_argument("--save-failures", action="store_true")
    ap.add_argument("--max-failures", type=int, default=5)
    return ap.parse_args()


def parse_max_memory(spec: str):
    """
    Parse a device-to-memory mapping used by Hugging Face model loading.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"Invalid --max-memory part {part!r}; expected e.g. 0:22GiB"
            )
        k, v = part.split(":", 1)
        k = k.strip()
        out[int(k) if k.isdigit() else k] = v.strip()
    return out


def build_molmo_block_device_map(num_gpus: int, num_blocks: int = 28) -> dict | None:
    """
    Build a layer-aware device map for Molmo transformer blocks. This is used when the model is split manually across multiple devices.
    """
    if num_gpus < 1:
        return None

    if num_gpus == 1:
        ranges = [(0, num_blocks - 1, 0)]
    elif num_gpus == 2:
        ranges = [(0, 11, 0), (12, num_blocks - 1, 1)]
    elif num_gpus == 3:
        ranges = [(0, 7, 0), (8, 17, 1), (18, num_blocks - 1, 2)]
    else:
        ranges = [(0, 5, 0), (6, 13, 1), (14, 20, 2), (21, num_blocks - 1, 3)]

    last_device = ranges[-1][2]
    device_map = {
        "model.transformer.wte": 0,
        "model.transformer.emb_drop": 0,
        "model.vision_backbone": 0,
        "model.transformer.ln_f": last_device,
        "model.transformer.ff_out": last_device,
    }
    for start, end, dev in ranges:
        for i in range(start, end + 1):
            device_map[f"model.transformer.blocks.{i}"] = dev
    return device_map


def mark_model_parallel(model):
    """
    Mark the loaded model as model-parallel when a custom device map is used.
    """
    candidates = [model]
    for attr in ("base_model", "model"):
        obj = getattr(model, attr, None)
        if obj is not None:
            candidates.append(obj)
            nested = getattr(obj, "model", None)
            if nested is not None:
                candidates.append(nested)

    for obj in candidates:
        try:
            obj.is_parallelizable = True
            obj.model_parallel = True
        except Exception:
            pass
    return model


def align_lora_devices_with_base(model):
    """
    Move LoRA adapter weights onto the same devices as the corresponding base-model weights.
    """
    moved = 0
    checked = 0

    for module in model.modules():
        base_layer = getattr(module, "base_layer", None)
        if base_layer is None:
            continue
        try:
            base_param = next(base_layer.parameters())
        except StopIteration:
            continue

        checked += 1
        target_device = base_param.device
        for attr in (
            "lora_A",
            "lora_B",
            "lora_embedding_A",
            "lora_embedding_B",
            "lora_dropout",
        ):
            obj = getattr(module, attr, None)
            if obj is None:
                continue
            if isinstance(obj, torch.nn.ModuleDict):
                for submodule in obj.values():
                    submodule.to(target_device)
                    moved += 1
            elif isinstance(obj, torch.nn.ParameterDict):
                for param in obj.values():
                    param.data = param.data.to(target_device)
                    if param.grad is not None:
                        param.grad = param.grad.to(target_device)
                    moved += 1
            elif isinstance(obj, torch.nn.Module):
                obj.to(target_device)
                moved += 1

    print(
        f"Checked LoRA-wrapped layers: {checked}; aligned LoRA submodules/params: {moved}"
    )
    return model


def normalize_page_id(page: str | None):
    """
    Normalize a page identifier by removing path and extension details.
    """
    return Path(str(page or "")).stem


def normalize_name(text: str | None):
    """
    Normalize a person name by collapsing whitespace and lowercasing it. Empty names are returned as an empty string.
    """
    if text is None:
        return None
    text = re.sub(r"\s+", " ", str(text).strip())
    return text or None


def load_peoplegator_annotations(jsonl_path: Path):
    """
    Load PeopleGator face annotations and group them by page key.
    """
    ann_by_key = defaultdict(list)
    seen = set()

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if "document_info" in obj and "face_info" in obj:
                di = obj.get("document_info", {})
                fi = obj.get("face_info", {})
                library = di.get("library")
                document = di.get("document")
                page_id = normalize_page_id(di.get("page", ""))
                person_name = normalize_name(obj.get("person_name"))
                x1, y1, w, h = (
                    fi.get("x"),
                    fi.get("y"),
                    fi.get("width"),
                    fi.get("height"),
                )
                raw_kps = []
                kp_ids = sorted(
                    int(m.group(1))
                    for k in fi.keys()
                    for m in [re.match(r"kp_(\d+)_x$", k)]
                    if m is not None
                )
                for kp_id in kp_ids:
                    kx, ky = fi.get(f"kp_{kp_id}_x"), fi.get(f"kp_{kp_id}_y")
                    if kx is not None and ky is not None:
                        raw_kps.append((kx, ky))
            else:
                library = obj.get("library")
                document = obj.get("document")
                page_id = normalize_page_id(obj.get("page", ""))
                person_name = normalize_name(obj.get("person_name"))
                x1, y1, w, h = (
                    obj.get("page_left"),
                    obj.get("page_top"),
                    obj.get("width"),
                    obj.get("height"),
                )
                raw_kps = obj.get("page_keypoints") or []

            if not library or not document or not page_id or not person_name:
                continue
            if x1 is None or y1 is None or w is None or h is None:
                continue

            x1 = float(x1)
            y1 = float(y1)
            w = float(w)
            h = float(h)
            if w <= 0 or h <= 0:
                continue
            x2 = x1 + w
            y2 = y1 + h

            keypoints = []
            for kp in raw_kps:
                if isinstance(kp, dict):
                    kx, ky = kp.get("x"), kp.get("y")
                elif isinstance(kp, (list, tuple)) and len(kp) >= 2:
                    kx, ky = kp[0], kp[1]
                else:
                    continue
                if kx is not None and ky is not None:
                    keypoints.append((round(float(kx), 3), round(float(ky), 3)))

            dedupe_key = (
                library,
                document,
                page_id,
                person_name,
                round(x1, 3),
                round(y1, 3),
                round(w, 3),
                round(h, 3),
                tuple(keypoints),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            ann_by_key[(library, document, page_id)].append(
                {
                    "library": library,
                    "document": document,
                    "page_id": page_id,
                    "person_name": person_name,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "bbox_w": w,
                    "bbox_h": h,
                    "keypoints": [{"x": x, "y": y} for x, y in keypoints],
                }
            )

    return dict(ann_by_key)


def load_unique_page_keys(jsonl_path: Path):
    """
    Load the unique page keys present in a JSONL annotation file.
    """
    keys = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "document_info" in obj:
                di = obj.get("document_info", {})
                library = di.get("library")
                document = di.get("document")
                page_id = normalize_page_id(di.get("page", ""))
            else:
                library = obj.get("library")
                document = obj.get("document")
                page_id = normalize_page_id(obj.get("page", ""))
            if library and document and page_id:
                keys.add((library, document, page_id))
    return keys


def _keypoints_from_flat_obj(obj: dict):
    """
    Extract face keypoints from a flat PeopleGator JSON object.
    """
    raw_kps = obj.get("page_keypoints") or []
    keypoints = []
    for kp in raw_kps:
        if isinstance(kp, dict):
            kx, ky = kp.get("x"), kp.get("y")
        elif isinstance(kp, (list, tuple)) and len(kp) >= 2:
            kx, ky = kp[0], kp[1]
        else:
            continue
        if kx is not None and ky is not None:
            keypoints.append((round(float(kx), 3), round(float(ky), 3)))
    return keypoints


def load_peoplegator_detections(jsonl_path: Path):
    """
    Load page-level PeopleGator detections and convert them to evaluation boxes.
    """
    det_by_key = defaultdict(list)
    seen = set()

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if "document_info" in obj and "face_info" in obj:
                di = obj.get("document_info", {})
                fi = obj.get("face_info", {})
                library = di.get("library")
                document = di.get("document")
                page_id = normalize_page_id(di.get("page", ""))
                x1, y1, w, h = (
                    fi.get("x"),
                    fi.get("y"),
                    fi.get("width"),
                    fi.get("height"),
                )
                kp_ids = sorted(
                    int(m.group(1))
                    for k in fi.keys()
                    for m in [re.match(r"kp_(\d+)_x$", k)]
                    if m is not None
                )
                keypoints = []
                for kp_id in kp_ids:
                    kx, ky = fi.get(f"kp_{kp_id}_x"), fi.get(f"kp_{kp_id}_y")
                    if kx is not None and ky is not None:
                        keypoints.append((round(float(kx), 3), round(float(ky), 3)))
            else:
                library = obj.get("library")
                document = obj.get("document")
                page_id = normalize_page_id(obj.get("page", ""))
                x1, y1, w, h = (
                    obj.get("page_left"),
                    obj.get("page_top"),
                    obj.get("width"),
                    obj.get("height"),
                )
                keypoints = _keypoints_from_flat_obj(obj)

            if not library or not document or not page_id:
                continue
            if x1 is None or y1 is None or w is None or h is None:
                continue

            x1 = float(x1)
            y1 = float(y1)
            w = float(w)
            h = float(h)
            if w <= 0 or h <= 0:
                continue
            x2 = x1 + w
            y2 = y1 + h

            dedupe_key = (
                library,
                document,
                page_id,
                round(x1, 3),
                round(y1, 3),
                round(w, 3),
                round(h, 3),
                tuple(keypoints),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            det_by_key[(library, document, page_id)].append(
                {
                    "library": library,
                    "document": document,
                    "page_id": page_id,
                    "person_name": normalize_name(obj.get("person_name"))
                    or "detected_face",
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "bbox_w": w,
                    "bbox_h": h,
                    "keypoints": [{"x": x, "y": y} for x, y in keypoints],
                    "confidence": obj.get("confidence"),
                    "crop_name": obj.get("crop_name"),
                    "image_name": obj.get("image_name"),
                }
            )

    return dict(det_by_key)


def bbox_iou(a: dict, b: dict) -> float:
    """
    Compute intersection-over-union for two bounding boxes.
    """
    ix1 = max(a["bbox_x1"], b["bbox_x1"])
    iy1 = max(a["bbox_y1"], b["bbox_y1"])
    ix2 = min(a["bbox_x2"], b["bbox_x2"])
    iy2 = min(a["bbox_y2"], b["bbox_y2"])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a["bbox_w"]) * max(0.0, a["bbox_h"])
    area_b = max(0.0, b["bbox_w"]) * max(0.0, b["bbox_h"])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def merge_overlapping_detections(
    anns: list[dict], iou_threshold: float = 0.5
) -> list[dict]:
    """
    Merge strongly overlapping detection boxes on each page. This reduces duplicate detections before all-person pointing is scored.
    """
    if len(anns) <= 1:
        out = [dict(a) for a in anns]
        for a in out:
            a["merged_from_count"] = 1
        return out

    parent = list(range(len(anns)))

    def find(x: int) -> int:
        """
        Return the disjoint-set representative for a detection index.
        """
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        """
        Merge two detection indices in the disjoint-set structure. The function is used when two boxes should belong to the same component.
        """
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(anns)):
        for j in range(i + 1, len(anns)):
            if bbox_iou(anns[i], anns[j]) >= iou_threshold:
                union(i, j)

    components = defaultdict(list)
    for i in range(len(anns)):
        components[find(i)].append(i)

    merged = []
    for inds in components.values():
        if len(inds) == 1:
            ann = dict(anns[inds[0]])
            ann["merged_from_count"] = 1
            merged.append(ann)
            continue

        x1 = min(anns[i]["bbox_x1"] for i in inds)
        y1 = min(anns[i]["bbox_y1"] for i in inds)
        x2 = max(anns[i]["bbox_x2"] for i in inds)
        y2 = max(anns[i]["bbox_y2"] for i in inds)

        kp_seen = set()
        keypoints = []
        for i in inds:
            for kp in anns[i].get("keypoints", []) or []:
                k = (round(float(kp["x"]), 3), round(float(kp["y"]), 3))
                if k not in kp_seen:
                    kp_seen.add(k)
                    keypoints.append({"x": k[0], "y": k[1]})

        source_bboxes = [
            {
                "bbox_x1": anns[i]["bbox_x1"],
                "bbox_y1": anns[i]["bbox_y1"],
                "bbox_x2": anns[i]["bbox_x2"],
                "bbox_y2": anns[i]["bbox_y2"],
                "bbox_w": anns[i]["bbox_w"],
                "bbox_h": anns[i]["bbox_h"],
            }
            for i in inds
        ]

        ann = dict(anns[inds[0]])
        ann.update(
            {
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "bbox_w": x2 - x1,
                "bbox_h": y2 - y1,
                "keypoints": keypoints,
                "merged_from_count": len(inds),
                "source_bboxes": source_bboxes,
            }
        )
        merged.append(ann)

    merged.sort(key=lambda a: (a["bbox_y1"], a["bbox_x1"], a["bbox_y2"], a["bbox_x2"]))
    return merged


def score_point_all_predictions(
    pred_points_px: list[dict], gt_anns: list[dict], parse_fail: int
) -> dict:
    """
    Score a set of predicted points against all reference face boxes on a page. The function returns count, precision/recall, distance, and hit-threshold metrics.
    """
    matches = (
        max_inbbox_match_points_to_anns(pred_points_px, gt_anns)
        if pred_points_px
        else []
    )
    tp = len(matches)
    n_pred = len(pred_points_px)
    n_gt = len(gt_anns)
    fp = max(0, n_pred - tp)
    fn = max(0, n_gt - tp)
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    count_abs_error = abs(n_pred - n_gt)
    count_exact = int(n_pred == n_gt)

    matched_dists = []
    matched_rel_dists = []
    for m in matches:
        pred = pred_points_px[m["pred_idx"]]
        ann = gt_anns[m["ann_idx"]]
        d = dist_to_center(pred["x_px"], pred["y_px"], ann)
        matched_dists.append(d["dist_px"])
        if d["dist_rel_bbox"] is not None:
            matched_rel_dists.append(d["dist_rel_bbox"])

    nearest_center_matches = (
        nearest_center_greedy_match_points_to_anns(pred_points_px, gt_anns)
        if pred_points_px
        else []
    )
    nearest_center_dists = [float(m["dist_px"]) for m in nearest_center_matches]
    nearest_center_hits = {
        t: sum(1 for d in nearest_center_dists if d <= t) for t in CENTER_THRESH_PX
    }

    return {
        "n_gt_faces": n_gt,
        "success": int(tp == n_gt and fp == 0 and fn == 0 and parse_fail == 0),
        "tp_bbox": tp,
        "fp_bbox": fp,
        "fn_bbox": fn,
        "precision_bbox": precision,
        "recall_bbox": recall,
        "f1_bbox": f1,
        "count_abs_error": count_abs_error,
        "count_exact": count_exact,
        "mean_matched_dist_px": (
            (sum(matched_dists) / len(matched_dists)) if matched_dists else None
        ),
        "median_matched_dist_px": (
            statistics.median(matched_dists) if matched_dists else None
        ),
        "mean_matched_dist_rel_bbox": (
            (sum(matched_rel_dists) / len(matched_rel_dists))
            if matched_rel_dists
            else None
        ),
        "n_nearest_center_matches": len(nearest_center_matches),
        "mean_nearest_center_dist_px": (
            (sum(nearest_center_dists) / len(nearest_center_dists))
            if nearest_center_dists
            else None
        ),
        "median_nearest_center_dist_px": (
            statistics.median(nearest_center_dists) if nearest_center_dists else None
        ),
        "nearest_center_hits": nearest_center_hits,
        "nearest_center_matches": nearest_center_matches,
        "matches": matches,
    }


def resolve_detections_path(args, ann_path: Path):
    """
    Find the detections JSONL path from explicit arguments or common dataset locations.
    """
    if args.detections_jsonl:
        det_path = Path(args.detections_jsonl).expanduser().resolve()
    else:
        det_path = ann_path.with_name("people_gator__detections.jsonl")
    if not det_path.exists():
        raise FileNotFoundError(
            "point_all_person_photos needs --detections-jsonl with all detections "
            f"(not found: {det_path})"
        )
    return det_path


def iter_export_images(root: Path):
    """
    Iterate over exported page images and yield their stable page keys with paths.
    """
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        rel = p.relative_to(root)
        if len(rel.parts) < 3:
            continue
        library = rel.parts[0]
        document = rel.parts[1]
        page_id = p.stem
        yield p, library, document, page_id


def anns_grouped_by_name(anns: list[dict]):
    """
    Group page annotations by normalized person name.
    """
    grouped = defaultdict(list)
    for ann in anns:
        grouped[ann["person_name"]].append(ann)
    return dict(grouped)


def sample_negative_names(
    global_name_pool: list[str], names_on_page: set[str], k: int, rng: random.Random
):
    """
    Sample names that are absent from the current page. The candidate list is shuffled with the provided random generator for reproducibility.
    """
    candidates = [name for name in global_name_pool if name not in names_on_page]
    if k <= 0 or not candidates:
        return []
    if len(candidates) <= k:
        return candidates[:]
    return rng.sample(candidates, k)


def gt_face_center(ann: dict):
    """
    Compute the ground-truth face center from keypoints when available, otherwise from the bounding box.
    """
    kps = ann.get("keypoints", [])
    if kps:
        cx = sum(k["x"] for k in kps) / len(kps)
        cy = sum(k["y"] for k in kps) / len(kps)
        return cx, cy, "keypoint_centroid"
    return (
        ann["bbox_x1"] + ann["bbox_w"] / 2.0,
        ann["bbox_y1"] + ann["bbox_h"] / 2.0,
        "bbox_center",
    )


def point_in_bbox(x_px: float, y_px: float, ann: dict):
    """
    Check whether a point lies inside a reference bounding box.
    """
    return (
        ann["bbox_x1"] <= x_px <= ann["bbox_x2"]
        and ann["bbox_y1"] <= y_px <= ann["bbox_y2"]
    )


def dist_to_center(x_px: float, y_px: float, ann: dict):
    """
    Compute the Euclidean distance from a point to the center of a reference annotation.
    """
    cx, cy, source = gt_face_center(ann)
    dx, dy = x_px - cx, y_px - cy
    dist = math.sqrt(dx * dx + dy * dy)
    diag = math.sqrt(ann["bbox_w"] ** 2 + ann["bbox_h"] ** 2)
    return {
        "gt_center_x": cx,
        "gt_center_y": cy,
        "gt_center_source": source,
        "dist_px": dist,
        "dist_rel_bbox": dist / diag if diag > 0 else None,
    }


def max_inbbox_match_points_to_anns(pred_points_px: list[dict], anns: list[dict]):
    """
    Find a maximum bipartite matching between predicted points and boxes using inside-box hits. Each point and annotation can be matched at most once.
    """
    adj = []
    for i, p in enumerate(pred_points_px):
        candidates = []
        for j, ann in enumerate(anns):
            if point_in_bbox(p["x_px"], p["y_px"], ann):
                d = dist_to_center(p["x_px"], p["y_px"], ann)["dist_px"]
                candidates.append((d, j))
        candidates.sort(key=lambda z: z[0])
        adj.append([j for _, j in candidates])

    match_to_pred = {}

    def dfs(pred_idx: int, seen: set[int]):
        """
        Search for an augmenting path in the point-to-annotation matching graph.
        """
        for ann_idx in adj[pred_idx]:
            if ann_idx in seen:
                continue
            seen.add(ann_idx)
            if ann_idx not in match_to_pred or dfs(match_to_pred[ann_idx], seen):
                match_to_pred[ann_idx] = pred_idx
                return True
        return False

    for pred_idx in sorted(range(len(pred_points_px)), key=lambda i: len(adj[i])):
        dfs(pred_idx, set())

    matches = []
    for ann_idx, pred_idx in match_to_pred.items():
        d = dist_to_center(
            pred_points_px[pred_idx]["x_px"],
            pred_points_px[pred_idx]["y_px"],
            anns[ann_idx],
        )["dist_px"]
        matches.append({"pred_idx": pred_idx, "ann_idx": ann_idx, "dist_px": d})
    matches.sort(key=lambda m: (m["ann_idx"], m["pred_idx"]))
    return matches


def nearest_center_greedy_match_points_to_anns(
    pred_points_px: list[dict], anns: list[dict]
):
    """
    Greedily match predictions to annotations by nearest face center.
    """
    pairs = []
    for i, p in enumerate(pred_points_px):
        for j, ann in enumerate(anns):
            d = dist_to_center(p["x_px"], p["y_px"], ann)["dist_px"]
            pairs.append((d, i, j))

    pairs.sort(key=lambda z: z[0])
    used_preds = set()
    used_anns = set()
    matches = []
    for d, i, j in pairs:
        if i in used_preds or j in used_anns:
            continue
        used_preds.add(i)
        used_anns.add(j)
        matches.append({"pred_idx": i, "ann_idx": j, "dist_px": d})

    matches.sort(key=lambda m: (m["ann_idx"], m["pred_idx"]))
    return matches


def center_accuracy_from_dist_rows(rows: list[dict], dist_key: str, denominator: int):
    """
    Compute center-distance accuracies for several pixel thresholds. Missing matches are treated as misses for all thresholds.
    """
    if denominator <= 0:
        return {str(t): None for t in CENTER_THRESH_PX}
    dists = [float(r[dist_key]) for r in rows if r.get(dist_key) is not None]
    return {
        str(t): 100.0 * sum(1 for d in dists if d <= t) / denominator
        for t in CENTER_THRESH_PX
    }


def center_accuracy_from_hit_columns(rows: list[dict], n_gt_key: str, suffix: str = ""):
    """
    Compute center-distance accuracies from already materialized hit columns.
    """
    denom = sum(int(r.get(n_gt_key, 0) or 0) for r in rows)
    if denom <= 0:
        return {str(t): None for t in CENTER_THRESH_PX}
    out = {}
    for t in CENTER_THRESH_PX:
        hit_key = f"nearest_center_hits_at_{t}px{suffix}"
        out[str(t)] = 100.0 * sum(int(r.get(hit_key, 0) or 0) for r in rows) / denom
    return out


def parse_points_from_text(text: str):
    """
    Parse Molmo-style <point> and <points> coordinates from generated text. Coordinates are returned in normalized 0-100 units.
    """
    out = []
    for m in POINT_TAG_RE.finditer(text):
        x100 = max(0.0, min(100.0, float(m.group(1))))
        y100 = max(0.0, min(100.0, float(m.group(2))))
        label = re.sub(r"\s+", " ", m.group(3)).strip()
        out.append({"x100": x100, "y100": y100, "label": label})
    if out:
        return out

    points_tag = POINTS_TAG_RE.search(text)
    if points_tag:
        attrs = dict(ATTR_RE.findall(points_tag.group(1)))
        coords = []
        for key, value in attrs.items():
            m = re.fullmatch(r"x(\d+)", key, flags=re.IGNORECASE)
            if not m:
                continue
            idx = int(m.group(1))
            y_key = f"y{idx}"
            if y_key not in attrs:
                continue
            coords.append(
                (
                    idx,
                    max(0.0, min(100.0, float(value))),
                    max(0.0, min(100.0, float(attrs[y_key]))),
                )
            )
        coords.sort(key=lambda z: z[0])
        label = re.sub(r"\s+", " ", points_tag.group(2)).strip()
        return [{"x100": x, "y100": y, "label": label} for _, x, y in coords]

    fallback = []
    for m in ANY_XY_RE.finditer(text):
        x100 = max(0.0, min(100.0, float(m.group(1))))
        y100 = max(0.0, min(100.0, float(m.group(2))))
        fallback.append({"x100": x100, "y100": y100, "label": None})
    return fallback


def is_none_response(text: str):
    """
    Detect the canonical negative answer used by the prompts.
    """
    return bool(NONE_RE.match(text.strip()))


def resize_image(image: Image.Image, resize_long_side: int):
    """
    Resize an image so that the longest side does not exceed the requested limit.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    if resize_long_side <= 0:
        return image
    w, h = image.size
    long_side = max(w, h)
    if long_side <= resize_long_side:
        return image
    scale = resize_long_side / float(long_side)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def first_model_device(model):
    """
    Return the first device that holds model parameters.
    """
    try:
        return model.device
    except Exception:
        pass
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def load_model_and_processor(args):
    """
    Load Molmo and its processor with the requested precision and device placement.
    """
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

    processor_source = (
        str(adapter_dir)
        if (adapter_dir / "processor_config.json").exists()
        else args.model_id
    )
    print("Loading processor from:", processor_source)
    processor = AutoProcessor.from_pretrained(
        processor_source, trust_remote_code=True, torch_dtype="auto"
    )

    if args.force_cpu:
        quantization_config = None
        device_map = None
        max_memory = None
        torch_dtype = torch.float32
    else:
        device_map_arg = args.device_map.strip() or None
        max_memory = parse_max_memory(args.max_memory)
        if device_map_arg == "block":
            visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            device_map = build_molmo_block_device_map(visible_gpus)
            print("Using device_map: block/manual whole-block map")
            print("Visible CUDA devices:", visible_gpus)
            print("Manual HF device map:", device_map)
        else:
            device_map = device_map_arg
        if args.disable_4bit:
            quantization_config = None
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            torch_dtype = torch.bfloat16

    print("Loading base model:", args.model_id)
    print(
        "device_map:",
        device_map,
        "max_memory:",
        max_memory,
        "4bit:",
        quantization_config is not None,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=device_map,
        max_memory=max_memory,
    )
    if args.force_cpu:
        base = base.to("cpu")

    if device_map:
        mark_model_parallel(base)
        hf_map = getattr(base, "hf_device_map", None)
        print("HF device map:", hf_map)

    print("Loading LoRA adapter:", adapter_dir)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    if device_map:
        model = align_lora_devices_with_base(model)
        mark_model_parallel(model)
    model.eval()
    print("Loaded. Input device:", first_model_device(model))
    return model, processor


def molmo_generate(model, processor, image_path: Path, prompt: str, args):
    """
    Run one Molmo generation call for an image and prompt.
    """
    original = Image.open(image_path)
    original_size = original.size
    image = resize_image(original, args.resize_long_side)
    inputs = processor.process(
        images=[image],
        text=prompt,
        images_kwargs={"max_crops": args.max_crops},
        text_kwargs={"sequence_length": args.sequence_length},
    )
    inputs = {
        k: (v if torch.is_tensor(v) else torch.tensor(v)) for k, v in inputs.items()
    }
    device = first_model_device(model)
    inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16
        ):
            generate_from_batch = getattr(model, "generate_from_batch", None)
            if generate_from_batch is None and hasattr(model, "get_base_model"):
                generate_from_batch = getattr(
                    model.get_base_model(), "generate_from_batch", None
                )
            if generate_from_batch is None:
                raise AttributeError(
                    "Loaded model does not expose Molmo generate_from_batch()."
                )
            out = generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    stop_strings="<|endoftext|>",
                ),
                tokenizer=processor.tokenizer,
            )
    gen_tokens = out[0, inputs["input_ids"].size(1) :]
    text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    points = parse_points_from_text(text)
    return text, points, original_size


def point_to_pixels(x100: float, y100: float, img_size: tuple[int, int]):
    """
    Convert normalized Molmo coordinates to pixel coordinates for the current image size.
    """
    w, h = img_size
    return (x100 / 100.0) * w, (y100 / 100.0) * h


def save_failure_overlay(
    out_dir: Path,
    row: dict,
    img_path: Path,
    all_anns: list[dict],
    target_anns: list[dict],
    pred_points_px: list[dict],
    idx: int,
):
    """
    Save a diagnostic overlay showing predictions, reference boxes, and the evaluated prompt. The image is written only for selected failure cases.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for ann in all_anns:
        draw.rectangle(
            [ann["bbox_x1"], ann["bbox_y1"], ann["bbox_x2"], ann["bbox_y2"]],
            outline="red",
            width=3,
        )
        cx, cy, _ = gt_face_center(ann)
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline="blue", width=3)
    for ann in target_anns:
        draw.rectangle(
            [ann["bbox_x1"], ann["bbox_y1"], ann["bbox_x2"], ann["bbox_y2"]],
            outline="yellow",
            width=5,
        )
    for p in pred_points_px:
        x, y = p["x_px"], p["y_px"]
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], outline="lime", width=4)
    stem = f"{idx:04d}__{row['eval_mode']}__{row['library']}__{row['document']}__{row['page_id']}"
    img.save(out_dir / f"{stem}.png", quality=95)
    with (out_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)


def eval_query(
    model,
    processor,
    args,
    img_path: Path,
    library: str,
    document: str,
    page_id: str,
    anns: list[dict],
    mode: str,
    target_name: str,
    target_anns: list[dict],
    expected_present: int,
):
    """
    Evaluate one name-conditioned query for a page and target name. It records text output, parsed points, hit status, and distance diagnostics.
    """
    if mode == "point_person_by_name":
        prompt = PROMPT_POINT_PERSON_BY_NAME.format(name=target_name)
    elif mode == "is_person_present":
        prompt = PROMPT_IS_PERSON_PRESENT.format(name=target_name)
    else:
        raise ValueError(mode)

    raw, points, img_size = molmo_generate(model, processor, img_path, prompt, args)
    pred_none = int(is_none_response(raw))
    pred_points_px = []
    for p in points:
        x_px, y_px = point_to_pixels(p["x100"], p["y100"], img_size)
        pred_points_px.append({**p, "x_px": x_px, "y_px": y_px})

    row = {
        "eval_mode": mode,
        "library": library,
        "document": document,
        "page_id": page_id,
        "image_path": str(img_path),
        "target_person_name": target_name,
        "expected_present": expected_present,
        "predicted_none": pred_none,
        "n_faces_on_page": len(anns),
        "n_target_faces_for_name": len(target_anns),
        "n_pred_points": len(points),
        "success": 0,
        "hit_any_target_bbox": 0,
        "parse_fail": 0,
        "raw": raw,
        "answer": raw,
    }

    if expected_present == 0:
        row["success"] = int(pred_none == 1)
        row["false_positive_point"] = int(pred_none == 0)
        if pred_points_px:
            row["pred_x100"] = pred_points_px[0]["x100"]
            row["pred_y100"] = pred_points_px[0]["y100"]
            row["pred_x_px"] = pred_points_px[0]["x_px"]
            row["pred_y_px"] = pred_points_px[0]["y_px"]
        return row, pred_points_px

    if pred_none == 1:
        row["success"] = 0
        row["missed_as_none"] = 1
        return row, pred_points_px
    if not pred_points_px:
        row["parse_fail"] = 1
        return row, pred_points_px

    p = pred_points_px[0]
    metrics = []
    for ann in target_anns:
        d = dist_to_center(p["x_px"], p["y_px"], ann)
        hit = int(point_in_bbox(p["x_px"], p["y_px"], ann))
        metrics.append((ann, d, hit))
    any_hit = any(hit for _, _, hit in metrics)
    best_ann, best_d, _ = min(metrics, key=lambda z: z[1]["dist_px"])

    row.update(
        {
            "success": int(any_hit),
            "hit_any_target_bbox": int(any_hit),
            "pred_x100": p["x100"],
            "pred_y100": p["y100"],
            "pred_x_px": p["x_px"],
            "pred_y_px": p["y_px"],
            "nearest_target_center_x": best_d["gt_center_x"],
            "nearest_target_center_y": best_d["gt_center_y"],
            "nearest_target_center_source": best_d["gt_center_source"],
            "dist_px_to_nearest_target": best_d["dist_px"],
            "dist_rel_to_nearest_target_bbox": best_d["dist_rel_bbox"],
        }
    )
    return row, pred_points_px


def eval_point_all_person_photos(
    model,
    processor,
    args,
    img_path: Path,
    library: str,
    document: str,
    page_id: str,
    raw_det_anns: list[dict],
    dedup_det_anns: list[dict],
):
    """
    Evaluate the page-level task that asks for all person photographs.
    """
    prompt = PROMPT_POINT_ALL_PERSON_PHOTOS
    raw, points, img_size = molmo_generate(model, processor, img_path, prompt, args)
    pred_none = int(is_none_response(raw))

    pred_points_px = []
    if pred_none == 0:
        for p in points:
            x_px, y_px = point_to_pixels(p["x100"], p["y100"], img_size)
            pred_points_px.append({**p, "x_px": x_px, "y_px": y_px})

    parse_fail = int(pred_none == 0 and len(pred_points_px) == 0)

    strict = score_point_all_predictions(pred_points_px, raw_det_anns, parse_fail)
    dedup = score_point_all_predictions(pred_points_px, dedup_det_anns, parse_fail)

    n_raw = len(raw_det_anns)
    n_dedup = len(dedup_det_anns)
    merged_away = max(0, n_raw - n_dedup)

    row = {
        "eval_mode": "point_all_person_photos",
        "library": library,
        "document": document,
        "page_id": page_id,
        "image_path": str(img_path),
        "target_person_name": "__all_person_photos__",
        "expected_present": int(n_dedup > 0),
        "predicted_none": pred_none,
        "n_pred_points": len(pred_points_px),
        "parse_fail": parse_fail,
        "n_faces_on_page": n_dedup,
        "n_gt_faces": n_dedup,
        "success": dedup["success"],
        "tp_bbox": dedup["tp_bbox"],
        "fp_bbox": dedup["fp_bbox"],
        "fn_bbox": dedup["fn_bbox"],
        "precision_bbox": dedup["precision_bbox"],
        "recall_bbox": dedup["recall_bbox"],
        "f1_bbox": dedup["f1_bbox"],
        "count_abs_error": dedup["count_abs_error"],
        "count_exact": dedup["count_exact"],
        "mean_matched_dist_px": dedup["mean_matched_dist_px"],
        "median_matched_dist_px": dedup["median_matched_dist_px"],
        "mean_matched_dist_rel_bbox": dedup["mean_matched_dist_rel_bbox"],
        "n_nearest_center_matches": dedup["n_nearest_center_matches"],
        "mean_nearest_center_dist_px": dedup["mean_nearest_center_dist_px"],
        "median_nearest_center_dist_px": dedup["median_nearest_center_dist_px"],
        "n_gt_faces_strict_with_duplicates": strict["n_gt_faces"],
        "success_strict_with_duplicates": strict["success"],
        "tp_bbox_strict_with_duplicates": strict["tp_bbox"],
        "fp_bbox_strict_with_duplicates": strict["fp_bbox"],
        "fn_bbox_strict_with_duplicates": strict["fn_bbox"],
        "precision_bbox_strict_with_duplicates": strict["precision_bbox"],
        "recall_bbox_strict_with_duplicates": strict["recall_bbox"],
        "f1_bbox_strict_with_duplicates": strict["f1_bbox"],
        "count_abs_error_strict_with_duplicates": strict["count_abs_error"],
        "count_exact_strict_with_duplicates": strict["count_exact"],
        "mean_matched_dist_px_strict_with_duplicates": strict["mean_matched_dist_px"],
        "median_matched_dist_px_strict_with_duplicates": strict[
            "median_matched_dist_px"
        ],
        "mean_matched_dist_rel_bbox_strict_with_duplicates": strict[
            "mean_matched_dist_rel_bbox"
        ],
        "n_nearest_center_matches_strict_with_duplicates": strict[
            "n_nearest_center_matches"
        ],
        "mean_nearest_center_dist_px_strict_with_duplicates": strict[
            "mean_nearest_center_dist_px"
        ],
        "median_nearest_center_dist_px_strict_with_duplicates": strict[
            "median_nearest_center_dist_px"
        ],
        "n_gt_faces_without_duplicates": dedup["n_gt_faces"],
        "success_without_duplicates": dedup["success"],
        "tp_bbox_without_duplicates": dedup["tp_bbox"],
        "fp_bbox_without_duplicates": dedup["fp_bbox"],
        "fn_bbox_without_duplicates": dedup["fn_bbox"],
        "precision_bbox_without_duplicates": dedup["precision_bbox"],
        "recall_bbox_without_duplicates": dedup["recall_bbox"],
        "f1_bbox_without_duplicates": dedup["f1_bbox"],
        "count_abs_error_without_duplicates": dedup["count_abs_error"],
        "count_exact_without_duplicates": dedup["count_exact"],
        "mean_matched_dist_px_without_duplicates": dedup["mean_matched_dist_px"],
        "median_matched_dist_px_without_duplicates": dedup["median_matched_dist_px"],
        "mean_matched_dist_rel_bbox_without_duplicates": dedup[
            "mean_matched_dist_rel_bbox"
        ],
        "n_nearest_center_matches_without_duplicates": dedup[
            "n_nearest_center_matches"
        ],
        "mean_nearest_center_dist_px_without_duplicates": dedup[
            "mean_nearest_center_dist_px"
        ],
        "median_nearest_center_dist_px_without_duplicates": dedup[
            "median_nearest_center_dist_px"
        ],
        "n_detections_merged_away": merged_away,
        "has_duplicate_gt_detections": int(merged_away > 0),
        "dedupe_iou_threshold": args.dedupe_iou_threshold,
        "raw": raw,
        "answer": raw,
        "pred_points_json": json.dumps(pred_points_px, ensure_ascii=False),
        "matches_json": json.dumps(dedup["matches"], ensure_ascii=False),
        "matches_without_duplicates_json": json.dumps(
            dedup["matches"], ensure_ascii=False
        ),
        "matches_strict_with_duplicates_json": json.dumps(
            strict["matches"], ensure_ascii=False
        ),
        "nearest_center_matches_json": json.dumps(
            dedup["nearest_center_matches"], ensure_ascii=False
        ),
        "nearest_center_matches_without_duplicates_json": json.dumps(
            dedup["nearest_center_matches"], ensure_ascii=False
        ),
        "nearest_center_matches_strict_with_duplicates_json": json.dumps(
            strict["nearest_center_matches"], ensure_ascii=False
        ),
    }
    for t in CENTER_THRESH_PX:
        row[f"nearest_center_hits_at_{t}px"] = dedup["nearest_center_hits"].get(t, 0)
        row[f"nearest_center_accuracy_at_{t}px"] = (
            (100.0 * dedup["nearest_center_hits"].get(t, 0) / dedup["n_gt_faces"])
            if dedup["n_gt_faces"]
            else None
        )
        row[f"nearest_center_hits_at_{t}px_without_duplicates"] = dedup[
            "nearest_center_hits"
        ].get(t, 0)
        row[f"nearest_center_accuracy_at_{t}px_without_duplicates"] = (
            (100.0 * dedup["nearest_center_hits"].get(t, 0) / dedup["n_gt_faces"])
            if dedup["n_gt_faces"]
            else None
        )
        row[f"nearest_center_hits_at_{t}px_strict_with_duplicates"] = strict[
            "nearest_center_hits"
        ].get(t, 0)
        row[f"nearest_center_accuracy_at_{t}px_strict_with_duplicates"] = (
            (100.0 * strict["nearest_center_hits"].get(t, 0) / strict["n_gt_faces"])
            if strict["n_gt_faces"]
            else None
        )
    return row, pred_points_px


def summarize(rows: list[dict]):
    """
    Aggregate per-example evaluation rows into task-level metrics. The summary includes classification, localization, and count-based measurements.
    """
    summary = {
        "rows_evaluated": len(rows),
        "by_mode": {},
    }
    for mode in sorted({r["eval_mode"] for r in rows}):
        mode_rows = [r for r in rows if r["eval_mode"] == mode]

        if mode == "point_all_person_photos":

            def point_all_summary_for(prefix: str):
                success_key = "success" if prefix == "" else f"success_{prefix}"
                precision_key = (
                    "precision_bbox" if prefix == "" else f"precision_bbox_{prefix}"
                )
                recall_key = "recall_bbox" if prefix == "" else f"recall_bbox_{prefix}"
                f1_key = "f1_bbox" if prefix == "" else f"f1_bbox_{prefix}"
                count_exact_key = (
                    "count_exact" if prefix == "" else f"count_exact_{prefix}"
                )
                count_abs_error_key = (
                    "count_abs_error" if prefix == "" else f"count_abs_error_{prefix}"
                )
                mean_dist_key = (
                    "mean_matched_dist_px"
                    if prefix == ""
                    else f"mean_matched_dist_px_{prefix}"
                )
                n_gt_key = "n_gt_faces" if prefix == "" else f"n_gt_faces_{prefix}"
                tp_key = "tp_bbox" if prefix == "" else f"tp_bbox_{prefix}"
                fp_key = "fp_bbox" if prefix == "" else f"fp_bbox_{prefix}"
                fn_key = "fn_bbox" if prefix == "" else f"fn_bbox_{prefix}"

                precisions = [float(r.get(precision_key, 0.0)) for r in mode_rows]
                recalls = [float(r.get(recall_key, 0.0)) for r in mode_rows]
                f1s = [float(r.get(f1_key, 0.0)) for r in mode_rows]
                count_exacts = [int(r.get(count_exact_key, 0)) for r in mode_rows]
                count_abs_errors = [
                    int(r.get(count_abs_error_key, 0)) for r in mode_rows
                ]
                mean_matched_dists = [
                    float(r[mean_dist_key])
                    for r in mode_rows
                    if r.get(mean_dist_key) is not None
                ]

                center_suffix = "" if prefix == "" else f"_{prefix}"
                nearest_center_acc_px = center_accuracy_from_hit_columns(
                    mode_rows, n_gt_key, center_suffix
                )
                mean_center_dist_key = (
                    "mean_nearest_center_dist_px"
                    if prefix == ""
                    else f"mean_nearest_center_dist_px_{prefix}"
                )
                mean_center_dists = [
                    float(r[mean_center_dist_key])
                    for r in mode_rows
                    if r.get(mean_center_dist_key) is not None
                ]

                return {
                    "rows": len(mode_rows),
                    "overall_success_percent": (
                        100.0
                        * sum(r.get(success_key, 0) for r in mode_rows)
                        / len(mode_rows)
                        if mode_rows
                        else None
                    ),
                    "mean_precision_bbox": (
                        sum(precisions) / len(precisions) if precisions else None
                    ),
                    "mean_recall_bbox": (
                        sum(recalls) / len(recalls) if recalls else None
                    ),
                    "mean_f1_bbox": sum(f1s) / len(f1s) if f1s else None,
                    "count_exact_percent": (
                        100.0 * sum(count_exacts) / len(count_exacts)
                        if count_exacts
                        else None
                    ),
                    "mean_count_abs_error": (
                        sum(count_abs_errors) / len(count_abs_errors)
                        if count_abs_errors
                        else None
                    ),
                    "median_count_abs_error": (
                        statistics.median(count_abs_errors)
                        if count_abs_errors
                        else None
                    ),
                    "sum_gt_faces": sum(int(r.get(n_gt_key, 0)) for r in mode_rows),
                    "sum_pred_points": sum(
                        int(r.get("n_pred_points", 0)) for r in mode_rows
                    ),
                    "sum_tp_bbox": sum(int(r.get(tp_key, 0)) for r in mode_rows),
                    "sum_fp_bbox": sum(int(r.get(fp_key, 0)) for r in mode_rows),
                    "sum_fn_bbox": sum(int(r.get(fn_key, 0)) for r in mode_rows),
                    "predicted_none_percent": (
                        100.0
                        * sum(r.get("predicted_none", 0) for r in mode_rows)
                        / len(mode_rows)
                        if mode_rows
                        else None
                    ),
                    "parse_fail_count": sum(r.get("parse_fail", 0) for r in mode_rows),
                    "mean_matched_dist_px_mean_over_pages": (
                        sum(mean_matched_dists) / len(mean_matched_dists)
                        if mean_matched_dists
                        else None
                    ),
                    "mean_nearest_center_dist_px_mean_over_pages": (
                        sum(mean_center_dists) / len(mean_center_dists)
                        if mean_center_dists
                        else None
                    ),
                    "nearest_center_accuracy_px": nearest_center_acc_px,
                }

            without_duplicates = point_all_summary_for("without_duplicates")
            strict_with_duplicates = point_all_summary_for("strict_with_duplicates")
            summary["by_mode"][mode] = {
                "rows": len(mode_rows),
                "overall_success_percent": without_duplicates[
                    "overall_success_percent"
                ],
                "mean_precision_bbox": without_duplicates["mean_precision_bbox"],
                "mean_recall_bbox": without_duplicates["mean_recall_bbox"],
                "mean_f1_bbox": without_duplicates["mean_f1_bbox"],
                "count_exact_percent": without_duplicates["count_exact_percent"],
                "mean_count_abs_error": without_duplicates["mean_count_abs_error"],
                "median_count_abs_error": without_duplicates["median_count_abs_error"],
                "predicted_none_percent": without_duplicates["predicted_none_percent"],
                "parse_fail_count": without_duplicates["parse_fail_count"],
                "mean_matched_dist_px_mean_over_pages": without_duplicates[
                    "mean_matched_dist_px_mean_over_pages"
                ],
                "mean_nearest_center_dist_px_mean_over_pages": without_duplicates[
                    "mean_nearest_center_dist_px_mean_over_pages"
                ],
                "nearest_center_accuracy_px": without_duplicates[
                    "nearest_center_accuracy_px"
                ],
                "without_duplicates": without_duplicates,
                "strict_with_duplicates": strict_with_duplicates,
                "duplicate_gt_pages": sum(
                    int(r.get("has_duplicate_gt_detections", 0)) for r in mode_rows
                ),
                "duplicate_gt_pages_percent": (
                    100.0
                    * sum(
                        int(r.get("has_duplicate_gt_detections", 0)) for r in mode_rows
                    )
                    / len(mode_rows)
                    if mode_rows
                    else None
                ),
                "detections_merged_away_total": sum(
                    int(r.get("n_detections_merged_away", 0)) for r in mode_rows
                ),
                "dedupe_iou_threshold": (
                    mode_rows[0].get("dedupe_iou_threshold") if mode_rows else None
                ),
                "note": "overall_success_percent and the legacy flat bbox metrics use without_duplicates. strict_with_duplicates uses raw detections exactly as loaded from people_gator__detections.jsonl.",
            }
            continue

        pos = [r for r in mode_rows if r.get("expected_present") == 1]
        neg = [r for r in mode_rows if r.get("expected_present") == 0]
        pos_with_dist = [
            r for r in pos if r.get("dist_px_to_nearest_target") is not None
        ]
        dists = [float(r["dist_px_to_nearest_target"]) for r in pos_with_dist]
        summary["by_mode"][mode] = {
            "rows": len(mode_rows),
            "overall_success_percent": (
                100.0 * sum(r.get("success", 0) for r in mode_rows) / len(mode_rows)
                if mode_rows
                else None
            ),
            "positive_rows": len(pos),
            "positive_success_percent": (
                100.0 * sum(r.get("success", 0) for r in pos) / len(pos)
                if pos
                else None
            ),
            "negative_rows": len(neg),
            "negative_none_success_percent": (
                100.0 * sum(r.get("success", 0) for r in neg) / len(neg)
                if neg
                else None
            ),
            "predicted_none_percent": (
                100.0
                * sum(r.get("predicted_none", 0) for r in mode_rows)
                / len(mode_rows)
                if mode_rows
                else None
            ),
            "parse_fail_count": sum(r.get("parse_fail", 0) for r in mode_rows),
            "mean_dist_px_to_nearest_target": (
                sum(dists) / len(dists) if dists else None
            ),
            "median_dist_px_to_nearest_target": (
                statistics.median(dists) if dists else None
            ),
            "nearest_center_accuracy_px": center_accuracy_from_dist_rows(
                pos, "dist_px_to_nearest_target", len(pos)
            ),
        }
    return summary


def main():
    """
    Run the full evaluation pipeline from loading data to writing CSV and JSON summaries.
    """
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    ann_path = Path(args.annotations_jsonl).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = out_dir / "failures"

    if not data_root.exists():
        raise FileNotFoundError(f"data-root not found: {data_root}")
    if not ann_path.exists():
        raise FileNotFoundError(f"annotations-jsonl not found: {ann_path}")

    model, processor = load_model_and_processor(args)

    print("Loading annotations:", ann_path)
    ann_by_key = load_peoplegator_annotations(ann_path)
    name_pool = sorted(
        {
            ann["person_name"]
            for anns in ann_by_key.values()
            for ann in anns
            if ann.get("person_name")
        }
    )
    print("Annotated/named pages:", len(ann_by_key), "unique names:", len(name_pool))

    if args.eval_mode == "both":
        modes = ["point_person_by_name", "is_person_present"]
    elif args.eval_mode == "all":
        modes = ["point_person_by_name", "is_person_present", "point_all_person_photos"]
    else:
        modes = [args.eval_mode]

    needs_name_modes = any(
        m in {"point_person_by_name", "is_person_present"} for m in modes
    )
    needs_point_all = "point_all_person_photos" in modes
    point_all_page_keys = set()
    detections_by_key = {}
    det_path = None
    if needs_point_all:
        point_all_page_keys = load_unique_page_keys(ann_path)
        det_path = resolve_detections_path(args, ann_path)
        print("Loading all detections for point_all_person_photos:", det_path)
        detections_by_key = load_peoplegator_detections(det_path)
        print("Point-all page set from annotations:", len(point_all_page_keys))
        print("Pages with all detections:", len(detections_by_key))

    print("Scanning images:", data_root)
    rng = random.Random(args.negative_random_seed)
    max_pages = None if args.max_pages == 0 else args.max_pages

    rows = []
    pages_seen = 0
    skipped_no_annotation = 0
    skipped_no_detections = 0
    skipped_too_many_detections = 0
    failures_saved = 0

    for img_path, library, document, page_id in iter_export_images(data_root):
        key = (library, document, page_id)
        anns = ann_by_key.get(key, [])

        include_name = needs_name_modes and bool(anns)
        raw_det_anns = []
        dedup_det_anns = []
        include_point_all = False
        if needs_point_all and key in point_all_page_keys:
            raw_det_anns = detections_by_key.get(key, [])
            dedup_det_anns = (
                merge_overlapping_detections(raw_det_anns, args.dedupe_iou_threshold)
                if raw_det_anns
                else []
            )
            if not raw_det_anns:
                skipped_no_detections += 1
            elif len(raw_det_anns) > args.max_point_all_detections:
                skipped_too_many_detections += 1
            else:
                include_point_all = True

        if not include_name and not include_point_all:
            skipped_no_annotation += 1
            continue

        rows_before_page = len(rows)
        anns_by_name = anns_grouped_by_name(anns) if anns else {}
        names_on_page = set(anns_by_name.keys())
        negative_names = (
            sample_negative_names(
                name_pool, names_on_page, args.negative_names_per_page, rng
            )
            if include_name
            else []
        )

        for mode in modes:
            if mode == "point_all_person_photos":
                if not include_point_all:
                    continue
                row, pred_points_px = eval_point_all_person_photos(
                    model,
                    processor,
                    args,
                    img_path,
                    library,
                    document,
                    page_id,
                    raw_det_anns,
                    dedup_det_anns,
                )
                rows.append(row)
                if args.print_raw:
                    print("\n--- RAW", mode, "---\n", row["raw"])
                if (
                    args.save_failures
                    and row["success"] == 0
                    and failures_saved < args.max_failures
                ):
                    save_failure_overlay(
                        failures_dir,
                        row,
                        img_path,
                        raw_det_anns,
                        dedup_det_anns,
                        pred_points_px,
                        failures_saved,
                    )
                    failures_saved += 1
                continue

            if not include_name:
                continue

            for target_name, target_anns in anns_by_name.items():
                row, pred_points_px = eval_query(
                    model,
                    processor,
                    args,
                    img_path,
                    library,
                    document,
                    page_id,
                    anns,
                    mode,
                    target_name,
                    target_anns,
                    expected_present=1,
                )
                rows.append(row)
                if args.print_raw:
                    print("\n--- RAW", mode, target_name, "---\n", row["raw"])
                if (
                    args.save_failures
                    and row["success"] == 0
                    and failures_saved < args.max_failures
                ):
                    save_failure_overlay(
                        failures_dir,
                        row,
                        img_path,
                        anns,
                        target_anns,
                        pred_points_px,
                        failures_saved,
                    )
                    failures_saved += 1

            for target_name in negative_names:
                row, pred_points_px = eval_query(
                    model,
                    processor,
                    args,
                    img_path,
                    library,
                    document,
                    page_id,
                    anns,
                    mode,
                    target_name,
                    [],
                    expected_present=0,
                )
                rows.append(row)
                if args.print_raw:
                    print("\n--- RAW", mode, target_name, "NEGATIVE ---\n", row["raw"])
                if (
                    args.save_failures
                    and row["success"] == 0
                    and failures_saved < args.max_failures
                ):
                    save_failure_overlay(
                        failures_dir,
                        row,
                        img_path,
                        anns,
                        [],
                        pred_points_px,
                        failures_saved,
                    )
                    failures_saved += 1

        if len(rows) > rows_before_page:
            pages_seen += 1
            print(
                f"Processed page {pages_seen}: {library}/{document}/{page_id} | rows={len(rows)}"
            )
            if max_pages is not None and pages_seen >= max_pages:
                break

    csv_path = out_dir / "results.csv"
    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        priority = [
            "eval_mode",
            "library",
            "document",
            "page_id",
            "target_person_name",
            "expected_present",
            "success",
            "success_without_duplicates",
            "success_strict_with_duplicates",
            "predicted_none",
            "hit_any_target_bbox",
            "dist_px_to_nearest_target",
            "n_gt_faces",
            "n_gt_faces_without_duplicates",
            "n_gt_faces_strict_with_duplicates",
            "n_detections_merged_away",
            "n_pred_points",
            "n_nearest_center_matches",
            "mean_nearest_center_dist_px",
            "nearest_center_hits_at_10px",
            "nearest_center_hits_at_20px",
            "nearest_center_hits_at_30px",
            "nearest_center_hits_at_50px",
            "nearest_center_hits_at_100px",
            "tp_bbox",
            "fp_bbox",
            "fn_bbox",
            "precision_bbox",
            "recall_bbox",
            "f1_bbox",
            "tp_bbox_without_duplicates",
            "fp_bbox_without_duplicates",
            "fn_bbox_without_duplicates",
            "precision_bbox_without_duplicates",
            "recall_bbox_without_duplicates",
            "f1_bbox_without_duplicates",
            "tp_bbox_strict_with_duplicates",
            "fp_bbox_strict_with_duplicates",
            "fn_bbox_strict_with_duplicates",
            "precision_bbox_strict_with_duplicates",
            "recall_bbox_strict_with_duplicates",
            "f1_bbox_strict_with_duplicates",
            "answer",
            "raw",
        ]
        fieldnames = [k for k in priority if k in fieldnames] + [
            k for k in fieldnames if k not in priority
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    answers_path = out_dir / "answers.txt"
    with answers_path.open("w", encoding="utf-8") as f:
        for r in rows:
            answer = (
                str(r.get("answer", r.get("raw", "")))
                .replace("\r", " ")
                .replace("\n", " ")
                .strip()
            )
            f.write(answer + "\n")

    summary = summarize(rows)
    summary.update(
        {
            "pages_seen": pages_seen,
            "skipped_no_annotation": skipped_no_annotation,
            "skipped_no_detections_for_point_all": skipped_no_detections,
            "skipped_too_many_detections_for_point_all": skipped_too_many_detections,
            "failures_saved": failures_saved,
            "adapter_dir": str(Path(args.adapter_dir).expanduser().resolve()),
            "annotations_jsonl": str(ann_path),
            "detections_jsonl": str(det_path) if det_path is not None else None,
            "data_root": str(data_root),
            "results_csv": str(csv_path),
            "answers_txt": str(answers_path),
            "out_dir": str(out_dir),
            "eval_mode_arg": args.eval_mode,
            "negative_names_per_page": args.negative_names_per_page,
            "resize_long_side": args.resize_long_side,
            "max_crops": args.max_crops,
            "sequence_length": args.sequence_length,
            "max_point_all_detections": args.max_point_all_detections,
            "dedupe_iou_threshold": args.dedupe_iou_threshold,
            "point_all_statistics": "Both without_duplicates and strict_with_duplicates are computed in one run.",
            "modes": modes,
        }
    )

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nResults:", csv_path)
    print("Answers:", answers_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
