#!/usr/bin/env python3
"""
Fine-tune Molmo with LoRA adapters on JSONL instruction data.
The script resolves page images, builds multimodal batches, configures PEFT, and saves adapter checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".jp2", ".j2k"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="allenai/Molmo-7B-D-0924")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument(
        "--images-root",
        required=True,
        help="Either project root ~/diplomka or directly datasets/digiknihovna_data",
    )
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--num-train-epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)

    ap.add_argument("--per-device-train-batch-size", type=int, default=1)
    ap.add_argument("--per-device-eval-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--eval-steps", type=int, default=25)

    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora-target-mode",
        choices=["text", "auto", "all"],
        default="text",
        help=(
            "text = train only language/text projection modules (recommended for 24GB smoke run); "
            "auto/all = include every discovered Linear short name, including vision modules like patch_embedding."
        ),
    )
    ap.add_argument(
        "--lora-target-modules",
        default="",
        help="Optional comma-separated explicit LoRA module short names, e.g. wq,wk,wv,wo,w1,w2,w3.",
    )
    ap.add_argument(
        "--freeze-vision-encoder",
        action="store_true",
        help=(
            "After PEFT/LoRA is attached, force every vision-backbone parameter, including any accidental "
            "vision LoRA parameters, to requires_grad=False. For a truly frozen vision encoder, also use "
            "--lora-target-mode text or explicit text modules: wq,wk,wv,wo,w1,w2,w3."
        ),
    )
    ap.add_argument(
        "--print-trainable-names",
        action="store_true",
        help="Print the exact trainable parameter names after LoRA/freeze setup.",
    )
    ap.add_argument(
        "--debug-token-first-n",
        type=int,
        default=3,
        help="Print/token-log the first N encoded examples seen by the collator. Use 0 to disable.",
    )
    ap.add_argument(
        "--debug-token-log-steps",
        type=int,
        default=0,
        help="If >0, also print/token-log every Nth encoded example after the first N.",
    )
    ap.add_argument(
        "--debug-token-log-file",
        default="",
        help="Optional JSONL file for token/debug stats during training, e.g. runs/debug_tokens.jsonl.",
    )
    ap.add_argument(
        "--resize-long-side",
        type=int,
        default=768,
        help="Resize page image so the longer side is at most this many pixels before Molmo processor. Use 0 to disable.",
    )
    ap.add_argument(
        "--max-crops",
        type=int,
        default=4,
        help=(
            "Molmo image processor max_crops. Default Molmo uses 12, which is usually too much for "
            "24GB QLoRA training on full newspaper pages. Try 2 for smoke test, 4 for 24GB if it fits."
        ),
    )
    ap.add_argument(
        "--sequence-length",
        type=int,
        default=1024,
        help="Molmo text/image sequence_length. Default Molmo uses 1536; lowering saves VRAM.",
    )

    ap.add_argument("--disable-4bit", action="store_true")
    ap.add_argument(
        "--device-map",
        default="",
        choices=["", "auto", "balanced", "balanced_low_0", "sequential", "block"],
        help=(
            "Optional Hugging Face device_map for single-process model sharding across visible GPUs. "
            "Use with CUDA_VISIBLE_DEVICES=0,1,3 and --device-map block to shard by whole transformer blocks and avoid Trainer DataParallel."
        ),
    )
    ap.add_argument(
        "--max-memory",
        default="",
        help=(
            "Optional per-logical-GPU memory cap for device_map, e.g. "
            "0:22GiB,1:22GiB,2:22GiB. IMPORTANT: after CUDA_VISIBLE_DEVICES=0,1,3, "
            "the visible GPUs are numbered 0,1,2 inside Python."
        ),
    )
    ap.add_argument(
        "--no-split-module-classes",
        default="MolmoBlock,OLMoBlock,OLMoSequentialBlock,MolmoSequentialBlock,Block",
        help=(
            "Comma-separated module class names that device_map=auto must not split across GPUs. "
            "This is important for Molmo + PEFT/LoRA, because splitting one transformer block "
            "can leave LoRA tensors and activations on different CUDA devices."
        ),
    )
    return ap.parse_args()


class JsonlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _unique_paths(paths):
    seen = set()
    out = []
    for p in paths:
        p = Path(p).expanduser()
        key = p.as_posix()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _possible_data_roots(images_root: Path):
    images_root = images_root.expanduser().resolve()
    roots = [
        images_root,
        images_root / "datasets" / "digiknihovna_data",
        images_root / "digiknihovna_data",
    ]
    return _unique_paths(roots)


def _strip_known_prefixes(rel_path: str):
    rel_str = Path(str(rel_path)).as_posix().lstrip("./")
    variants = [rel_str]
    for prefix in ["datasets/digiknihovna_data/", "digiknihovna_data/"]:
        if rel_str.startswith(prefix):
            variants.append(rel_str[len(prefix) :])
    return list(dict.fromkeys(variants))


def _candidate_paths(images_root: Path, rel_path: str):
    candidates = []
    stripped_variants = _strip_known_prefixes(rel_path)
    roots = _possible_data_roots(images_root)

    rel_raw = Path(str(rel_path)).expanduser()
    if rel_raw.is_absolute():
        candidates.append(rel_raw)

    for root in roots:
        for rel_str in stripped_variants:
            rel = Path(rel_str)
            candidates.append(root / rel)
            candidates.append(root / rel.parent / "detections" / rel.name)

            parts = rel.parts
            if len(parts) >= 3:
                library, document = parts[0], parts[1]
                filename = parts[-1]
                doc_dir = root / library / document
                candidates.append(doc_dir / filename)
                candidates.append(doc_dir / "detections" / filename)

    return _unique_paths(candidates)


def find_image(images_root: Path, rel_path: str) -> Path:
    if not rel_path:
        raise FileNotFoundError("Empty image path in dataset row.")

    candidates = _candidate_paths(images_root, rel_path)

    for base in candidates:
        if base.exists() and base.is_file():
            return base
        stem = base.with_suffix("")
        for ext in IMG_EXTS:
            cand = stem.with_suffix(ext)
            if cand.exists() and cand.is_file():
                return cand

    for root in _possible_data_roots(images_root):
        for rel_str in _strip_known_prefixes(rel_path):
            parts = Path(rel_str).parts
            if len(parts) < 3:
                continue
            library, document = parts[0], parts[1]
            filename = parts[-1]
            stem = Path(filename).stem
            doc_dir = root / library / document
            if not doc_dir.exists():
                continue
            for ext in IMG_EXTS:
                hits = list(doc_dir.rglob(f"{stem}{ext}"))
                if hits:
                    return hits[0]
            for h in doc_dir.rglob(f"{stem}.*"):
                if h.is_file() and h.suffix.lower() in IMG_EXTS:
                    return h

    example_candidates = "\n".join(f"  - {p}" for p in candidates[:12])
    raise FileNotFoundError(
        f"Image not found for rel path: {rel_path}\n"
        f"images_root={Path(images_root).expanduser().resolve()}\n"
        f"Tried first candidates:\n{example_candidates}\n"
        "Use --images-root ../.. from src/train, or --images-root ../../datasets/digiknihovna_data."
    )


def discover_lora_targets(model, mode: str = "text", explicit: str = "") -> list[str]:
    if explicit.strip():
        return [x.strip() for x in explicit.split(",") if x.strip()]

    text_preferred = {"wq", "wk", "wv", "wo", "w1", "w2", "w3", "ff_proj", "ff_out"}
    vision_like = {"patch_embedding", "att_proj", "attn_out"}

    found_all = set()
    found_text = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            short = name.split(".")[-1]
            if short == "lm_head":
                continue
            found_all.add(short)
            if short in text_preferred:
                found_text.add(short)

    if mode == "text":
        if found_text:
            return sorted(found_text)
        return sorted(x for x in found_all if x not in vision_like)

    return sorted(found_all)


class MolmoBatchCollator:
    def __init__(
        self,
        processor,
        images_root: str,
        resize_long_side: int = 768,
        max_crops: int = 4,
        sequence_length: int = 1024,
        label_vocab_size: int | None = None,
        debug_first_n: int = 0,
        debug_every_n: int = 0,
        debug_log_file: str = "",
    ):
        self.processor = processor
        self.images_root = Path(images_root).expanduser()
        self.resize_long_side = int(resize_long_side or 0)
        self.max_crops = max(1, int(max_crops))
        self.sequence_length = max(128, int(sequence_length))
        self.label_vocab_size = (
            int(label_vocab_size) if label_vocab_size is not None else None
        )
        self.debug_first_n = max(0, int(debug_first_n or 0))
        self.debug_every_n = max(0, int(debug_every_n or 0))
        self.debug_log_file = (
            Path(debug_log_file).expanduser() if debug_log_file else None
        )
        self._encoded_examples_seen = 0
        self._printed_label_mask_warning = False

        if self.debug_log_file is not None:
            self.debug_log_file.parent.mkdir(parents=True, exist_ok=True)
            self.debug_log_file.write_text("", encoding="utf-8")

        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def _to_tensor_dict(self, obj):
        out = {}
        for k, v in obj.items():
            if torch.is_tensor(v):
                out[k] = v
            else:
                out[k] = torch.tensor(v)
        return out

    def _resize_image(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if self.resize_long_side <= 0:
            return image
        w, h = image.size
        long_side = max(w, h)
        if long_side <= self.resize_long_side:
            return image
        scale = self.resize_long_side / float(long_side)
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _processor_kwargs(self):
        return {
            "images_kwargs": {
                "max_crops": self.max_crops,
            },
            "text_kwargs": {
                "sequence_length": self.sequence_length,
            },
        }

    def _tensor_shape_summary(self, tensor_dict):
        out = {}
        for k, v in tensor_dict.items():
            if torch.is_tensor(v):
                out[k] = list(v.shape)
        return out

    def _maybe_log_token_debug(
        self,
        row,
        prompt: str,
        answer: str,
        image_path: Path,
        image_size_before,
        image_size_after,
        prompt_only,
        full,
        labels,
    ):
        self._encoded_examples_seen += 1
        idx = self._encoded_examples_seen
        should_log = idx <= self.debug_first_n or (
            self.debug_every_n > 0 and idx % self.debug_every_n == 0
        )
        if not should_log:
            return

        input_ids = full.get("input_ids")
        attention_mask = full.get("attention_mask")
        n_input_tokens = int(input_ids.numel()) if torch.is_tensor(input_ids) else None
        n_prompt_tokens = (
            int(prompt_only["input_ids"].shape[-1])
            if "input_ids" in prompt_only
            else None
        )
        n_supervised_answer_tokens = int((labels != -100).sum().item())
        n_attended_tokens = (
            int(attention_mask.sum().item())
            if torch.is_tensor(attention_mask)
            else None
        )
        n_invalid_input_ids = None
        if torch.is_tensor(input_ids) and self.label_vocab_size is not None:
            n_invalid_input_ids = int(
                ((input_ids < 0) | (input_ids >= self.label_vocab_size)).sum().item()
            )

        record = {
            "encoded_example_index": idx,
            "image_path": str(image_path),
            "image_size_before_resize": list(image_size_before),
            "image_size_after_resize": list(image_size_after),
            "prompt_preview": prompt[:160],
            "answer": answer,
            "sequence_length_arg": self.sequence_length,
            "max_crops_arg": self.max_crops,
            "input_tokens_total_tensor_length": n_input_tokens,
            "input_tokens_attended": n_attended_tokens,
            "prompt_tokens": n_prompt_tokens,
            "supervised_answer_tokens": n_supervised_answer_tokens,
            "non_text_or_image_token_ids_in_input": n_invalid_input_ids,
            "processor_tensor_shapes": self._tensor_shape_summary(full),
        }
        msg = "TOKEN_DEBUG " + json.dumps(record, ensure_ascii=False)
        print(msg)
        if self.debug_log_file is not None:
            with self.debug_log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _encode_one(self, row):
        rel_path = (
            row.get("image_rel_path")
            or row.get("image_rel_path_from_project_root")
            or row.get("image_rel_path_from_data_root")
        )
        image_path = find_image(self.images_root, rel_path)
        original_image = Image.open(image_path)
        image_size_before = original_image.size
        image = self._resize_image(original_image)
        image_size_after = image.size

        prompt = row["messages"][0]["content"].strip()
        answer = row["messages"][1]["content"].strip()
        eos = self.processor.tokenizer.eos_token or ""
        full_text = prompt + "\n" + answer + eos

        process_kwargs = self._processor_kwargs()
        prompt_only = self.processor.process(
            images=[image], text=prompt, **process_kwargs
        )
        full = self.processor.process(images=[image], text=full_text, **process_kwargs)

        prompt_only = self._to_tensor_dict(prompt_only)
        full = self._to_tensor_dict(full)

        input_ids = full["input_ids"].clone().long()
        labels = input_ids.clone()

        prompt_len = int(prompt_only["input_ids"].shape[-1])
        labels[:prompt_len] = -100

        if self.label_vocab_size is not None:
            invalid = (labels != -100) & (
                (labels < 0) | (labels >= self.label_vocab_size)
            )
            if bool(invalid.any()):
                if not self._printed_label_mask_warning:
                    bad_vals = labels[invalid][:10].detach().cpu().tolist()
                    print(
                        f"Masking invalid label ids outside vocab_size={self.label_vocab_size}; "
                        f"example invalid ids={bad_vals}"
                    )
                    self._printed_label_mask_warning = True
                labels[invalid] = -100

        if "attention_mask" in full:
            labels[full["attention_mask"].to(torch.bool) == 0] = -100

        full["labels"] = labels
        self._maybe_log_token_debug(
            row,
            prompt,
            answer,
            image_path,
            image_size_before,
            image_size_after,
            prompt_only,
            full,
            labels,
        )
        return full

    def _pad_1d(self, tensors, pad_value):
        max_len = max(int(t.shape[0]) for t in tensors)
        out = []
        for t in tensors:
            if t.shape[0] < max_len:
                pad = torch.full((max_len - t.shape[0],), pad_value, dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            out.append(t)
        return torch.stack(out, dim=0)

    def _batch_examples(self, examples):
        if len(examples) == 1:
            return {
                k: (v.unsqueeze(0) if torch.is_tensor(v) else v)
                for k, v in examples[0].items()
            }

        pad_token_id = self.processor.tokenizer.pad_token_id
        batched = {}
        keys = examples[0].keys()
        for k in keys:
            vals = [ex[k] for ex in examples]
            if not all(torch.is_tensor(v) for v in vals):
                batched[k] = vals
                continue

            if vals[0].ndim == 1 and k in {"input_ids", "attention_mask", "labels"}:
                if k == "input_ids":
                    pad_value = pad_token_id
                elif k == "attention_mask":
                    pad_value = 0
                else:
                    pad_value = -100
                batched[k] = self._pad_1d(vals, pad_value)
                continue

            shapes = [tuple(v.shape) for v in vals]
            if len(set(shapes)) == 1:
                batched[k] = torch.stack(vals, dim=0)
                continue

            raise ValueError(
                f"Cannot batch key={k!r} with variable shapes {shapes}. "
                "For this Molmo smoke run, use --per-device-train-batch-size 1 "
                "and --per-device-eval-batch-size 1."
            )
        return batched

    def __call__(self, features):
        examples = [self._encode_one(row) for row in features]
        return self._batch_examples(examples)


def parse_max_memory(spec: str):
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
        v = v.strip()
        out[int(k) if k.isdigit() else k] = v
    return out


def get_label_vocab_size(model) -> int | None:
    try:
        out_emb = model.get_output_embeddings()
        if hasattr(out_emb, "out_features") and out_emb.out_features is not None:
            return int(out_emb.out_features)
        if hasattr(out_emb, "weight") and out_emb.weight is not None:
            return int(out_emb.weight.shape[0])
    except Exception:
        pass
    try:
        return int(model.config.vocab_size)
    except Exception:
        return None


VISION_PARAM_KEYWORDS = (
    "vision_backbone",
    "vision_tower",
    "vision_model",
    "image_encoder",
    "image_vit",
    "visual",
    ".vit",
)


def is_vision_parameter_name(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in VISION_PARAM_KEYWORDS)


def freeze_vision_encoder_parameters(model) -> int:
    frozen = 0
    for name, param in model.named_parameters():
        if is_vision_parameter_name(name) and param.requires_grad:
            param.requires_grad_(False)
            frozen += param.numel()
    print(
        f"Freeze vision encoder: set requires_grad=False for {frozen:,} trainable vision parameters/adapters."
    )
    return frozen


def print_trainable_parameter_report(model, print_names: bool = False):
    total_params = 0
    trainable_params = 0
    groups = {}
    trainable_names = []

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        if not param.requires_grad:
            continue
        trainable_params += n
        if is_vision_parameter_name(name):
            group = "vision_encoder_or_vision_lora"
        elif "lora_" in name.lower():
            group = "text_lora_or_other_lora"
        else:
            group = "other_trainable"
        groups[group] = groups.get(group, 0) + n
        trainable_names.append((name, n, str(param.dtype), str(param.device)))

    pct = 100.0 * trainable_params / total_params if total_params else 0.0
    print("Trainable parameter report:")
    print(
        json.dumps(
            {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "trainable_percent": pct,
                "trainable_by_group": groups,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if print_names:
        print("Exact trainable parameter names:")
        for name, n, dtype, device in trainable_names:
            print(f"  {name} | {n:,} | {dtype} | {device}")


def parse_csv_list(spec: str) -> list[str]:
    return [x.strip() for x in (spec or "").split(",") if x.strip()]


def build_molmo_block_device_map(num_gpus: int, num_blocks: int = 28) -> dict | None:
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    train_ds = JsonlDataset(args.train_jsonl)
    val_ds = JsonlDataset(args.val_jsonl)
    print("Dataset sizes:", {"train": len(train_ds), "val": len(val_ds)})
    if len(train_ds) == 0:
        raise ValueError(
            "Train dataset is empty. First run repair_peoplegator_trial_dataset.py and check that "
            "Available pages with images is greater than 0."
        )

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    if args.disable_4bit:
        quantization_config = None
        torch_dtype = torch.bfloat16
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        torch_dtype = torch.bfloat16

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
        if device_map:
            print("Using device_map:", device_map)
            print("Using max_memory:", max_memory)

    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=device_map,
        max_memory=max_memory,
    )
    no_split_module_classes = parse_csv_list(args.no_split_module_classes)
    if device_map_arg and device_map_arg != "block" and no_split_module_classes:
        model_kwargs["no_split_module_classes"] = no_split_module_classes
        print("No-split module classes:", no_split_module_classes)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            **model_kwargs,
        )
    except TypeError as exc:
        if "no_split_module_classes" not in str(exc):
            raise
        print(
            "Warning: this Transformers version rejected no_split_module_classes; retrying without it."
        )
        model_kwargs.pop("no_split_module_classes", None)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            **model_kwargs,
        )

    if device_map:
        mark_model_parallel(model)
        hf_map = getattr(model, "hf_device_map", None)
        print("HF device map:", hf_map)
        if device_map_arg != "block" and isinstance(hf_map, dict):
            split_like = [
                k for k in hf_map if ".blocks." in str(k) and str(k).count(".") > 3
            ]
            if split_like:
                print(
                    "WARNING: HF automatic device_map appears to split inside transformer blocks."
                )
                print("         Prefer --device-map block for LoRA training on Molmo.")

    model.config.use_cache = False

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,
        )

    target_modules = discover_lora_targets(
        model,
        mode=args.lora_target_mode,
        explicit=args.lora_target_modules,
    )
    print("LoRA target mode:", args.lora_target_mode)
    print("LoRA target modules:", target_modules)
    print("Image resize long side:", args.resize_long_side)
    print("Molmo processor max_crops:", args.max_crops)
    print("Molmo processor sequence_length:", args.sequence_length)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    if device_map:
        model = align_lora_devices_with_base(model)
        mark_model_parallel(model)
    if args.freeze_vision_encoder:
        freeze_vision_encoder_parameters(model)
    model.print_trainable_parameters()
    print_trainable_parameter_report(model, print_names=args.print_trainable_names)

    label_vocab_size = get_label_vocab_size(model)
    print("Label vocab size:", label_vocab_size)

    collator = MolmoBatchCollator(
        processor=processor,
        images_root=args.images_root,
        resize_long_side=args.resize_long_side,
        max_crops=args.max_crops,
        sequence_length=args.sequence_length,
        label_vocab_size=label_vocab_size,
        debug_first_n=args.debug_token_first_n,
        debug_every_n=args.debug_token_log_steps,
        debug_log_file=args.debug_token_log_file,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=torch.cuda.is_available(),
        fp16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        optim="paged_adamw_8bit" if quantization_config is not None else "adamw_torch",
        load_best_model_at_end=False,
    )

    if device_map:
        training_args._n_gpu = 1

    print(
        "Batch config:",
        {
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "trainer_n_gpu": getattr(training_args, "_n_gpu", None),
            "train_batch_size": training_args.train_batch_size,
        },
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("Done. Saved adapter/checkpoint to:", args.output_dir)


if __name__ == "__main__":
    main()
