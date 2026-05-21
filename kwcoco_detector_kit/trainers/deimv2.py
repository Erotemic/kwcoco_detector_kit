"""
DEIMv2 trainer plugin — covers all 12 variants (8 HGNetv2 + 4 DINOv3).

This module is the load-bearing Python rewrite of the prior project's
``_train_deimv2_variant.sh``. The bash heredoc generator triggered
failure #13 (collate_fn indent leak) — moving the generator to Python
with a ``yaml.safe_dump`` round-trip eliminates the entire class of
indent bugs. The companion test suite in
``tests/unit/test_train_config_gen.py`` enforces:

  - collate_fn is a SIBLING of dataset under train_dataloader (#13)
  - HGNetv2 variants always emit base_size_repeat=None (#14)
  - The five sizes (eval_spatial_size, train Resize, val Resize,
    collate base_size, Mosaic output_size) move in lockstep with the
    requested input_hw (#18)
  - Optimizer block uses the right backbone regex per family
  - Per-variant memory-table batch lookup shrinks with input area
    and grows with VRAM (#16)

The trainer subprocess (DEIMv2 ``train.py``) is invoked by ``launch()``;
that path is only taken when the DEIMv2 submodule is installed under
``$KCD_DEIMV2_REPO_DPATH``. The config-gen path runs without DEIMv2.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from kwcoco_detector_kit.trainers._registry import register_trainer
from kwcoco_detector_kit._env import raise_nofile_limit


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------


_HGNETV2_SIZES = ["atto", "femto", "pico", "n", "s", "m", "l", "x"]
_DINOV3_SIZES = ["s", "m", "l", "x"]


# Upstream `configs/deimv2/deimv2_hgnetv2_*_coco.yml`:
#   atto/femto/pico   -> explicit `use_gateway: False`
#   n/s/m/l/x         -> no override (Python default `True`)
# The kit's generated YAML setting this EXPLICITLY for every variant
# guarantees the eval-time model architecture matches the training-time
# one (which in turn matches what the COCO-pretrained .pth was saved
# with). Without it, a single missed merge or YAMLConfig quirk can
# silently flip the value -- which is what bit us on v7 n@640.
_USE_GATEWAY_BY_SIZE: Dict[str, bool] = {
    "atto":  False,
    "femto": False,
    "pico":  False,
    "n":     True,
    "s":     True,
    "m":     True,
    "l":     True,
    "x":     True,
}


# Per-variant DEIMTransformer.num_queries. Mirrors upstream configs at
# tpl/DEIMv2/configs/deimv2/*. The smaller HGNetv2 variants override the
# 300 default with a smaller value (matches their decoder capacity); the
# rest inherit 300 from configs/base/deimv2.yml.
_NUM_QUERIES_BY_VARIANT: Dict[str, int] = {
    "deimv2_hgnetv2_atto":  100,
    "deimv2_hgnetv2_femto": 150,
    "deimv2_hgnetv2_pico":  200,
}
_DEFAULT_NUM_QUERIES = 300


def _build_variants() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for size in _HGNETV2_SIZES:
        name = f"deimv2_hgnetv2_{size}"
        out[name] = {
            "family": "hgnetv2",
            "size": size,
            "upstream_config_relpath": f"configs/deimv2/deimv2_hgnetv2_{size}_coco.yml",
            "supports_dynamic_input": False,
            "num_queries": _NUM_QUERIES_BY_VARIANT.get(name, _DEFAULT_NUM_QUERIES),
            "use_gateway": _USE_GATEWAY_BY_SIZE.get(size, True),
        }
    for size in _DINOV3_SIZES:
        name = f"deimv2_dinov3_{size}"
        out[name] = {
            "family": "dinov3",
            "size": size,
            "upstream_config_relpath": f"configs/deimv2/deimv2_dinov3_{size}_coco.yml",
            "supports_dynamic_input": True,
            "num_queries": _NUM_QUERIES_BY_VARIANT.get(name, _DEFAULT_NUM_QUERIES),
            "use_gateway": True,  # DINOv3 variants follow upstream default
        }
    return out


VARIANTS = _build_variants()


# Convenience aliases mirroring the prior project's shorthand:
#   deimv2_atto/femto/pico/n  -> HGNetv2 family
#   deimv2_s/m/l/x            -> DINOv3 family (defaulting to DINOv3 mirrors the
#                               prior bash, which treated `s/m/l/x` as DINOv3).
ALIASES: Dict[str, str] = {
    "deimv2_atto": "deimv2_hgnetv2_atto",
    "deimv2_femto": "deimv2_hgnetv2_femto",
    "deimv2_pico": "deimv2_hgnetv2_pico",
    "deimv2_n": "deimv2_hgnetv2_n",
    "deimv2_s": "deimv2_dinov3_s",
    "deimv2_m": "deimv2_dinov3_m",
    "deimv2_l": "deimv2_dinov3_l",
    "deimv2_x": "deimv2_dinov3_x",
}


def _resolve_variant(name: str) -> str:
    if name in VARIANTS:
        return name
    if name in ALIASES:
        return ALIASES[name]
    raise KeyError(
        f"unknown DEIMv2 variant {name!r}; "
        f"known: {sorted(list(VARIANTS) + list(ALIASES))}"
    )


# ---------------------------------------------------------------------------
# Per-variant memory table (failure #16)
#
# Anchor point measured on the prior project: deimv2_hgnetv2_n @ 320x320
# batch=32 -> 8.1 GB training footprint on a 24 GB consumer GPU. Cost
# scales ~linearly with batch and ~quadratically with input area.
#
# The table below is a 2D lookup: per family-size, the (batch, vram_floor)
# break points at input_long=320. Larger inputs auto-shrink by area ratio.
# ---------------------------------------------------------------------------


@dataclass
class _BatchRow:
    """Per-variant default batch as a function of (input_long, vram_gb)."""
    base_input_long: int          # reference long side
    base_batch_24gb: int          # batch at base_input_long on a 24 GB GPU
    decay_above_pixel_area: float  # multiplier per 2x area increase


_BATCH_TABLE: Dict[str, _BatchRow] = {
    # HGNetv2 family
    "deimv2_hgnetv2_atto":  _BatchRow(320, 64, 0.5),
    "deimv2_hgnetv2_femto": _BatchRow(320, 48, 0.5),
    "deimv2_hgnetv2_pico":  _BatchRow(320, 32, 0.5),
    "deimv2_hgnetv2_n":     _BatchRow(320, 32, 0.5),
    "deimv2_hgnetv2_s":     _BatchRow(320, 24, 0.5),
    "deimv2_hgnetv2_m":     _BatchRow(320, 16, 0.5),
    "deimv2_hgnetv2_l":     _BatchRow(320, 12, 0.5),
    "deimv2_hgnetv2_x":     _BatchRow(320, 8,  0.5),
    # DINOv3 family — heavier per sample, anchor at 640
    "deimv2_dinov3_s":      _BatchRow(640, 8,  0.5),
    "deimv2_dinov3_m":      _BatchRow(640, 4,  0.5),
    "deimv2_dinov3_l":      _BatchRow(640, 2,  0.5),
    "deimv2_dinov3_x":      _BatchRow(640, 1,  0.5),
}


def _batch_for(variant: str, input_hw: Tuple[int, int], total_vram_gb: float) -> int:
    """Look up the recommended per-GPU batch.

    Scaling rules:
      - input area is the dominant axis. Doubling input area -> halve batch.
      - VRAM scales linearly relative to 24 GB.

    Always returns >= 1.
    """
    variant = _resolve_variant(variant)
    row = _BATCH_TABLE[variant]
    input_long = max(int(input_hw[0]), int(input_hw[1]))
    base_long = int(row.base_input_long)
    area_ratio = (input_long ** 2) / float(base_long ** 2)
    # area_ratio of 1 -> no decay; 2 -> halve; 4 -> quarter; etc.
    area_factor = row.decay_above_pixel_area ** max(0.0, (area_ratio - 1.0))
    vram_factor = float(total_vram_gb) / 24.0
    batch = int(round(row.base_batch_24gb * area_factor * vram_factor))
    return max(1, batch)


# ---------------------------------------------------------------------------
# Train-resolution-policy parser (mirrors the prior bash logic)
# ---------------------------------------------------------------------------


@dataclass
class _PolicyResolution:
    base_size: int              # collate base_size
    base_size_repeat: Optional[int]  # None for fixed-scale
    stop_epoch: int
    requested_min: int
    requested_max: int
    effective_scales: List[int]
    effective_min: int
    effective_max: int


def _generate_scales(base_size: int, base_size_repeat: Optional[int]) -> List[int]:
    """Mirror of ``tpl/DEIMv2/engine/data/dataloader.py:generate_scales``."""
    if not base_size_repeat:
        return [int(base_size)]
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


def _resolve_policy(
    policy: str,
    input_hw: Tuple[int, int],
    num_epochs: int,
    *,
    supports_dynamic: bool,
    multiscale_repeat: int = 12,
) -> _PolicyResolution:
    """Parse the policy string + variant's dynamic-input support into a resolution.

    Failure #14: HGNetv2 architectural constraint — when the variant
    does NOT support dynamic input, force base_size_repeat=None even
    when the policy string requested multiscale. The orchestration
    layer (``round_loop``) catches this earlier and emits a warning;
    the trainer plugin is the second line of defense.
    """
    H, W = int(input_hw[0]), int(input_hw[1])
    input_long = max(H, W)
    stop_epoch_default = max(1, int(num_epochs) - 4)

    if policy == "fixed":
        base = input_long
        repeat = None
        req_min = input_long
        req_max = input_long
    elif policy == "multiscale":
        base = input_long
        repeat = int(multiscale_repeat) if supports_dynamic else None
        req_min = (base * 75) // 100
        req_max = (base * 125) // 100
    elif policy.startswith("multiscale_"):
        rest = policy[len("multiscale_"):]
        if "_" in rest:
            lo_str, hi_str = rest.split("_", 1)
            try:
                lo = int(lo_str)
                hi = int(hi_str)
            except ValueError as ex:
                raise ValueError(f"bad multiscale policy {policy!r}: {ex}")
            mid = (lo + hi) // 2
            base = ((mid + 16) // 32) * 32
            repeat = int(multiscale_repeat) if supports_dynamic else None
            req_min = lo
            req_max = hi
        else:
            try:
                base = int(rest)
            except ValueError as ex:
                raise ValueError(f"bad multiscale policy {policy!r}: {ex}")
            repeat = int(multiscale_repeat) if supports_dynamic else None
            req_min = (base * 75) // 100
            req_max = (base * 125) // 100
    else:
        raise ValueError(
            f"unsupported train_policy {policy!r}; "
            "expected: fixed | multiscale | multiscale_<S> | multiscale_<lo>_<hi>"
        )

    scales = sorted(set(_generate_scales(base, repeat)))
    return _PolicyResolution(
        base_size=int(base),
        base_size_repeat=repeat,
        stop_epoch=int(stop_epoch_default if repeat else 1),
        requested_min=int(req_min),
        requested_max=int(req_max),
        effective_scales=scales,
        effective_min=int(scales[0]),
        effective_max=int(scales[-1]),
    )


# ---------------------------------------------------------------------------
# YAML config generator
# ---------------------------------------------------------------------------


def _hgnetv2_optimizer_block(lr: float, backbone_lr: float) -> Dict[str, Any]:
    return {
        "type": "AdamW",
        "params": [
            {
                "params": r"^(?=.*backbone)(?!.*norm|bn).*$",
                "lr": float(backbone_lr),
            },
            {
                "params": r"^(?=.*backbone)(?=.*norm|bn).*$",
                "lr": float(backbone_lr),
                "weight_decay": 0.0,
            },
            {
                "params": r"^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$",
                "weight_decay": 0.0,
            },
        ],
        "lr": float(lr),
        "betas": [0.9, 0.999],
        "weight_decay": 0.0001,
    }


def _dinov3_optimizer_block(lr: float, backbone_lr: float) -> Dict[str, Any]:
    return {
        "type": "AdamW",
        "params": [
            {
                "params": r"^(?=.*.dinov3)(?!.*(?:norm|bn|bias)).*$",
                "lr": float(backbone_lr),
            },
            {
                "params": r"^(?=.*.dinov3)(?=.*(?:norm|bn|bias)).*$",
                "lr": float(backbone_lr),
                "weight_decay": 0.0,
            },
            {
                "params": r"^(?=.*(?:sta|encoder|decoder))(?=.*(?:norm|bn|bias)).*$",
                "weight_decay": 0.0,
            },
        ],
        "lr": float(lr),
        "betas": [0.9, 0.999],
        "weight_decay": 0.0001,
    }


def _train_transforms_block(input_hw: Tuple[int, int]) -> Dict[str, Any]:
    H, W = int(input_hw[0]), int(input_hw[1])
    mosaic_out = H // 2
    return {
        "type": "Compose",
        "ops": [
            {
                "type": "Mosaic",
                "output_size": int(mosaic_out),
                "rotation_range": 10,
                "translation_range": [0.1, 0.1],
                "scaling_range": [0.5, 1.5],
                "probability": 1.0,
                "fill_value": 0,
                "use_cache": True,
                "max_cached_images": 50,
                "random_pop": True,
            },
            {"type": "RandomPhotometricDistort", "p": 0.5},
            {"type": "RandomZoomOut", "fill": 0},
            {"type": "RandomIoUCrop", "p": 0.8},
            {"type": "SanitizeBoundingBoxes", "min_size": 1},
            {"type": "RandomHorizontalFlip"},
            {"type": "Resize", "size": [H, W]},
            {"type": "SanitizeBoundingBoxes", "min_size": 1},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
            {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
        ],
        "policy": {
            "name": "stop_epoch",
            "epoch": [4, 78, 148],
            "ops": ["Mosaic", "RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop"],
        },
        "mosaic_prob": 0.5,
    }


def _val_transforms_block(input_hw: Tuple[int, int]) -> Dict[str, Any]:
    H, W = int(input_hw[0]), int(input_hw[1])
    return {
        "type": "Compose",
        "ops": [
            {"type": "Resize", "size": [H, W]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        ],
    }


def _effective_num_top_queries(num_queries: int, num_classes: int,
                                default_topk: int = 300) -> int:
    """Compute a safe ``PostProcessor.num_top_queries``.

    DEIMv2's PostProcessor selects topk over ``scores.flatten(1)`` whose
    shape is ``[batch, num_queries * num_classes]`` (one logit per
    (query, class) pair). The upstream default ``num_top_queries=300``
    assumes COCO's ``num_classes=91`` so 100*91=9100 >> 300; with the
    kit's ``num_classes=1`` override the flattened axis collapses to
    ``num_queries`` and topk(k=300) raises ``selected index k out of range``.

    Returns ``min(default_topk, num_queries * num_classes)`` (always
    ≥ 1). See lesson #26.
    """
    upper = max(1, int(num_queries) * int(num_classes))
    return min(int(default_topk), upper)


def _build_train_yml(
    *,
    workdir: Path,
    upstream_cfg_fpath: str,
    train_mscoco_fpath: str,
    vali_mscoco_fpath: str,
    family: str,
    num_queries: int,
    use_gateway: bool,
    input_hw: Tuple[int, int],
    num_classes: int,
    batch_size: int,
    val_batch_size: int,
    num_epochs: int,
    lr: float,
    backbone_lr: float,
    use_amp: bool,
    policy: _PolicyResolution,
) -> Dict[str, Any]:
    H, W = int(input_hw[0]), int(input_hw[1])
    if family == "hgnetv2":
        optimizer = _hgnetv2_optimizer_block(lr, backbone_lr)
    elif family == "dinov3":
        optimizer = _dinov3_optimizer_block(lr, backbone_lr)
    else:
        raise ValueError(f"unknown variant family {family!r}")

    return {
        "__include__": [str(upstream_cfg_fpath)],
        "output_dir": str(workdir),
        "summary_dir": str(workdir / "summary"),
        "use_amp": bool(use_amp),
        "task": "detection",
        "num_classes": int(num_classes),
        "remap_mscoco_category": False,
        "evaluator": {"type": "CocoEvaluator", "iou_types": ["bbox"]},
        "eval_spatial_size": [H, W],
        # Failure #18 lockstep — and the postprocessor's topk has to be
        # consistent with num_queries * num_classes (lesson #26).
        "PostProcessor": {
            "num_top_queries": _effective_num_top_queries(num_queries, num_classes),
        },
        # Explicit use_gateway per variant. The pico/atto/femto upstream
        # configs disable it; the larger HGNetv2 variants and DINOv3
        # variants leave it at the Python default (True). Without this
        # explicit setting, eval-time YAMLConfig has been observed to
        # pick a different value than train-time, breaking the eval-time
        # state_dict load (v7 n@640 episode).
        "DEIMTransformer": {
            "use_gateway": bool(use_gateway),
        },
        "train_dataloader": {
            "total_batch_size": int(batch_size),
            "num_workers": 4,
            "dataset": {
                "img_folder": "/",
                "ann_file": str(train_mscoco_fpath),
                "return_masks": False,
                "transforms": _train_transforms_block(input_hw),
            },
            "collate_fn": {
                "type": "BatchImageCollateFunction",
                "base_size": int(policy.base_size),
                "base_size_repeat": policy.base_size_repeat,
                "stop_epoch": int(policy.stop_epoch),
            },
        },
        "val_dataloader": {
            "total_batch_size": int(val_batch_size),
            "num_workers": 2,
            "dataset": {
                "img_folder": "/",
                "ann_file": str(vali_mscoco_fpath),
                "return_masks": False,
                "transforms": _val_transforms_block(input_hw),
            },
        },
        "epoches": int(num_epochs),
        "optimizer": optimizer,
    }


def _ensure_mscoco(input_fpath, dst_fpath: Path, *, category_names) -> Path:
    """If `input_fpath` is already a .mscoco.json, return it unchanged.
    Otherwise (kwcoco bundle), export to MSCOCO at `dst_fpath` and return that.

    DEIMv2's train.py consumes MSCOCO; the kit's pipeline gives us kwcoco.
    This helper isolates the conversion at the trainer-plugin boundary.
    The MSCOCO ``category_id`` assigned to each name is ``i`` for the i-th
    name in ``category_names``, matching DEIMv2's 0-indexed class labels.
    """
    from kwcoco_detector_kit.data.coco_export import export_mscoco

    src = str(input_fpath)
    dst_fpath = Path(dst_fpath)
    if src.endswith(".mscoco.json") or src.endswith(".coco.json"):
        return Path(src)
    dst_fpath.parent.mkdir(parents=True, exist_ok=True)
    export_mscoco(
        src, dst_fpath, category_names=list(category_names),
        include_segmentations=False, category_id_start=0,
    )
    return dst_fpath


def _kit_root_tpl_path(submodule_name: str) -> Optional[Path]:
    """Return ``<kit_root>/tpl/<submodule_name>`` if that submodule is on disk.

    Used as the fallback when ``$KCD_DEIMV2_REPO_DPATH`` is unset. `<kit_root>`
    is `Path(kwcoco_detector_kit.__file__).parent.parent` — the kit's working
    copy when installed editable; in a site-packages install this typically
    won't have ``tpl/`` so we return None.
    """
    import kwcoco_detector_kit
    pkg_init = Path(kwcoco_detector_kit.__file__).resolve()
    candidate = pkg_init.parent.parent / "tpl" / submodule_name
    if candidate.is_dir() and any(candidate.iterdir()):
        return candidate
    return None


def _resolve_deimv2_repo() -> Optional[Path]:
    """The DEIMv2 checkout, in priority order:

    1. ``$KCD_DEIMV2_REPO_DPATH`` (explicit override)
    2. ``<kit_root>/tpl/DEIMv2`` (the kit's own submodule, if initialised)
    3. None (config-gen still works; launch() raises with a clear hint)
    """
    repo = os.environ.get("KCD_DEIMV2_REPO_DPATH")
    if repo:
        return Path(repo).expanduser().resolve()
    return _kit_root_tpl_path("DEIMv2")


def _resolve_upstream_cfg_fpath(variant_name: str) -> str:
    """Compose the absolute path to the upstream DEIMv2 config.

    Reads ``$KCD_DEIMV2_REPO_DPATH`` first, then falls back to
    ``<kit_root>/tpl/DEIMv2``. Returns a relative path when neither is
    found — the YAML will still parse, but the upstream consumer must be
    launched with cwd inside the DEIMv2 repo.
    """
    info = VARIANTS[_resolve_variant(variant_name)]
    rel = info["upstream_config_relpath"]
    repo = _resolve_deimv2_repo()
    if repo:
        return str(repo / rel)
    return str(rel)


def _dump_policy_json(workdir: Path, *, candidate_id: str, variant: str,
                     input_hw, policy_name: str, policy: _PolicyResolution,
                     batch: int, val_batch: int, num_epochs: int,
                     lr: float, backbone_lr: float, use_amp: bool,
                     init_ckpt: str, generated_cfg_fpath: Path):
    H, W = int(input_hw[0]), int(input_hw[1])
    obj = {
        "candidate_id": candidate_id,
        "variant": _resolve_variant(variant),
        "candidate_kind": "real",
        "run_tag": os.environ.get("KCD_RUN_TAG", ""),
        "export_input_h": H,
        "export_input_w": W,
        "train_resolution_policy": policy_name,
        "requested_train_resolution_min": int(policy.requested_min),
        "requested_train_resolution_max": int(policy.requested_max),
        "multiscale_base_size": int(policy.base_size),
        "multiscale_repeat": int(policy.base_size_repeat or 0),
        "multiscale_stop_epoch": int(policy.stop_epoch),
        "train_batch": int(batch),
        "val_batch": int(val_batch),
        "num_epochs": int(num_epochs),
        "lr": float(lr),
        "backbone_lr": float(backbone_lr),
        "use_amp": bool(use_amp),
        "init_ckpt": str(init_ckpt or ""),
        "generated_train_cfg": str(generated_cfg_fpath),
        "effective_train_scales": list(policy.effective_scales),
        "effective_train_scale_min": int(policy.effective_min),
        "effective_train_scale_max": int(policy.effective_max),
    }
    (workdir / "policy.json").write_text(json.dumps(obj, indent=2))


# ---------------------------------------------------------------------------
# Predictor adapter
# ---------------------------------------------------------------------------


class DEIMv2Predictor:
    """Inference adapter for a trained DEIMv2 checkpoint.

    Requires the DEIMv2 submodule to be importable (``engine.core.YAMLConfig``).
    Phase 1 keeps this minimal — the kit's smoke tests use mock_tiny instead.
    """

    def __init__(self, ckpt_fpath, config_fpath, device: str = "cpu"):
        import torch
        from torch import nn

        repo = _resolve_deimv2_repo()
        if repo and str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        try:
            YAMLConfig = __import__("engine.core", fromlist=["YAMLConfig"]).YAMLConfig
        except Exception as ex:
            raise ImportError(
                "DEIMv2Predictor needs a DEIMv2 checkout. Either set "
                "$KCD_DEIMV2_REPO_DPATH or run "
                "`git submodule update --init tpl/DEIMv2` from the kit's repo root."
            ) from ex

        ckpt_fpath = str(ckpt_fpath)
        config_fpath = str(config_fpath)

        state = torch.load(ckpt_fpath, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            if "ema" in state and "module" in state["ema"]:
                state = state["ema"]["module"]
            elif "model" in state:
                state = state["model"]

        cfg = YAMLConfig(config_fpath, resume=ckpt_fpath)
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        # Detect architecture knobs from the saved state_dict and force-set
        # the YAML before building the model. Without this, eval-time can
        # pick a different DEIMTransformer default than train-time and the
        # subsequent load_state_dict fails on missing/unexpected keys.
        # Currently covers: use_gateway. The n/s/m/l/x variants train with
        # gateway=True (Python default + COCO pretrained .pth has gateway
        # keys); the pico/atto/femto variants explicitly set False in their
        # upstream configs.
        any_gateway_key = any("gateway." in k for k in state.keys())
        cfg.yaml_cfg.setdefault("DEIMTransformer", {})
        existing_gw = cfg.yaml_cfg["DEIMTransformer"].get("use_gateway")
        if existing_gw is None or bool(existing_gw) != bool(any_gateway_key):
            cfg.yaml_cfg["DEIMTransformer"]["use_gateway"] = bool(any_gateway_key)
            print(
                f"[DEIMv2Predictor] forcing DEIMTransformer.use_gateway="
                f"{bool(any_gateway_key)} (inferred from checkpoint state_dict)",
                flush=True,
            )

        cfg.model.load_state_dict(state)

        class _Wrapper(nn.Module):
            def __init__(self, cfg_):
                super().__init__()
                self.model = cfg_.model.deploy()
                self.post = cfg_.postprocessor.deploy()

            def forward(self, im, sz):
                return self.post(self.model(im), sz)

        self._model = _Wrapper(cfg).to(device).eval()
        eval_h, eval_w = cfg.yaml_cfg["eval_spatial_size"]
        self._eval_h = int(eval_h)
        self._eval_w = int(eval_w)
        self._device = device

    @property
    def eval_spatial_size(self) -> Tuple[int, int]:
        return (self._eval_h, self._eval_w)

    def predict_image(self, image_np, orig_size):
        import kwimage
        import numpy as np
        import torch
        if image_np.ndim == 2:
            image_np = np.repeat(image_np[..., None], 3, axis=-1)
        if image_np.shape[2] == 4:
            image_np = image_np[..., :3]
        try:
            resized = kwimage.imresize(image_np, dsize=(self._eval_w, self._eval_h),
                                       interpolation="area")
        except NotImplementedError:
            resized = kwimage.imresize(image_np, dsize=(self._eval_w, self._eval_h),
                                       interpolation="linear")
        chw = torch.from_numpy(
            (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        ).to(self._device)
        W, H = int(orig_size[0]), int(orig_size[1])
        sz = torch.tensor([[W, H]], dtype=torch.int64, device=self._device)
        with torch.no_grad():
            labels, boxes, scores = self._model(chw, sz)
        out: List[dict] = []
        b = boxes[0].cpu().numpy()
        s = scores[0].cpu().numpy()
        l = labels[0].cpu().numpy()
        for k in range(b.shape[0]):
            score = float(s[k])
            x1, y1, x2, y2 = [float(v) for v in b[k]]
            out.append({
                "label": int(l[k]),
                "bbox_xyxy": [x1, y1, x2, y2],
                "score": score,
            })
        return out


# ---------------------------------------------------------------------------
# Trainer plugin
# ---------------------------------------------------------------------------


@register_trainer
class DEIMv2Trainer:
    """Driver for upstream DEIMv2 train.py across all 12 variants."""

    name = "deimv2"
    variants = VARIANTS
    supports_onnx_export = True

    # ---- Protocol methods ----

    def generate_config(
        self,
        train_kwcoco_fpath,
        vali_kwcoco_fpath,
        workdir,
        *,
        variant: str,
        input_hw: Tuple[int, int],
        train_policy: str = "fixed",
        num_classes: int = 1,
        batch_size: int = 16,
        val_batch_size: int = 32,
        num_epochs: int = 60,
        lr: float = 5e-4,
        backbone_lr: float = 2.5e-5,
        use_amp: bool = True,
        init_checkpoint: Optional[str] = None,
        channels: str = "r|g|b",
        scale_tier: str = "M",
        num_gpus: int = 1,
        data_format: str = "kwcoco",
        extra: Optional[dict] = None,
    ) -> Path:
        canonical = _resolve_variant(variant)
        info = VARIANTS[canonical]
        family = info["family"]
        supports_dynamic = bool(info["supports_dynamic_input"])

        workdir = Path(workdir)
        gen_dpath = workdir / "generated_configs"
        gen_dpath.mkdir(parents=True, exist_ok=True)
        cfg_fpath = gen_dpath / "train.yml"

        # DEIMv2's train.py reads MSCOCO json (not kwcoco). When the caller
        # hands us a .kwcoco path, convert it to MSCOCO inside the workdir.
        # Already-MSCOCO inputs (.mscoco.json) pass through unchanged.
        category_names = list((extra or {}).get("category_names") or ["widget"])
        if len(category_names) != int(num_classes):
            raise ValueError(
                f"num_classes={num_classes} disagrees with "
                f"len(category_names)={len(category_names)} (names={category_names!r}); "
                "DEIMv2 maps the i-th category name to class index i."
            )
        train_ann_fpath = _ensure_mscoco(
            train_kwcoco_fpath, workdir / "detector_prepared" / "train.mscoco.json",
            category_names=category_names,
        )
        vali_ann_fpath = _ensure_mscoco(
            vali_kwcoco_fpath, workdir / "detector_prepared" / "vali.mscoco.json",
            category_names=category_names,
        )

        policy = _resolve_policy(
            train_policy, input_hw, num_epochs,
            supports_dynamic=supports_dynamic,
        )
        upstream_cfg = _resolve_upstream_cfg_fpath(canonical)

        yml = _build_train_yml(
            workdir=workdir,
            upstream_cfg_fpath=upstream_cfg,
            train_mscoco_fpath=str(train_ann_fpath),
            vali_mscoco_fpath=str(vali_ann_fpath),
            family=family,
            num_queries=int(info["num_queries"]),
            use_gateway=bool(info["use_gateway"]),
            input_hw=tuple(input_hw),
            num_classes=int(num_classes),
            batch_size=int(batch_size),
            val_batch_size=int(val_batch_size),
            num_epochs=int(num_epochs),
            lr=float(lr),
            backbone_lr=float(backbone_lr),
            use_amp=bool(use_amp),
            policy=policy,
        )

        cfg_fpath.write_text(yaml.safe_dump(yml, sort_keys=False))
        # Resolved-effective-config side-by-side sidecar — Phase 1 emits a
        # copy of the generator's view; the upstream __include__ expansion
        # happens at launch time when the DEIMv2 repo is on PYTHONPATH.
        (gen_dpath / "resolved_effective_config.yml").write_text(
            yaml.safe_dump(yml, sort_keys=False)
        )

        # policy.json — used by orchestration/eligibility.py to read
        # everything back without parsing the YAML.
        candidate_id = (extra or {}).get(
            "candidate_id",
            f"{canonical}_{int(input_hw[0])}x{int(input_hw[1])}",
        )
        # Prefer the explicit init_checkpoint kwarg; fall back to extra
        # for backwards-compat with callers that still pass it that way.
        _effective_init_ckpt = (
            init_checkpoint
            if init_checkpoint is not None
            else (extra or {}).get("init_checkpoint", "")
        )
        _dump_policy_json(
            workdir,
            candidate_id=str(candidate_id),
            variant=canonical,
            input_hw=input_hw,
            policy_name=str(train_policy),
            policy=policy,
            batch=int(batch_size),
            val_batch=int(val_batch_size),
            num_epochs=int(num_epochs),
            lr=float(lr),
            backbone_lr=float(backbone_lr),
            use_amp=bool(use_amp),
            init_ckpt=str(_effective_init_ckpt or ""),
            generated_cfg_fpath=cfg_fpath,
        )

        return cfg_fpath

    def launch(
        self,
        config_fpath,
        *,
        init_checkpoint=None,
        resume=None,
        num_gpus: int = 1,
        distributed: bool = False,
    ) -> Path:
        """Subprocess DEIMv2's train.py against the generated config.

        Uses ``_resolve_deimv2_repo()`` — ``$KCD_DEIMV2_REPO_DPATH`` then
        ``<kit_root>/tpl/DEIMv2``. Returns the workdir.
        """
        cfg_fpath = Path(config_fpath)
        workdir = cfg_fpath.parent.parent

        repo = _resolve_deimv2_repo()
        if not repo:
            raise EnvironmentError(
                "DEIMv2Trainer.launch needs a DEIMv2 checkout. Either set "
                "$KCD_DEIMV2_REPO_DPATH or run "
                "`git submodule update --init tpl/DEIMv2` from the kit's repo root."
            )
        train_py = repo / "train.py"
        if not train_py.exists():
            raise FileNotFoundError(train_py)

        # Failure #15: raise RLIMIT_NOFILE before launching torch
        # multiprocessing IPC. Clamp to hard cap.
        raise_nofile_limit(target=65536)

        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

        # Always launch DEIMv2 under torch.distributed.run, even single-GPU.
        # Why: DEIMv2's setup_distributed() calls init_process_group(env://)
        # which silently fails when RANK / LOCAL_RANK / WORLD_SIZE are unset.
        # That used to be OK because torch <= 2.9 let get_rank() return 0 when
        # no process group was initialized — but several DEIMv2 modules
        # (engine/backbone/hgnetv2.py:562 most prominently) call get_rank()
        # unconditionally. On torch 2.10+ that raises ValueError("Default
        # process group has not been initialized") and the whole subprocess
        # dies before training starts. Running every DEIMv2 launch under
        # torchrun sets RANK=0 LOCAL_RANK=0 WORLD_SIZE=N so the env-based
        # init succeeds on both old and new torch. See lessons #24.
        args = [
            sys.executable, "-m", "torch.distributed.run",
            "--master_port", str(env.get("KCD_MASTER_PORT", "29500")),
            "--nproc_per_node", str(int(num_gpus)),
            str(train_py), "-c", str(cfg_fpath),
        ]
        if init_checkpoint:
            args += ["-t", str(init_checkpoint)]
            print(
                f"[deimv2.launch] fine-tuning from init_checkpoint="
                f"{init_checkpoint}",
                flush=True,
            )
        else:
            print(
                "[deimv2.launch] no init_checkpoint -- training from scratch "
                "(HGNetv2 stem only). For shitspotter-scale data this loses "
                "~5-10 AP vs. fine-tuning from deimv2_<variant>_coco.pth.",
                flush=True,
            )
        if resume:
            args += ["-r", str(resume)]

        subprocess.run(args, check=True, env=env, cwd=str(repo))
        return workdir

    def find_checkpoint(self, workdir) -> Path:
        workdir = Path(workdir)
        for cand in ("best_stg2.pth", "best_stg1.pth", "last.pth"):
            p = workdir / cand
            if p.exists():
                return p
        epochs = sorted(workdir.glob("checkpoint*.pth"))
        if epochs:
            return epochs[-1]
        raise FileNotFoundError(f"no DEIMv2 checkpoint in {workdir}")

    def supports_dynamic_input(self, variant: str) -> bool:
        canonical = _resolve_variant(variant)
        return bool(VARIANTS[canonical]["supports_dynamic_input"])

    def memory_tier_default_batch(
        self,
        variant: str,
        input_hw: Tuple[int, int],
        total_vram_gb: float,
    ) -> int:
        return _batch_for(variant, tuple(input_hw), float(total_vram_gb))

    def supports_webdataset_input(self) -> bool:
        return False  # Phase 1 — kwcoco only

    def build_predictor(self, workdir, *, device: str = "cpu"):
        workdir = Path(workdir)
        ckpt = self.find_checkpoint(workdir)
        cfg = workdir / "generated_configs" / "train.yml"
        return DEIMv2Predictor(ckpt, cfg, device=device)
