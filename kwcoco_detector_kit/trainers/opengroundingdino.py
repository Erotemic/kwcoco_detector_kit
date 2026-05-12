"""
OpenGroundingDINO trainer plugin — DINOv2 + BERT + DETR.

Ported from the v9 shell pipeline. The trainer:

1. Exports the source kwcoco to MSCOCO json (via ``data.coco_export``).
2. Subprocesses OpenGroundingDINO's ``tools/coco2odvg.py`` to convert
   the train MSCOCO to ODVG (Object Detection Vision Grounding) format.
3. Writes the OpenGroundingDINO config (copy of upstream's ``cfg_odvg.py``
   + a small Python-overrides block — analogous to the DEIMv2 generator's
   override layering).
4. Writes ``datasets.json`` listing train ODVG + val MSCOCO.
5. Subprocesses ``train_dist.sh`` from the OpenGroundingDINO repo.

Variants registered:

  ``opengroundingdino_swint``   Swin-Tiny backbone; tier L sweet spot.
  ``opengroundingdino_swinb``   Swin-Base backbone; tier XL.

Both DINOv2-prompted, both DETR decoders — both support dynamic input
via per-batch positional-embedding interpolation (the architectural
mirror of DEIMv2 DINOv3 variants).
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

from kwcoco_detector_kit.trainers._registry import register_trainer
from kwcoco_detector_kit._env import raise_nofile_limit


VARIANTS: Dict[str, Dict[str, Any]] = {
    "opengroundingdino_swint": {
        "family": "swin",
        "size": "tiny",
        "upstream_config_relpath": "config/cfg_odvg.py",
        "supports_dynamic_input": True,
        "default_tier": "L",
    },
    "opengroundingdino_swinb": {
        "family": "swin",
        "size": "base",
        "upstream_config_relpath": "config/cfg_odvg.py",
        "supports_dynamic_input": True,
        "default_tier": "XL",
    },
}


@dataclass
class _BatchRow:
    base_vram_gb: float       # reference VRAM
    base_batch_per_gpu: int   # batch at base_vram, 1 GPU
    decay_above_pixel_area: float


_BATCH_TABLE: Dict[str, _BatchRow] = {
    "opengroundingdino_swint": _BatchRow(48.0, 4, 0.5),  # Swin-Tiny @ 800 input
    "opengroundingdino_swinb": _BatchRow(80.0, 4, 0.5),  # Swin-Base @ 1024 input
}


def _batch_for(variant: str, input_hw: Tuple[int, int], total_vram_gb: float) -> int:
    row = _BATCH_TABLE[variant]
    input_long = max(int(input_hw[0]), int(input_hw[1]))
    base_long = 800
    area_ratio = (input_long ** 2) / float(base_long ** 2)
    area_factor = row.decay_above_pixel_area ** max(0.0, (area_ratio - 1.0))
    vram_factor = float(total_vram_gb) / row.base_vram_gb
    batch = int(round(row.base_batch_per_gpu * area_factor * vram_factor))
    return max(1, batch)


# ---------------------------------------------------------------------------
# Config generator
# ---------------------------------------------------------------------------


_CONFIG_OVERRIDE_TEMPLATE = """

# --- kwcoco_detector_kit overrides (Phase 2) ---
label_list = {label_list!r}
batch_size = {batch_size}
lr = {lr}
lr_backbone = {backbone_lr}
epochs = {num_epochs}
"""


def _resolve_repo() -> Optional[Path]:
    """The Open-GroundingDino checkout, in priority order:

    1. ``$KCD_OPENGROUNDINGDINO_REPO_DPATH`` (explicit override)
    2. ``<kit_root>/tpl/Open-GroundingDino`` (the kit's own submodule)
    3. None
    """
    repo = os.environ.get("KCD_OPENGROUNDINGDINO_REPO_DPATH")
    if repo:
        return Path(repo).expanduser().resolve()
    # Fall back to the kit's tpl submodule when on disk.
    import kwcoco_detector_kit
    pkg_init = Path(kwcoco_detector_kit.__file__).resolve()
    candidate = pkg_init.parent.parent / "tpl" / "Open-GroundingDino"
    if candidate.is_dir() and any(candidate.iterdir()):
        return candidate
    return None


def _write_overlay_config(upstream_cfg_fpath: Path, out_fpath: Path, *,
                           label_list: List[str], batch_size: int, lr: float,
                           backbone_lr: float, num_epochs: int) -> Path:
    """Copy upstream cfg + append the kit's override block.

    The upstream config is a Python file (cfg_odvg.py) not a YAML, so we
    write a copy-with-overrides rather than a structured-merge.
    """
    base_text = upstream_cfg_fpath.read_text()
    # Flip the use_coco_eval flag the same way the v9 pipeline does.
    base_text = base_text.replace("use_coco_eval = True", "use_coco_eval = False")
    overlay = _CONFIG_OVERRIDE_TEMPLATE.format(
        label_list=label_list,
        batch_size=int(batch_size),
        lr=float(lr),
        backbone_lr=float(backbone_lr),
        num_epochs=int(num_epochs),
    )
    out_fpath.write_text(base_text + overlay)
    return out_fpath


def _write_datasets_json(out_fpath: Path, *,
                          prep_dpath: Path,
                          train_odvg_fpath: Path,
                          vali_mscoco_fpath: Path,
                          label_map_fpath: Path) -> Path:
    payload = {
        "train": [{
            "root": str(prep_dpath),
            "anno": str(train_odvg_fpath),
            "label_map": str(label_map_fpath),
            "dataset_mode": "odvg",
        }],
        "val": [{
            "root": str(prep_dpath),
            "anno": str(vali_mscoco_fpath),
            "label_map": None,
            "dataset_mode": "coco",
        }],
    }
    out_fpath.write_text(json.dumps(payload, indent=2))
    return out_fpath


def _coco_to_odvg(coco_json_fpath: Path, out_fpath: Path, *,
                   repo: Path) -> Path:
    """Subprocess OpenGroundingDINO's ``tools/coco2odvg.py``.

    Raises subprocess.CalledProcessError if the upstream tool can't run
    (commonly because the env lacks ``jsonlines`` — see lesson #25 and
    the ``[opengroundingdino]`` extras group).
    """
    tool = repo / "tools" / "coco2odvg.py"
    if not tool.exists():
        raise FileNotFoundError(tool)
    cmd = [
        sys.executable, str(tool),
        "--input", str(coco_json_fpath),
        "--output", str(out_fpath),
        "--idmap=False",
    ]
    subprocess.run(cmd, check=True, cwd=str(repo))
    return out_fpath


def _write_label_map(coco_json_fpath: Path, out_fpath: Path) -> Path:
    data = json.loads(coco_json_fpath.read_text())
    label_map = {str(cat["id"]): cat["name"] for cat in data.get("categories", [])}
    out_fpath.write_text(json.dumps(label_map, indent=2))
    return out_fpath


def _tail_text(fpath: Path, *, max_lines: int = 80) -> str:
    """Return the tail of a text log, tolerating partial UTF-8."""
    if not fpath.exists():
        return ""
    text = fpath.read_text(errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


# ---------------------------------------------------------------------------
# Predictor adapter
# ---------------------------------------------------------------------------


class OpenGroundingDINOPredictor:
    """Inference adapter for a trained OpenGroundingDINO checkpoint.

    Requires the OpenGroundingDINO submodule to be importable. Phase 2
    keeps this minimal — full inference is exercised by the upstream
    repo's tools; the kit drives the orchestration layer.
    """

    def __init__(self, ckpt_fpath, config_fpath, device: str = "cpu",
                 label_list: Optional[List[str]] = None):
        repo = _resolve_repo()
        if repo and str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            # Defer the import until __init__ — the upstream repo's
            # init has heavy module-level side effects.
            from groundingdino.util.inference import (  # type: ignore
                load_model, predict,
            )
        except Exception as ex:
            raise ImportError(
                "OpenGroundingDINOPredictor needs the OpenGroundingDINO "
                "submodule on PYTHONPATH; set "
                "$KCD_OPENGROUNDINGDINO_REPO_DPATH."
            ) from ex
        self._device = device
        self._label_list = list(label_list or ["widget"])
        self._model = load_model(str(config_fpath), str(ckpt_fpath))
        self._predict = predict
        # eval_spatial_size — OGDino is flexible; report a typical inference size.
        self._eval_h = 800
        self._eval_w = 800

    @property
    def eval_spatial_size(self) -> Tuple[int, int]:
        return (self._eval_h, self._eval_w)

    def predict_image(self, image_np, orig_size):
        # OpenGroundingDINO consumes a PIL/torch image with text prompt.
        from PIL import Image
        import torch
        import groundingdino.datasets.transforms as T  # type: ignore

        pil = Image.fromarray(image_np)
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(pil, None)
        boxes, logits, phrases = self._predict(
            model=self._model,
            image=image_transformed,
            caption=" . ".join(self._label_list) + " .",
            box_threshold=0.30,
            text_threshold=0.25,
            device=self._device,
        )
        W, H = int(orig_size[0]), int(orig_size[1])
        out: List[dict] = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            # OGDino boxes are normalized cxcywh; convert to pixel xyxy.
            cx, cy, bw, bh = [float(v) for v in box]
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            label_idx = self._label_list.index(phrase) if phrase in self._label_list else 0
            out.append({
                "label": int(label_idx),
                "bbox_xyxy": [x1, y1, x2, y2],
                "score": float(logit),
            })
        return out


# ---------------------------------------------------------------------------
# Trainer plugin
# ---------------------------------------------------------------------------


@register_trainer
class OpenGroundingDINOTrainer:
    """Driver for OpenGroundingDINO's train_dist.sh — DINOv2 + BERT + DETR."""

    name = "opengroundingdino"
    variants = VARIANTS
    supports_onnx_export = False  # OGDino has no canonical ONNX exporter

    def generate_config(
        self,
        train_kwcoco_fpath,
        vali_kwcoco_fpath,
        workdir,
        *,
        variant: str = "opengroundingdino_swint",
        input_hw: Tuple[int, int] = (800, 800),
        train_policy: str = "fixed",   # OGDino's collate isn't size-keyed; this is informational.
        num_classes: int = 1,
        batch_size: int = 4,
        val_batch_size: int = 4,
        num_epochs: int = 15,
        lr: float = 1e-4,
        backbone_lr: float = 1e-5,
        use_amp: bool = True,
        channels: str = "r|g|b",
        scale_tier: str = "L",
        num_gpus: int = 1,
        data_format: str = "kwcoco",
        extra: Optional[dict] = None,
    ) -> Path:
        if variant not in VARIANTS:
            raise KeyError(
                f"unknown OpenGroundingDINO variant {variant!r}; "
                f"known: {list(VARIANTS)}"
            )
        workdir = Path(workdir)
        prep_dpath = workdir / "detector_prepared"
        prep_dpath.mkdir(parents=True, exist_ok=True)
        gen_dpath = workdir / "generated_configs"
        gen_dpath.mkdir(parents=True, exist_ok=True)

        category_name = (extra or {}).get("category_name", "widget")
        label_list = (extra or {}).get("label_list", [category_name])

        # 1. kwcoco -> MSCOCO json for both splits
        from kwcoco_detector_kit.data.coco_export import export_mscoco

        train_mscoco = prep_dpath / "train.mscoco.json"
        vali_mscoco = prep_dpath / "vali.mscoco.json"
        export_mscoco(
            train_kwcoco_fpath, train_mscoco,
            category_name=category_name,
            include_segmentations=False,
            category_id=0,
        )
        export_mscoco(
            vali_kwcoco_fpath, vali_mscoco,
            category_name=category_name,
            include_segmentations=False,
            category_id=0,
        )

        # 2. Convert train MSCOCO -> ODVG (needs the OGDino submodule).
        repo = _resolve_repo()
        train_odvg = prep_dpath / "train.odvg.jsonl"
        if repo:
            try:
                _coco_to_odvg(train_mscoco, train_odvg, repo=repo)
            except subprocess.CalledProcessError as ex:
                # coco2odvg.py needs `jsonlines` and a few other transitive
                # deps. When they're missing, fall back to the stub-config
                # path so the rest of generate_config still completes — the
                # later launch() step will fail loudly if the user actually
                # tries to train. See lesson #25.
                import warnings as _warnings
                _warnings.warn(
                    f"opengroundingdino: coco2odvg conversion failed "
                    f"(exit {ex.returncode}); generated config is a stub. "
                    "Install the [opengroundingdino] extras "
                    "(pip install -e '.[opengroundingdino]') or run "
                    "`kwcoco-detector-kit check-env --groups opengroundingdino "
                    "--strict_import` for the actionable hint."
                )
            except FileNotFoundError:
                pass
        else:
            # Phase-1-style "generate config without launching" fallback:
            # leave the ODVG file unwritten and document the path. The
            # launch() step will raise informatively if the submodule's
            # missing at run time.
            pass

        # 3. Label map
        label_map_fpath = prep_dpath / "label_map.json"
        _write_label_map(train_mscoco, label_map_fpath)

        # 4. datasets.json
        datasets_json_fpath = prep_dpath / "datasets.json"
        _write_datasets_json(
            datasets_json_fpath,
            prep_dpath=prep_dpath,
            train_odvg_fpath=train_odvg,
            vali_mscoco_fpath=vali_mscoco,
            label_map_fpath=label_map_fpath,
        )

        # 5. OGDino config (Python file, not YAML)
        info = VARIANTS[variant]
        gen_cfg_fpath = gen_dpath / "ogdino_cfg.py"
        if repo:
            upstream_cfg_fpath = repo / info["upstream_config_relpath"]
            _write_overlay_config(
                upstream_cfg_fpath, gen_cfg_fpath,
                label_list=label_list,
                batch_size=int(batch_size),
                lr=float(lr),
                backbone_lr=float(backbone_lr),
                num_epochs=int(num_epochs),
            )
        else:
            # Without the repo on disk, emit a stub Python config that
            # documents the overrides — used by tests and by users who
            # want to preview the config before installing the submodule.
            stub_text = (
                "# kwcoco_detector_kit OpenGroundingDINO config preview\n"
                "# Real upstream cfg_odvg.py is required at launch time.\n"
                f"label_list = {label_list!r}\n"
                f"batch_size = {int(batch_size)}\n"
                f"lr = {float(lr)}\n"
                f"lr_backbone = {float(backbone_lr)}\n"
                f"epochs = {int(num_epochs)}\n"
                "use_coco_eval = False\n"
            )
            gen_cfg_fpath.write_text(stub_text)

        # 6. policy.json — same shape as DEIMv2's so eligibility.py joins.
        H, W = int(input_hw[0]), int(input_hw[1])
        candidate_id = (extra or {}).get(
            "candidate_id",
            f"{variant}_{H}x{W}",
        )
        policy = {
            "candidate_id": candidate_id,
            "variant": variant,
            "candidate_kind": "real",
            "run_tag": os.environ.get("KCD_RUN_TAG", ""),
            "export_input_h": H,
            "export_input_w": W,
            "train_resolution_policy": str(train_policy),
            "requested_train_resolution_min": H,
            "requested_train_resolution_max": H,
            "multiscale_base_size": H,
            "multiscale_repeat": 0,
            "multiscale_stop_epoch": int(num_epochs),
            "train_batch": int(batch_size),
            "val_batch": int(val_batch_size),
            "num_epochs": int(num_epochs),
            "lr": float(lr),
            "backbone_lr": float(backbone_lr),
            "use_amp": bool(use_amp),
            "init_ckpt": str((extra or {}).get("init_checkpoint", "")),
            "generated_train_cfg": str(gen_cfg_fpath),
            "effective_train_scales": [H],
            "effective_train_scale_min": H,
            "effective_train_scale_max": H,
            "label_list": label_list,
        }
        (workdir / "policy.json").write_text(json.dumps(policy, indent=2))

        return gen_cfg_fpath

    def launch(
        self,
        config_fpath,
        *,
        init_checkpoint=None,
        resume=None,
        num_gpus: int = 1,
        distributed: bool = True,
    ) -> Path:
        """Subprocess OpenGroundingDINO's train_dist.sh."""
        cfg_fpath = Path(config_fpath)
        workdir = cfg_fpath.parent.parent
        repo = _resolve_repo()
        if not repo:
            raise EnvironmentError(
                "OpenGroundingDINO launch needs a checkout. Either set "
                "$KCD_OPENGROUNDINGDINO_REPO_DPATH or run "
                "`git submodule update --init tpl/Open-GroundingDino` from "
                "the kit's repo root."
            )
        train_sh = repo / "train_dist.sh"
        if not train_sh.exists():
            raise FileNotFoundError(train_sh)

        raise_nofile_limit(target=65536)

        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        env["GPU_NUM"] = str(int(num_gpus))
        env["CFG"] = str(cfg_fpath)
        env["DATASETS"] = str(workdir / "detector_prepared" / "datasets.json")
        env["OUTPUT_DIR"] = str(workdir)
        if init_checkpoint:
            env["PRETRAIN_MODEL_PATH"] = str(init_checkpoint)
        env.setdefault("TEXT_ENCODER_TYPE", "bert-base-uncased")
        policy_fpath = workdir / "policy.json"
        if policy_fpath.exists():
            policy = json.loads(policy_fpath.read_text())
            if bool(policy.get("use_amp", False)):
                env["USE_AMP"] = "1"

        cmd = [
            "bash", str(train_sh),
            str(int(num_gpus)),
            str(cfg_fpath),
            str(workdir / "detector_prepared" / "datasets.json"),
            str(workdir),
        ]
        log_fpath = workdir / "train.log"
        (workdir / "train_command.json").write_text(json.dumps({
            "cmd": cmd,
            "cwd": str(repo),
            "log": str(log_fpath),
        }, indent=2))
        with log_fpath.open("w") as log_file:
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=str(repo),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
            ret = proc.wait()
        if ret != 0:
            tail = _tail_text(log_fpath)
            raise RuntimeError(
                "OpenGroundingDINO training failed "
                f"(exit {ret}). Full log: {log_fpath}\n\n"
                f"--- train.log tail ---\n{tail}"
            )
        return workdir

    def find_checkpoint(self, workdir) -> Path:
        workdir = Path(workdir)
        # OpenGroundingDINO writes per-epoch checkpoint%04d.pth files;
        # pick the highest-numbered one as the canonical "last".
        cands = sorted(workdir.glob("checkpoint*.pth"))
        if cands:
            return cands[-1]
        raise FileNotFoundError(f"no OpenGroundingDINO checkpoint in {workdir}")

    def supports_dynamic_input(self, variant: str) -> bool:
        if variant not in VARIANTS:
            raise KeyError(variant)
        return bool(VARIANTS[variant]["supports_dynamic_input"])

    def memory_tier_default_batch(
        self,
        variant: str,
        input_hw: Tuple[int, int],
        total_vram_gb: float,
    ) -> int:
        return _batch_for(variant, tuple(input_hw), float(total_vram_gb))

    def supports_webdataset_input(self) -> bool:
        return False  # Phase 2: kwcoco only

    def build_predictor(self, workdir, *, device: str = "cpu"):
        workdir = Path(workdir)
        ckpt = self.find_checkpoint(workdir)
        cfg = workdir / "generated_configs" / "ogdino_cfg.py"
        policy = {}
        pol_fpath = workdir / "policy.json"
        if pol_fpath.exists():
            policy = json.loads(pol_fpath.read_text())
        label_list = policy.get("label_list", ["widget"])
        return OpenGroundingDINOPredictor(
            ckpt, cfg, device=device, label_list=label_list,
        )
