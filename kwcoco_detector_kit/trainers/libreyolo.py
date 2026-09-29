"""LibreYOLO model-engine integration.

KDK owns the rich data/control plane: kwcoco remains authoritative, prepared
windows remain KDK artifacts, and source-space evaluation/mining stays in KDK.
LibreYOLO owns model-family mechanics: model construction, losses, optimizers,
DDP, checkpointing, and family-native inference/export.

The initial bridge intentionally materializes only COCO *annotations*. Image
paths in that COCO JSON remain absolute references to the KDK-prepared kwcoco
view, so integrating a model engine does not create another tile corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Optional, Tuple

import yaml

from kwcoco_detector_kit.trainers._interface import _pin_checkpoint
from kwcoco_detector_kit.trainers._registry import register_trainer


def _variant(
    family: str,
    size: str,
    *,
    task: str = "detect",
    resolution: int = 640,
    base_batch_24gb: int = 4,
    dynamic: bool = True,
    supports_backbone_lr: bool = False,
) -> dict:
    suffix = "-seg" if task == "segment" else ""
    prefixes = {
        "deimv2": "LibreDEIMv2",
        "dfine": "LibreDFINE",
        "rfdetr": "LibreRFDETR",
        "yolo9": "LibreYOLO9",
        "gtr": "LibreGTR",
        "tinyformer": "LibreTinyFormer",
    }
    return {
        "family": family,
        "size": size,
        "task": task,
        "model_ref": f"{prefixes[family]}{size}{suffix}.pt",
        "default_resolution": int(resolution),
        "base_batch_24gb": int(base_batch_24gb),
        "supports_dynamic_input": bool(dynamic),
        "supports_backbone_lr": bool(supports_backbone_lr),
        "supports_masks": task == "segment",
    }


def _build_variants() -> dict[str, dict]:
    """Trainable modern families exposed through one KDK backend.

    Keep this declarative. The integration is intended to make adding a new
    LibreYOLO family a catalog change unless its data/task contract differs.
    """
    variants: dict[str, dict] = {}

    for size, resolution, batch in [
        ("atto", 320, 32),
        ("femto", 416, 24),
        ("pico", 640, 16),
        ("n", 640, 12),
        ("s", 640, 8),
        ("m", 640, 6),
        ("l", 640, 4),
        ("x", 640, 2),
    ]:
        variants[f"deimv2_{size}"] = _variant(
            "deimv2", size, resolution=resolution, base_batch_24gb=batch,
            dynamic=size in {"s", "m", "l", "x"},
            supports_backbone_lr=True,
        )

    for size, batch in [("n", 12), ("s", 8), ("m", 6), ("l", 4), ("x", 2)]:
        variants[f"dfine_{size}"] = _variant(
            "dfine", size, base_batch_24gb=batch, dynamic=size != "n",
            supports_backbone_lr=True,
        )
        variants[f"dfine_{size}_seg"] = _variant(
            "dfine", size, task="segment", base_batch_24gb=max(1, batch // 2),
            dynamic=size != "n", supports_backbone_lr=True,
        )

    for size, resolution, batch in [
        ("n", 384, 12), ("s", 512, 8), ("m", 576, 6), ("l", 704, 3),
    ]:
        variants[f"rfdetr_{size}"] = _variant(
            "rfdetr", size, resolution=resolution, base_batch_24gb=batch,
            supports_backbone_lr=True,
        )
    for size, resolution, batch in [
        ("n", 312, 10), ("s", 384, 8), ("m", 432, 6), ("l", 504, 4),
        ("x", 624, 2), ("xx", 768, 1),
    ]:
        variants[f"rfdetr_{size}_seg"] = _variant(
            "rfdetr", size, task="segment", resolution=resolution,
            base_batch_24gb=batch, supports_backbone_lr=True,
        )

    for size, batch in [("t", 20), ("s", 14), ("m", 8), ("c", 4)]:
        variants[f"yolo9_{size}"] = _variant(
            "yolo9", size, base_batch_24gb=batch, dynamic=False,
        )

    for size, batch in [("s", 10), ("m", 7), ("l", 4), ("x", 2)]:
        variants[f"gtr_{size}"] = _variant(
            "gtr", size, base_batch_24gb=batch, supports_backbone_lr=True,
        )
        variants[f"gtr_{size}_seg"] = _variant(
            "gtr", size, task="segment", base_batch_24gb=max(1, batch // 2),
            supports_backbone_lr=True,
        )

    for size, batch in [("s", 10), ("m", 7), ("l", 4), ("x", 3), ("xl", 2)]:
        variants[f"tinyformer_{size}"] = _variant(
            "tinyformer", size, base_batch_24gb=batch, supports_backbone_lr=True,
        )
    return variants


VARIANTS = _build_variants()


def _kit_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_libreyolo_repo() -> Path:
    """Resolve the pinned source checkout used by this integration."""
    override = os.environ.get("KCD_LIBREYOLO_REPO_DPATH")
    repo = Path(override).expanduser().resolve() if override else _kit_root() / "tpl" / "libreyolo"
    if not (repo / "libreyolo" / "__init__.py").is_file():
        raise FileNotFoundError(
            "LibreYOLO source checkout not found. Initialize tpl/libreyolo with "
            "`git submodule update --init --recursive`, or set "
            "KCD_LIBREYOLO_REPO_DPATH. Resolved path: " + str(repo)
        )
    return repo


def _ensure_libreyolo_importable():
    """Import LibreYOLO from the pinned submodule (or explicit override)."""
    repo = _resolve_libreyolo_repo()
    repo_s = str(repo)
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)
    importlib.invalidate_caches()
    module = importlib.import_module("libreyolo")
    module_path = Path(module.__file__).resolve()
    try:
        module_path.relative_to(repo)
    except ValueError as ex:
        raise RuntimeError(
            "libreyolo was already imported from a different installation: "
            f"{module_path}. KDK pins the model engine to {repo}. Start a fresh "
            "process or remove the conflicting import."
        ) from ex
    return module


def _sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _prepare_coco_bridge(
    train_src: str | Path,
    vali_src: str | Path,
    output_dpath: str | Path,
    *,
    category_names: list[str],
    task: str,
) -> Path:
    """Prepare LibreYOLO's native-COCO dataset view without copying images."""
    from kwcoco_detector_kit.data.coco_export import export_mscoco

    output_dpath = Path(output_dpath).resolve()
    output_dpath.mkdir(parents=True, exist_ok=True)
    train_src = Path(train_src).resolve()
    vali_src = Path(vali_src).resolve()
    include_segmentations = task == "segment"

    train_json = output_dpath / "train.mscoco.json"
    vali_json = output_dpath / "vali.mscoco.json"
    yaml_fpath = output_dpath / "dataset.yaml"
    receipt_fpath = output_dpath / "PREPARED.json"
    expected = {
        "schema_version": 1,
        "bridge": "kwcoco_to_libreyolo_native_coco",
        "train_src": str(train_src),
        "train_sha256": _sha256_file(train_src),
        "vali_src": str(vali_src),
        "vali_sha256": _sha256_file(vali_src),
        "category_names": list(category_names),
        "task": str(task),
        "include_segmentations": bool(include_segmentations),
    }
    if receipt_fpath.is_file():
        try:
            prior = json.loads(receipt_fpath.read_text())
        except Exception:
            prior = None
        if prior == expected and train_json.is_file() and vali_json.is_file() and yaml_fpath.is_file():
            return yaml_fpath

    export_mscoco(
        train_src,
        train_json,
        category_names=category_names,
        include_segmentations=include_segmentations,
        category_id_start=0,
        progress=True,
    )
    export_mscoco(
        vali_src,
        vali_json,
        category_names=category_names,
        include_segmentations=include_segmentations,
        category_id_start=0,
        progress=True,
    )

    # KDK's COCO exporter writes absolute source-image paths. The split roots
    # therefore only need to exist; joining an absolute file_name preserves it.
    dataset = {
        "path": str(output_dpath),
        "train": ".",
        "val": ".",
        "annotations": {
            "train": str(train_json),
            "val": str(vali_json),
        },
        "nc": len(category_names),
        "names": {idx: name for idx, name in enumerate(category_names)},
    }
    yaml_fpath.write_text(yaml.safe_dump(dataset, sort_keys=False))
    tmp = receipt_fpath.with_suffix(".tmp")
    tmp.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    tmp.replace(receipt_fpath)
    return yaml_fpath


def _device_arg(num_gpus: int):
    num_gpus = int(num_gpus)
    if num_gpus <= 0:
        return "cpu"
    if num_gpus == 1:
        return "0"
    return list(range(num_gpus))


def _build_train_kwargs(cfg: dict, *, resume=None, num_gpus: Optional[int] = None) -> dict:
    """Translate KDK's stable run config to LibreYOLO's shared train surface."""
    train = cfg["train"]
    model = cfg["model"]
    runtime = cfg["runtime"]
    if num_gpus is None:
        num_gpus = int(runtime["num_gpus"])
    kwargs = {
        "data": str(cfg["data"]["yaml"]),
        "epochs": int(train["epochs"]),
        "batch": int(train["batch"]),
        "imgsz": int(model["input_hw"][0]),
        "lr0": float(train["lr"]),
        "device": _device_arg(int(num_gpus)),
        "workers": int(train["workers"]),
        "seed": int(train["seed"]),
        "project": str(Path(cfg["workdir"]).parent),
        "name": Path(cfg["workdir"]).name,
        "exist_ok": True,
        "amp": bool(train["amp"]),
        "patience": int(train["patience"]),
    }
    if resume:
        kwargs["resume"] = str(Path(resume).resolve())
    else:
        kwargs["resume"] = False

    # These are common TrainConfig fields on the model families in this
    # catalog. Keep backend-specific knobs explicit rather than forwarding all
    # of KDK's orchestration ``extra`` dictionary into LibreYOLO.
    if bool(model.get("supports_dynamic_input", False)):
        kwargs["multi_scale"] = str(train["policy"]) != "fixed"
    if bool(model.get("supports_backbone_lr", False)) and float(train["lr"]) > 0:
        kwargs["backbone_lr_mult"] = float(train["backbone_lr"]) / float(train["lr"])
    kwargs.update(dict(train.get("libreyolo_train_kwargs") or {}))
    return kwargs


def _load_model(cfg: dict, *, init_checkpoint=None, resume=None, device="cpu"):
    _ensure_libreyolo_importable()
    from libreyolo import LibreYOLO

    model_cfg = cfg["model"]
    source = resume or init_checkpoint or model_cfg["model_ref"]
    return LibreYOLO(
        str(source),
        size=str(model_cfg["size"]),
        task=str(model_cfg["task"]),
        device=str(device),
    )


def _run_training_config(config_fpath, *, init_checkpoint=None, resume=None, num_gpus=None):
    cfg = json.loads(Path(config_fpath).read_text())
    if init_checkpoint is None:
        init_checkpoint = cfg.get("init_checkpoint")
    if resume and init_checkpoint:
        init_checkpoint = None
    model = _load_model(
        cfg,
        init_checkpoint=init_checkpoint,
        resume=resume,
        device="cpu" if int(num_gpus or cfg["runtime"]["num_gpus"]) <= 0 else "auto",
    )
    kwargs = _build_train_kwargs(cfg, resume=resume, num_gpus=num_gpus)
    result = model.train(**kwargs)
    summary = {
        "schema": "kwcoco_detector_kit.libreyolo_result.v1",
        "family": cfg["model"]["family"],
        "variant": cfg["variant"],
        "task": cfg["model"]["task"],
        "save_dir": str(result.get("save_dir") or result.get("output_dir") or cfg["workdir"]),
        "best_checkpoint": str(result.get("best_checkpoint") or ""),
    }
    Path(cfg["workdir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["workdir"]) / "libreyolo_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return result


def _launcher_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--num-gpus", type=int, required=True)
    args = parser.parse_args(argv)
    _run_training_config(
        args.config,
        init_checkpoint=args.init_checkpoint,
        resume=args.resume,
        num_gpus=args.num_gpus,
    )


_LAUNCHER_TEXT = """#!/usr/bin/env python\nfrom kwcoco_detector_kit.trainers.libreyolo import _launcher_main\n\nif __name__ == '__main__':\n    _launcher_main()\n"""


def _as_numpy(data):
    if data is None:
        return None
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()
    return data


class LibreYOLOPredictor:
    """Adapt LibreYOLO ``Results`` to KDK's source-window record contract."""

    def __init__(self, model_source, policy, device="cpu"):
        _ensure_libreyolo_importable()
        from libreyolo import LibreYOLO

        self._H, self._W = [int(v) for v in policy["input_hw"]]
        self._threshold = float(policy.get("predict_score_floor", 0.0))
        self._num_classes = int(policy["num_classes"])
        framework = policy.get("framework") or {}
        self._task = str(framework.get("task") or policy.get("task") or "detect")
        self._size = str(framework.get("size") or "") or None
        self._model = LibreYOLO(
            str(model_source),
            size=self._size,
            task=self._task,
            device=str(device),
        )

    @property
    def eval_spatial_size(self):
        return self._H, self._W

    def set_score_thresh(self, score_thresh):
        self._threshold = float(score_thresh)

    def _records(self, result, *, orig_size, actual_hw):
        import cv2
        import numpy as np

        boxes_obj = getattr(result, "boxes", None)
        if boxes_obj is None:
            return []
        boxes_obj = boxes_obj.numpy() if hasattr(boxes_obj, "numpy") else boxes_obj
        boxes = np.asarray(_as_numpy(boxes_obj.xyxy), dtype=np.float32)
        scores = np.asarray(_as_numpy(boxes_obj.conf), dtype=np.float32)
        labels = np.asarray(_as_numpy(boxes_obj.cls), dtype=np.int64)
        masks_obj = getattr(result, "masks", None)
        masks = None
        if masks_obj is not None:
            masks_obj = masks_obj.numpy() if hasattr(masks_obj, "numpy") else masks_obj
            masks = np.asarray(_as_numpy(masks_obj.data))

        actual_h, actual_w = [int(v) for v in actual_hw]
        target_w, target_h = [int(v) for v in orig_size]
        sx = target_w / float(max(1, actual_w))
        sy = target_h / float(max(1, actual_h))
        records = []
        for idx, box0 in enumerate(boxes):
            label = int(labels[idx])
            score = float(scores[idx])
            if label < 0 or label >= self._num_classes or score < self._threshold:
                continue
            box = box0.astype(float, copy=True)
            box[[0, 2]] *= sx
            box[[1, 3]] *= sy
            rec = {"label": label, "bbox_xyxy": box.tolist(), "score": score}
            if masks is not None and idx < len(masks):
                mask = np.asarray(masks[idx]) > 0.5
                if mask.shape != (target_h, target_w):
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (target_w, target_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                rec["mask"] = mask
            records.append(rec)
        return records

    def _predict(self, source, *, batch=1):
        return self._model.predict(
            source,
            conf=float(self._threshold),
            imgsz=int(self._H) if self._H == self._W else (self._H, self._W),
            batch=int(batch),
            color_format="rgb",
        )

    def predict_image(self, image_np, orig_size):
        result = self._predict(image_np, batch=1)
        if isinstance(result, (list, tuple)):
            result = result[0]
        return self._records(
            result,
            orig_size=orig_size,
            actual_hw=image_np.shape[:2],
        )

    def predict_batch(self, images_np, orig_sizes):
        if not images_np:
            return []
        results = self._predict(list(images_np), batch=len(images_np))
        if not isinstance(results, (list, tuple)):
            results = [results]
        if len(results) != len(images_np):
            raise RuntimeError(
                f"LibreYOLO returned {len(results)} results for {len(images_np)} images"
            )
        return [
            self._records(result, orig_size=orig_size, actual_hw=image.shape[:2])
            for result, image, orig_size in zip(results, images_np, orig_sizes)
        ]


@register_trainer
class LibreYOLOTrainer:
    """General KDK backend for LibreYOLO trainable detector families."""

    name = "libreyolo"
    variants = VARIANTS
    supports_onnx_export = True

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
        batch_size: int = 4,
        val_batch_size: int = 8,
        num_epochs: int = 60,
        lr: float = 1e-4,
        backbone_lr: float = 1e-5,
        use_amp: bool = True,
        init_checkpoint: Optional[str] = None,
        channels: str = "r|g|b",
        scale_tier: str = "M",
        num_gpus: int = 1,
        data_format: str = "kwcoco",
        extra: Optional[dict] = None,
    ) -> Path:
        extra = dict(extra or {})
        if variant not in VARIANTS:
            raise KeyError(f"unknown LibreYOLO variant {variant!r}; choose {sorted(VARIANTS)}")
        info = dict(VARIANTS[variant])
        if str(channels) != "r|g|b":
            raise ValueError(
                "The current LibreYOLO detector catalog consumes RGB tensors. "
                "Keep multispectral/source assets in kwcoco and prepare an RGB view, "
                "or add a model family with an explicit non-RGB input contract."
            )
        if str(data_format) not in {"kwcoco", "coco"}:
            raise ValueError(
                "LibreYOLO's first KDK bridge accepts kwcoco/COCO prepared views; "
                f"got data_format={data_format!r}"
            )
        h, w = [int(v) for v in input_hw]
        if h != w:
            raise ValueError(
                "The initial LibreYOLO catalog uses square detector canvases; "
                f"got input_hw={(h, w)}"
            )
        requested_policy = str(train_policy)
        if requested_policy not in {"fixed", "multiscale"}:
            raise ValueError(
                "LibreYOLO's common KDK bridge currently supports "
                "train_policy='fixed' or 'multiscale'. Exact KDK multiscale "
                f"range policies are not representable generically; got {requested_policy!r}."
            )
        effective_policy = requested_policy
        if requested_policy == "multiscale" and not info["supports_dynamic_input"]:
            warnings.warn(
                f"LibreYOLO variant {variant!r} does not expose family-native "
                "multiscale training; coercing train_policy='multiscale' -> 'fixed'.",
                stacklevel=2,
            )
            effective_policy = "fixed"
        category_names = list(extra.get("category_names") or ["object"])
        if len(category_names) != int(num_classes):
            raise ValueError(
                f"num_classes={num_classes} != len(category_names)={len(category_names)}"
            )
        workdir = Path(workdir).resolve()
        gen_dpath = workdir / "generated_configs"
        gen_dpath.mkdir(parents=True, exist_ok=True)
        dataset_yaml = _prepare_coco_bridge(
            train_kwcoco_fpath,
            vali_kwcoco_fpath,
            workdir / "detector_prepared" / "libreyolo_coco",
            category_names=category_names,
            task=info["task"],
        )
        config = {
            "schema_version": 1,
            "trainer": self.name,
            "variant": variant,
            "workdir": str(workdir),
            "model": {
                **info,
                "input_hw": [h, w],
                "num_classes": int(num_classes),
            },
            "data": {
                "yaml": str(dataset_yaml),
                "train_kwcoco": str(Path(train_kwcoco_fpath).resolve()),
                "vali_kwcoco": str(Path(vali_kwcoco_fpath).resolve()),
                "bridge": "native_coco",
            },
            "train": {
                "policy": effective_policy,
                "requested_policy": requested_policy,
                "batch": int(batch_size),
                "val_batch": int(val_batch_size),
                "epochs": int(num_epochs),
                "lr": float(lr),
                "backbone_lr": float(backbone_lr),
                "amp": bool(use_amp),
                "workers": int(extra.get("num_workers", 8)),
                "seed": int(extra.get("seed", 0)),
                "patience": int(extra.get("patience", 50)),
                "libreyolo_train_kwargs": dict(extra.get("libreyolo_train_kwargs") or {}),
            },
            "runtime": {
                "num_gpus": int(num_gpus),
                "distributed": int(num_gpus) > 1,
                "scale_tier": str(scale_tier),
            },
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        }
        policy = {
            "trainer": self.name,
            "variant": variant,
            "input_hw": [h, w],
            "train_policy": effective_policy,
            "train_policy_requested": requested_policy,
            "num_classes": int(num_classes),
            "category_names": category_names,
            "channels": str(channels),
            "predict_score_floor": float(extra.get("predict_score_floor", 0.0)),
            "supports_boxes": True,
            "supports_masks": bool(info["supports_masks"]),
            "task": info["task"],
            "framework": {
                "name": "libreyolo",
                "trainer": self.name,
                "family": info["family"],
                "size": info["size"],
                "task": info["task"],
                "model_ref": info["model_ref"],
            },
            "data_contract": {
                "source_of_truth": "kwcoco",
                "training_bridge": "native_coco",
                "prepared_images_copied": False,
                "source_space_evaluation": "kdk",
            },
            "inference": {
                "mode": "windowed",
                "input_hw": [h, w],
                "window": [h, w],
                "overlap": float(extra.get("window_overlap", 0.25)),
                "nms_iou": float(extra.get("nms_iou", 0.5)),
                "supports_masks": bool(info["supports_masks"]),
            },
        }
        cfg_fpath = gen_dpath / "libreyolo_train.json"
        cfg_fpath.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        launcher_fpath = gen_dpath / "launch_libreyolo.py"
        launcher_fpath.write_text(_LAUNCHER_TEXT)
        (workdir / "policy.json").write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
        return cfg_fpath

    def launch(
        self,
        config_fpath,
        *,
        init_checkpoint=None,
        resume=None,
        num_gpus=1,
        distributed=False,
    ) -> Path:
        config_fpath = Path(config_fpath).resolve()
        cfg = json.loads(config_fpath.read_text())
        configured_gpus = int(cfg["runtime"]["num_gpus"])
        if configured_gpus != int(num_gpus):
            raise ValueError(
                f"launch num_gpus={num_gpus} != generated config {configured_gpus}"
            )
        # LibreYOLO owns local DDP spawning. Do not wrap it in torchrun: that
        # would create nested process groups. The distributed flag is retained
        # in KDK's protocol but num_gpus is the operative value here.
        launcher = config_fpath.parent / "launch_libreyolo.py"
        cmd = [
            sys.executable,
            str(launcher),
            "--config",
            str(config_fpath),
            "--num-gpus",
            str(int(num_gpus)),
        ]
        if init_checkpoint:
            cmd += ["--init-checkpoint", str(Path(init_checkpoint).resolve())]
        if resume:
            cmd += ["--resume", str(Path(resume).resolve())]
        env = os.environ.copy()
        paths = [str(_kit_root()), str(_resolve_libreyolo_repo())]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        subprocess.run(cmd, check=True, cwd=str(Path(cfg["workdir"])), env=env)
        return Path(cfg["workdir"])

    def find_checkpoint(self, workdir) -> Path:
        workdir = Path(workdir)
        candidates = [
            workdir / "weights" / "best.pt",
            workdir / "weights" / "last.pt",
            workdir / "best.pt",
            workdir / "last.pt",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        # Some family trainers may place named best checkpoints in weights/.
        named = sorted((workdir / "weights").glob("best*.pt")) if (workdir / "weights").is_dir() else []
        if named:
            return named[0]
        raise FileNotFoundError(f"no LibreYOLO checkpoint found under {workdir}")

    def supports_dynamic_input(self, variant):
        return bool(VARIANTS[variant]["supports_dynamic_input"])

    def memory_tier_default_batch(self, variant, input_hw, total_vram_gb):
        info = VARIANTS[variant]
        h, w = [max(1, int(v)) for v in input_hw]
        native = float(info["default_resolution"])
        area_scale = (native * native) / float(h * w)
        vram_scale = max(0.25, float(total_vram_gb) / 24.0)
        batch = int(info["base_batch_24gb"] * area_scale * vram_scale)
        return max(1, batch)

    def supports_webdataset_input(self):
        return False

    def build_predictor(self, workdir, *, device="cpu", checkpoint=None):
        workdir = Path(workdir)
        checkpoint = _pin_checkpoint(self, workdir, checkpoint)
        policy = json.loads((workdir / "policy.json").read_text())
        return LibreYOLOPredictor(checkpoint, policy, device=device)


if __name__ == "__main__":  # pragma: no cover - generated launcher entrypoint
    _launcher_main()
