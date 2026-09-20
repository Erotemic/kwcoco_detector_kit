"""RF-DETR native instance-segmentation trainer and predictor adapter."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from kwcoco_detector_kit.trainers._interface import _pin_checkpoint
from kwcoco_detector_kit.trainers._registry import register_trainer


VARIANTS = {
    "seg_2xlarge": {
        "upstream_class": "RFDETRSeg2XLarge",
        "default_resolution": 768,
        "input_multiple": 24,
        "supports_dynamic_input": True,
    },
}


def _prepare_roboflow_coco_layout(train_src, vali_src, output_dpath, category_names):
    """Create the directory/annotation layout expected by RF-DETR."""
    from kwcoco_detector_kit.data.coco_export import export_mscoco

    output_dpath = Path(output_dpath).resolve()
    for split, src in [("train", train_src), ("valid", vali_src)]:
        split_dpath = output_dpath / split
        split_dpath.mkdir(parents=True, exist_ok=True)
        export_mscoco(
            src,
            split_dpath / "_annotations.coco.json",
            category_names=category_names,
            include_segmentations=True,
            category_id_start=0,
        )
    return output_dpath


_LAUNCHER_TEXT = r'''#!/usr/bin/env python
import argparse
import json
import os

from rfdetr import RFDETRSeg2XLarge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume")
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    model_kwargs = dict(cfg["model"])
    if args.init_checkpoint:
        model_kwargs["pretrain_weights"] = args.init_checkpoint
    model = RFDETRSeg2XLarge(**model_kwargs)
    train_kwargs = dict(cfg["train"])
    if args.resume:
        train_kwargs["resume"] = args.resume
    # Explicit even under torchrun: upstream otherwise defaults to one device.
    train_kwargs["devices"] = int(cfg["runtime"]["num_gpus"])
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
'''


class RFDETRSegPredictor:
    """Adapt Supervision ``Detections`` to the kit's native-mask records."""

    def __init__(self, checkpoint, policy, device="cpu"):
        from rfdetr import RFDETRSeg2XLarge

        self._H, self._W = [int(v) for v in policy["input_hw"]]
        self._threshold = float(policy.get("predict_score_floor", 0.0))
        self._num_classes = int(policy["num_classes"])
        self._model = RFDETRSeg2XLarge(
            pretrain_weights=str(checkpoint),
            num_classes=int(policy["num_classes"]),
            resolution=self._H,
            device=str(device),
            trust_checkpoint=True,
        )

    @property
    def eval_spatial_size(self):
        return self._H, self._W

    def set_score_thresh(self, score_thresh):
        """Push KDK's requested score floor into upstream mask postprocessing."""
        self._threshold = float(score_thresh)

    @staticmethod
    def _records(detections, orig_size, actual_hw, num_classes=None):
        import cv2
        import numpy as np

        xyxy = np.asarray(detections.xyxy)
        scores = np.asarray(detections.confidence)
        labels = np.asarray(detections.class_id)
        masks = getattr(detections, "mask", None)
        actual_h, actual_w = actual_hw
        target_w, target_h = [int(v) for v in orig_size]
        sx = target_w / float(actual_w)
        sy = target_h / float(actual_h)
        result = []
        for idx in range(len(xyxy)):
            # RF-DETR detection/segmentation heads have num_classes + 1 logits;
            # the final slot is the no-object/background sentinel.  Upstream's
            # high-level predict() preserves that row and labels it
            # "__background__".  It must not enter KDK prediction KWCoco as a
            # detector class.
            if num_classes is not None and int(labels[idx]) >= int(num_classes):
                continue
            box = xyxy[idx].astype(float)
            box[[0, 2]] *= sx
            box[[1, 3]] *= sy
            record = {
                "label": int(labels[idx]),
                "score": float(scores[idx]),
                "bbox_xyxy": box.tolist(),
            }
            if masks is not None:
                mask = np.asarray(masks[idx]).astype(np.uint8)
                if mask.shape != (target_h, target_w):
                    mask = cv2.resize(
                        mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST,
                    )
                record["mask"] = mask.astype(bool)
            result.append(record)
        return result

    def predict_image(self, image_np, orig_size):
        detections = self._model.predict(
            image_np, threshold=self._threshold, shape=(self._H, self._W),
            include_source_image=False,
        )
        return self._records(
            detections, orig_size, image_np.shape[:2], num_classes=self._num_classes
        )

    def predict_batch(self, images_np, orig_sizes):
        if not images_np:
            return []
        detections = self._model.predict(
            list(images_np), threshold=self._threshold, shape=(self._H, self._W),
            include_source_image=False,
        )
        return [
            self._records(det, size, image.shape[:2], num_classes=self._num_classes)
            for det, size, image in zip(detections, orig_sizes, images_np)
        ]


@register_trainer
class RFDETRTrainer:
    """KDK trainer plugin for upstream RF-DETR Segmentation 2XLarge."""

    name = "rfdetr"
    variants = VARIANTS
    supports_onnx_export = True

    def generate_config(
        self,
        train_kwcoco_fpath,
        vali_kwcoco_fpath,
        workdir,
        *,
        variant="seg_2xlarge",
        input_hw=(768, 768),
        train_policy="fixed",
        num_classes=1,
        batch_size=2,
        val_batch_size=4,
        num_epochs=60,
        lr=1e-4,
        backbone_lr=1.5e-4,
        use_amp=True,
        init_checkpoint=None,
        channels="r|g|b",
        scale_tier="XL",
        num_gpus=1,
        data_format="kwcoco",
        extra=None,
    ) -> Path:
        extra = dict(extra or {})
        if variant not in VARIANTS:
            raise KeyError(f"unknown RF-DETR variant {variant!r}; choose {sorted(VARIANTS)}")
        if str(channels) != "r|g|b":
            raise ValueError("RF-DETR adapter currently supports RGB channels only")
        h, w = [int(v) for v in input_hw]
        multiple = VARIANTS[variant]["input_multiple"]
        if h != w or h % multiple:
            raise ValueError(f"RF-DETR Seg 2XL requires square resolution divisible by {multiple}; got {(h, w)}")
        category_names = list(extra.get("category_names") or ["object"])
        if len(category_names) != int(num_classes):
            raise ValueError("num_classes must equal len(extra['category_names'])")

        workdir = Path(workdir).resolve()
        gen_dpath = workdir / "generated_configs"
        gen_dpath.mkdir(parents=True, exist_ok=True)
        dataset_dir = _prepare_roboflow_coco_layout(
            train_kwcoco_fpath, vali_kwcoco_fpath,
            workdir / "detector_prepared" / "rfdetr_coco", category_names,
        )
        config = {
            "schema_version": 1,
            "trainer": self.name,
            "variant": variant,
            "model": {
                "resolution": h,
                "num_classes": int(num_classes),
                "gradient_checkpointing": bool(extra.get("gradient_checkpointing", False)),
                "amp": bool(use_amp),
            },
            "train": {
                "dataset_dir": str(dataset_dir),
                "dataset_file": "roboflow",
                "output_dir": str(workdir),
                "epochs": int(num_epochs),
                "batch_size": int(batch_size),
                "eval_batch_size": int(val_batch_size),
                "grad_accum_steps": int(extra.get("grad_accum_steps", 1)),
                "lr": float(lr),
                "lr_encoder": float(backbone_lr),
                "lr_scheduler": str(extra.get("lr_scheduler", "step")),
                "lr_scheduler_kwargs": dict(extra.get("lr_scheduler_kwargs") or {}),
                "warmup_epochs": float(extra.get("warmup_epochs", 0.0)),
                "num_workers": int(extra.get("num_workers", 8)),
                "multi_scale": str(train_policy) != "fixed",
                "use_ema": bool(extra.get("use_ema", True)),
                "eval_base_model": bool(extra.get("eval_base_model", False)),
                "checkpoint_interval": int(extra.get("checkpoint_interval", 1)),
                "best_model_metric": str(extra.get("best_model_metric", "map")),
                "early_stopping": bool(extra.get("early_stopping", False)),
                "early_stopping_patience": int(extra.get("early_stopping_patience", 10)),
                "early_stopping_min_delta": float(extra.get("early_stopping_min_delta", 0.001)),
                "early_stopping_use_ema": bool(extra.get("early_stopping_use_ema", False)),
                "skip_best_epochs": int(extra.get("skip_best_epochs", 0)),
                "tensorboard": bool(extra.get("tensorboard", True)),
                "wandb": bool(extra.get("wandb", False)),
                "class_names": category_names,
            },
            "runtime": {"num_gpus": int(num_gpus), "distributed": int(num_gpus) > 1},
            "policy": {
                "variant": variant,
                "input_hw": [h, w],
                "num_classes": int(num_classes),
                "category_names": category_names,
                "predict_score_floor": float(extra.get("predict_score_floor", 0.0)),
                "supports_boxes": True,
                "supports_masks": True,
                "framework": {"name": "rfdetr"},
                "preprocess": {
                    "resize": "bilinear_half_pixel_antialias_false",
                    "scale": 1.0 / 255.0,
                    "normalize_mean": [0.485, 0.456, 0.406],
                    "normalize_std": [0.229, 0.224, 0.225],
                },
                "inference": {
                    "mode": "windowed",
                    "input_hw": [h, w],
                    "window": [h, w],
                    "overlap": 0.25,
                    "nms_iou": 0.5,
                    "supports_masks": True,
                },
            },
        }
        cfg_fpath = gen_dpath / "rfdetr_train.json"
        cfg_fpath.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        launcher_fpath = gen_dpath / "launch_rfdetr.py"
        launcher_fpath.write_text(_LAUNCHER_TEXT)
        (workdir / "policy.json").write_text(json.dumps(config["policy"], indent=2) + "\n")
        return cfg_fpath

    def launch(self, config_fpath, *, init_checkpoint=None, resume=None,
               num_gpus=1, distributed=False) -> Path:
        config_fpath = Path(config_fpath).resolve()
        cfg = json.loads(config_fpath.read_text())
        configured_gpus = int(cfg["runtime"]["num_gpus"])
        if configured_gpus != int(num_gpus):
            raise ValueError(f"launch num_gpus={num_gpus} != generated config {configured_gpus}")
        launcher = config_fpath.parent / "launch_rfdetr.py"
        cmd = [sys.executable]
        if bool(distributed) or int(num_gpus) > 1:
            cmd += ["-m", "torch.distributed.run", f"--nproc_per_node={int(num_gpus)}"]
        cmd += [str(launcher), "--config", str(config_fpath)]
        if init_checkpoint:
            cmd += ["--init-checkpoint", str(Path(init_checkpoint).resolve())]
        if resume:
            cmd += ["--resume", str(Path(resume).resolve())]
        subprocess.run(cmd, check=True, cwd=str(Path(cfg["train"]["output_dir"])))
        return Path(cfg["train"]["output_dir"])

    def find_checkpoint(self, workdir) -> Path:
        workdir = Path(workdir)
        for name in ["checkpoint_best_total.pth", "checkpoint_best_ema.pth", "last.ckpt"]:
            candidate = workdir / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"no RF-DETR checkpoint found under {workdir}")

    def supports_dynamic_input(self, variant):
        return bool(VARIANTS[variant]["supports_dynamic_input"])

    def memory_tier_default_batch(self, variant, input_hw, total_vram_gb):
        # Conservative starting points; the aiq smoke stage must measure and
        # raise this before a production run.
        return 2 if float(total_vram_gb) >= 80 else 1

    def supports_webdataset_input(self):
        return False

    def build_predictor(self, workdir, *, device="cpu", checkpoint=None):
        workdir = Path(workdir)
        checkpoint = _pin_checkpoint(self, workdir, checkpoint)
        policy = json.loads((workdir / "policy.json").read_text())
        return RFDETRSegPredictor(checkpoint, policy, device=device)
