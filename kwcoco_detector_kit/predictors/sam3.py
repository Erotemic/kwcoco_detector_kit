"""SAM 3 / SAM 3.1 text-prompted instance-segmentation adapter.

This module intentionally keeps the admission-test integration small.  It
wraps the official image-model API behind KDK's ``DetectorPredictor`` record
contract without registering a full trainer yet.  The training integration is
admitted only after the pretrained model proves useful and fits the target
3090 inference budget.
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import numpy as np


def resolve_sam3_repo() -> Path | None:
    """Resolve the SAM3 source checkout used by KDK.

    Resolution order:
      1. ``$KCD_SAM3_REPO_DPATH``
      2. ``<kdk-root>/tpl/sam3``
      3. None (an installed ``sam3`` package may still be importable)
    """
    if env := os.environ.get("KCD_SAM3_REPO_DPATH"):
        repo = Path(env).expanduser().resolve()
        if not repo.is_dir():
            raise FileNotFoundError(
                f"KCD_SAM3_REPO_DPATH does not exist or is not a directory: {repo}"
            )
        return repo
    try:
        import kwcoco_detector_kit

        kit_root = Path(kwcoco_detector_kit.__file__).resolve().parent.parent
    except Exception:
        return None
    repo = kit_root / "tpl" / "sam3"
    return repo if repo.is_dir() else None


def _ensure_sam3_importable() -> Path | None:
    repo = resolve_sam3_repo()
    if repo is not None and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import sam3  # noqa: F401
    except Exception as ex:
        where = str(repo) if repo is not None else "<not found>"
        raise ImportError(
            "SAM3 is not importable. Initialize/install the KDK fork first, e.g.\n"
            "  git submodule update --init --recursive tpl/sam3\n"
            "  uv pip install -e ./tpl/sam3\n"
            f"Resolved SAM3 repo: {where}\n"
            f"Underlying error: {type(ex).__name__}: {ex}"
        ) from ex
    return repo


def _coerce_numpy(value):
    """Detach a torch-like value to NumPy without importing torch for tests."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class SAM3TextPredictor:
    """Adapt SAM3 text-prompted image inference to KDK detector records.

    Args:
        prompt: Text concept to exhaustively segment, e.g. ``"poop"``.
        device: Torch device. The admission target is normally ``cuda:0``.
        checkpoint_version: ``"sam3"`` or ``"sam3.1"`` when no explicit
            checkpoint is supplied.
        checkpoint: Optional local checkpoint path. Avoids Hugging Face access.
        score_thresh: SAM3 confidence threshold applied before mask upsampling.
        resolution: Model processor resolution. The current image model is 1008.
        amp_dtype: ``bfloat16``, ``float16`` or ``none``. CUDA inference only.
        compile: Forwarded to SAM3's image-model builder.
    """

    def __init__(
        self,
        *,
        prompt: str = "poop",
        device: str = "cuda:0",
        checkpoint_version: str = "sam3.1",
        checkpoint: str | Path | None = None,
        score_thresh: float = 0.05,
        resolution: int = 1008,
        amp_dtype: str = "bfloat16",
        compile: bool = False,
    ):
        _ensure_sam3_importable()
        import torch
        from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
        from sam3.model.sam3_image_processor import Sam3Processor

        self.prompt = str(prompt)
        self.device = str(device)
        self.resolution = int(resolution)
        self.score_thresh = float(score_thresh)
        self.checkpoint_version = str(checkpoint_version)
        self.amp_dtype = str(amp_dtype).lower()
        self.compile = bool(compile)

        if checkpoint is None:
            checkpoint = download_ckpt_from_hf(version=self.checkpoint_version)
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        self.checkpoint = checkpoint

        # Build/load on CPU and move explicitly.  SAM3's builder currently only
        # special-cases device == "cuda", so this also makes cuda:0/cuda:1 sane.
        model = build_sam3_image_model(
            device="cpu",
            eval_mode=True,
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=self.compile,
        )
        self._model = model.to(self.device).eval()
        self._processor = Sam3Processor(
            self._model,
            resolution=self.resolution,
            device=self.device,
            confidence_threshold=self.score_thresh,
        )
        self._torch = torch

    @property
    def eval_spatial_size(self):
        return self.resolution, self.resolution

    def set_score_thresh(self, score_thresh):
        self.score_thresh = float(score_thresh)
        self._processor.set_confidence_threshold(self.score_thresh)

    def _autocast_context(self):
        torch = self._torch
        dev = torch.device(self.device)
        if dev.type != "cuda" or self.amp_dtype in {"none", "false", "off", "fp32"}:
            return contextlib.nullcontext()
        dtype_lut = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        if self.amp_dtype not in dtype_lut:
            raise KeyError(
                f"unknown amp_dtype={self.amp_dtype!r}; choose bfloat16, float16, none"
            )
        return torch.autocast(device_type="cuda", dtype=dtype_lut[self.amp_dtype])

    @staticmethod
    def _state_to_records(state, *, actual_hw, orig_size):
        """Convert a SAM3 processor state to KDK mask records."""
        import cv2

        boxes = _coerce_numpy(state.get("boxes", np.empty((0, 4)))).reshape(-1, 4)
        scores = _coerce_numpy(state.get("scores", np.empty((0,)))).reshape(-1)
        masks = _coerce_numpy(state.get("masks", np.empty((0, 1, *actual_hw))))
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None, ...]

        actual_h, actual_w = [int(v) for v in actual_hw]
        target_w, target_h = [int(v) for v in orig_size]
        sx = target_w / float(actual_w)
        sy = target_h / float(actual_h)

        result = []
        for idx in range(min(len(boxes), len(scores), len(masks))):
            box = boxes[idx].astype(float).copy()
            box[[0, 2]] *= sx
            box[[1, 3]] *= sy
            mask = np.asarray(masks[idx]).astype(bool)
            if mask.shape != (target_h, target_w):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            result.append({
                "label": 0,
                "score": float(scores[idx]),
                "bbox_xyxy": box.tolist(),
                "mask": mask,
            })
        return result

    def predict_image(self, image_np, orig_size=None):
        from PIL import Image

        image_np = np.asarray(image_np)
        actual_h, actual_w = image_np.shape[:2]
        if orig_size is None:
            orig_size = (actual_w, actual_h)

        # Pass PIL deliberately.  Current SAM3 Sam3Processor records NumPy
        # dimensions via image.shape[-2:], which is wrong for HWC arrays.
        pil_image = Image.fromarray(image_np.astype(np.uint8, copy=False), mode="RGB")
        with self._autocast_context():
            state = self._processor.set_image(pil_image)
            state = self._processor.set_text_prompt(self.prompt, state)
        return self._state_to_records(
            state,
            actual_hw=(actual_h, actual_w),
            orig_size=orig_size,
        )
