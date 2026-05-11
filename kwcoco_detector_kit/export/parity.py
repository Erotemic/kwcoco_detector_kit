"""
torch ↔ ONNX parity guard.

Runs the same input through the torch checkpoint and the exported ONNX,
asserts the outputs agree to within a tolerance. Catches the class of
bug where the ONNX export ran but produces different outputs from the
torch model (op-level adapter divergence, missing custom-op handler,
wrong opset, etc.).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple


def check_parity(
    *,
    trainer,
    workdir: Path,
    onnx_fpath: Path,
    input_hw: Tuple[int, int],
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> dict:
    """Compare torch vs ONNX inference on a synthetic image.

    Returns a dict::

        {"ok": bool, "max_abs_diff_scores": float, "max_abs_diff_boxes": float,
         "max_abs_diff_labels": float, "rtol": ..., "atol": ...}
    """
    import numpy as np
    import onnxruntime as ort
    import torch

    workdir = Path(workdir)
    H, W = int(input_hw[0]), int(input_hw[1])

    # Build a deterministic dummy input that both backends can ingest.
    img = (np.random.RandomState(0).rand(1, 3, H, W) * 255).astype(np.float32) / 255.0
    sz = np.array([[W, H]], dtype=np.int64)

    # ONNX
    sess = ort.InferenceSession(str(onnx_fpath), providers=["CPUExecutionProvider"])
    onnx_outs = sess.run(None, {"images": img, "orig_target_sizes": sz})
    onnx_labels, onnx_boxes, onnx_scores = onnx_outs[:3]

    # Torch
    predictor = trainer.build_predictor(workdir, device="cpu")
    model = getattr(predictor, "_model", None)
    if model is None:
        raise RuntimeError("predictor doesn't expose ._model; can't check parity")
    with torch.no_grad():
        t_labels, t_boxes, t_scores = model(
            torch.from_numpy(img), torch.from_numpy(sz)
        )
    t_labels = t_labels.cpu().numpy()
    t_boxes = t_boxes.cpu().numpy()
    t_scores = t_scores.cpu().numpy()

    max_lab = float(np.max(np.abs(onnx_labels.astype(np.float64) - t_labels.astype(np.float64))))
    max_box = float(np.max(np.abs(onnx_boxes - t_boxes)))
    max_score = float(np.max(np.abs(onnx_scores - t_scores)))
    ok = bool(
        np.allclose(onnx_scores, t_scores, rtol=rtol, atol=atol)
        and np.allclose(onnx_boxes, t_boxes, rtol=rtol, atol=atol)
    )
    return {
        "ok": ok,
        "max_abs_diff_scores": max_score,
        "max_abs_diff_boxes": max_box,
        "max_abs_diff_labels": max_lab,
        "rtol": rtol,
        "atol": atol,
    }
