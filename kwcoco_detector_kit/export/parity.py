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

import scriptconfig as scfg


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


class ParityConfig(scfg.DataConfig):
    """Check torch ↔ ONNX output parity for a trained workdir."""

    workdir = scfg.Value(None, position=1, required=True,
                         help="trained workdir (must contain export/*.onnx and checkpoint)")
    rtol = scfg.Value(1e-3, help="relative tolerance for allclose check")
    atol = scfg.Value(1e-3, help="absolute tolerance for allclose check")

    @classmethod
    def main(cls, argv=1, **kwargs):
        import kwcoco_detector_kit.trainers  # noqa: F401 — register plugins
        from kwcoco_detector_kit.trainers._registry import get_trainer
        from kwcoco_detector_kit.export.onnx import _read_policy

        config = cls.cli(argv=argv, data=kwargs, strict=True)
        workdir = Path(str(config.workdir)).expanduser().resolve()
        policy = _read_policy(workdir)

        variant = policy.get("variant", "")
        trainer_name = variant.split("_")[0] if variant else "deimv2"
        trainer = get_trainer(trainer_name)

        H = int(policy.get("export_input_h", 640))
        W = int(policy.get("export_input_w", 640))

        export_dpath = workdir / "export"
        onnx_files = sorted(export_dpath.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"no .onnx found in {export_dpath}")
        onnx_fpath = onnx_files[0]

        result = check_parity(
            trainer=trainer,
            workdir=workdir,
            onnx_fpath=onnx_fpath,
            input_hw=(H, W),
            rtol=float(config.rtol),
            atol=float(config.atol),
        )
        status = "PASS" if result["ok"] else "FAIL"
        print(
            f"[parity] {status}  "
            f"scores Δ={result['max_abs_diff_scores']:.2e}  "
            f"boxes Δ={result['max_abs_diff_boxes']:.2e}  "
            f"labels Δ={result['max_abs_diff_labels']:.2e}"
        )
        if not result["ok"]:
            raise SystemExit(1)


def run(config):
    ParityConfig.main(argv=False, **{k: v for k, v in config.items()})


__cli__ = ParityConfig
