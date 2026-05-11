"""
ONNX export — covers the mock_tiny in-tree path and the DEIMv2 subprocess
path.

Failures #9 + #10 are handled here:

- ``torch.onnx.export`` on torch >= 2.5 imports ``onnxscript`` at
  function-call time. ``pyproject.toml`` makes onnxscript a hard dep.
- DEIMv2's exporter calls ``onnxsim`` as a post-export simplify step;
  if missing or buggy, the simplify step crashes but the unsimplified
  ``.onnx`` is on disk. We detect onnxsim availability before passing
  ``--simplify`` and recover the artifact if the subprocess crashed.
- Default opset 18 (torch >= 2.5's dynamo exporter targets opset 18;
  earlier opsets have no adapter for the Pad op).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from kwcoco_detector_kit.export.modelspec import write_modelspec


DEFAULT_OPSET = 18


def _read_policy(workdir: Path) -> dict:
    p = workdir / "policy.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def export_onnx(
    *,
    trainer,
    workdir,
    input_hw: Tuple[int, int],
    out_fpath: Optional[Path] = None,
    opset: int = DEFAULT_OPSET,
    score_thresh: float = 0.30,
    category_name: str = "widget",
) -> Path:
    """Dispatch to the trainer-appropriate ONNX exporter.

    mock_tiny exports in-process via torch.onnx.export.
    DEIMv2 exports via ``tools/deployment/export_onnx.py`` subprocess.
    Other trainer plugins should implement their own export path; this
    function falls back to the mock_tiny path if the trainer has the
    expected structure.
    """
    workdir = Path(workdir)
    if trainer.name == "deimv2":
        return _export_deimv2(
            trainer=trainer,
            workdir=workdir,
            input_hw=input_hw,
            out_fpath=out_fpath,
            opset=opset,
            score_thresh=score_thresh,
            category_name=category_name,
        )
    # Default: torch.onnx.export against the predictor's underlying model.
    return _export_inproc(
        trainer=trainer,
        workdir=workdir,
        input_hw=input_hw,
        out_fpath=out_fpath,
        opset=opset,
        score_thresh=score_thresh,
        category_name=category_name,
    )


def _export_inproc(
    *,
    trainer,
    workdir: Path,
    input_hw: Tuple[int, int],
    out_fpath: Optional[Path],
    opset: int,
    score_thresh: float,
    category_name: str,
) -> Path:
    """In-process torch.onnx.export for trainer plugins (mock_tiny etc.).

    Uses the trainer's predictor to find the loaded model, then traces
    the (image, orig_size) -> (labels, boxes, scores) signature that
    mirrors DEIMv2's deploy output format.
    """
    import torch

    H, W = int(input_hw[0]), int(input_hw[1])
    export_dpath = workdir / "export"
    export_dpath.mkdir(parents=True, exist_ok=True)

    policy = _read_policy(workdir)
    variant = policy.get("variant", trainer.name)
    candidate_kind = policy.get("candidate_kind", "")

    out_fpath = out_fpath or (export_dpath / f"{trainer.name}_h{H}_w{W}.onnx")

    # mock_tiny has the load_state_dict + _build_model utilities exposed
    # but the simplest path is to ask the predictor for its wrapped model.
    predictor = trainer.build_predictor(workdir, device="cpu")
    model = getattr(predictor, "_model", None)
    if model is None:
        raise RuntimeError(
            f"trainer {trainer.name!r} predictor doesn't expose ._model; "
            "in-process ONNX export requires it."
        )

    dummy_img = torch.zeros(1, 3, H, W, dtype=torch.float32)
    dummy_size = torch.tensor([[W, H]], dtype=torch.int64)

    torch.onnx.export(
        model,
        (dummy_img, dummy_size),
        str(out_fpath),
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        opset_version=int(opset),
        do_constant_folding=True,
        dynamic_axes={"images": {0: "N"}, "orig_target_sizes": {0: "N"}},
    )

    write_modelspec(
        out_fpath,
        input_hw=(H, W),
        postprocess_score_thresh=float(score_thresh),
        variant=variant,
        category_name=category_name,
        candidate_kind=candidate_kind,
        model_id=policy.get("candidate_id"),
        extra_meta={"opset": int(opset)},
    )
    return out_fpath


def _export_deimv2(
    *,
    trainer,
    workdir: Path,
    input_hw: Tuple[int, int],
    out_fpath: Optional[Path],
    opset: int,
    score_thresh: float,
    category_name: str,
) -> Path:
    """Subprocess DEIMv2's ``tools/deployment/export_onnx.py``.

    Falls back to recovering the .onnx if the post-export `--simplify`
    step crashed (failure #10).
    """
    repo = os.environ.get("KCD_DEIMV2_REPO_DPATH")
    if not repo:
        raise EnvironmentError(
            "DEIMv2 ONNX export needs $KCD_DEIMV2_REPO_DPATH."
        )
    repo = Path(repo).expanduser().resolve()
    export_script = repo / "tools" / "deployment" / "export_onnx.py"
    if not export_script.exists():
        raise FileNotFoundError(export_script)

    H, W = int(input_hw[0]), int(input_hw[1])
    export_dpath = workdir / "export"
    export_dpath.mkdir(parents=True, exist_ok=True)
    out_fpath = out_fpath or (export_dpath / f"deimv2_h{H}_w{W}.onnx")

    ckpt = trainer.find_checkpoint(workdir)
    cfg = workdir / "generated_configs" / "train.yml"

    args = [
        sys.executable, str(export_script),
        "-c", str(cfg),
        "-r", str(ckpt),
        "-o", str(out_fpath),
        "--check",
        "--opset", str(int(opset)),
    ]
    if importlib.util.find_spec("onnxsim") is not None:
        args.append("--simplify")

    try:
        subprocess.run(args, check=True, cwd=str(repo))
    except subprocess.CalledProcessError as ex:
        # Recover the unsimplified .onnx if the subprocess crashed
        # during --simplify (failure #10).
        if not out_fpath.exists():
            raise
        print(
            f"[export.onnx] DEIMv2 exporter exited {ex.returncode} but "
            f"{out_fpath} exists — recovering unsimplified artifact."
        )

    policy = _read_policy(workdir)
    write_modelspec(
        out_fpath,
        input_hw=(H, W),
        postprocess_score_thresh=float(score_thresh),
        variant=policy.get("variant", "deimv2"),
        category_name=category_name,
        candidate_kind=policy.get("candidate_kind", "real"),
        model_id=policy.get("candidate_id"),
        extra_meta={"opset": int(opset)},
    )
    return out_fpath
