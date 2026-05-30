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
from typing import Optional, Sequence, Tuple

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
    category_names: Sequence[str] = ("widget",),
    force: bool = False,
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
            category_names=category_names,
            force=force,
        )
    # Default: torch.onnx.export against the predictor's underlying model.
    return _export_inproc(
        trainer=trainer,
        workdir=workdir,
        input_hw=input_hw,
        out_fpath=out_fpath,
        opset=opset,
        score_thresh=score_thresh,
        category_names=category_names,
        force=force,
    )


def _export_inproc(
    *,
    trainer,
    workdir: Path,
    input_hw: Tuple[int, int],
    out_fpath: Optional[Path],
    opset: int,
    score_thresh: float,
    category_names: Sequence[str],
    force: bool,
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
    if out_fpath.exists() and out_fpath.stat().st_size >= 262144 and not bool(force):
        print(f"  reusing existing ONNX export: {out_fpath}")
        return out_fpath

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
        category_names=category_names,
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
    category_names: Sequence[str],
    force: bool,
) -> Path:
    """Subprocess DEIMv2's ``tools/deployment/export_onnx.py``.

    Falls back to recovering the .onnx if the post-export `--simplify`
    step crashed (failure #10).
    """
    from kwcoco_detector_kit.trainers.deimv2 import _resolve_deimv2_repo
    repo = _resolve_deimv2_repo()
    if not repo:
        raise EnvironmentError(
            "DEIMv2 ONNX export needs a DEIMv2 checkout. Either set "
            "$KCD_DEIMV2_REPO_DPATH or run `git submodule update --init "
            "tpl/DEIMv2` from the kit's repo root."
        )
    export_script = repo / "tools" / "deployment" / "export_onnx.py"
    if not export_script.exists():
        raise FileNotFoundError(export_script)

    H, W = int(input_hw[0]), int(input_hw[1])
    export_dpath = workdir / "export"
    export_dpath.mkdir(parents=True, exist_ok=True)
    out_fpath = out_fpath or (export_dpath / f"deimv2_h{H}_w{W}.onnx")
    if out_fpath.exists() and out_fpath.stat().st_size >= 262144 and not bool(force):
        print(f"  reusing existing ONNX export: {out_fpath}")
        return out_fpath
    if out_fpath.exists() and out_fpath.stat().st_size < 262144:
        print(
            f"  existing ONNX export is suspiciously small "
            f"({out_fpath.stat().st_size} bytes); re-exporting"
        )

    ckpt = trainer.find_checkpoint(workdir)
    cfg = workdir / "generated_configs" / "train.yml"

    # DEIMv2's tools/deployment/export_onnx.py has no `-o`/`--output` flag —
    # it derives the output path from `--resume`:
    #     output_file = args.resume.replace('.pth', '.onnx')
    # We let it write there, then move to the kit's canonical
    # `<workdir>/export/<name>.onnx` slot. See lesson #27.
    args = [
        sys.executable, str(export_script),
        "-c", str(cfg),
        "-r", str(ckpt),
        "--check",
        "--opset", str(int(opset)),
    ]
    if importlib.util.find_spec("onnxsim") is not None:
        args.append("--simplify")

    derived_onnx = Path(str(ckpt).replace(".pth", ".onnx"))
    try:
        subprocess.run(args, check=True, cwd=str(repo))
    except subprocess.CalledProcessError as ex:
        # Recover the unsimplified .onnx if the subprocess crashed during
        # --simplify (failure #10). Upstream may have written the .onnx
        # to either the derived path (next to the checkpoint) or the kit's
        # intended path (if we ever land an upstream patch that honors -o).
        if not (derived_onnx.exists() or out_fpath.exists()):
            raise
        print(
            f"[export.onnx] DEIMv2 exporter exited {ex.returncode} but "
            f"a .onnx artifact exists — recovering unsimplified output."
        )

    # Move the upstream-derived artifact to the kit's canonical path.
    # torch >= 2.12 saves model weights as a separate ``<file>.onnx.data``
    # sidecar by default (even for small models), so a naive shutil.move
    # of just the .onnx leaves the runtime unable to find the weights:
    #   ONNXRuntimeError: External data path does not exist: ...
    # Repack via onnx.load + onnx.save with save_as_external_data=False so
    # the destination is a single self-contained file — works on any torch
    # version, sidesteps "where does the .data file go" entirely.
    if derived_onnx.exists() and derived_onnx != out_fpath:
        out_fpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            import onnx
            model = onnx.load(str(derived_onnx),
                              load_external_data=True)
            onnx.save(model, str(out_fpath),
                      save_as_external_data=False)
            derived_onnx.unlink()
            # Best-effort cleanup of the sidecar(s) next to derived path.
            for sidecar in derived_onnx.parent.glob(
                    f"{derived_onnx.name}*.data"):
                sidecar.unlink()
            # Also catch the bare `<name>.onnx.data` case.
            sidecar = derived_onnx.with_suffix(".onnx.data")
            if sidecar.exists():
                sidecar.unlink()
        except ImportError:
            # No onnx module — fall back to shutil.move, which works on
            # torch < 2.12 where the .onnx is self-contained.
            import shutil
            shutil.move(str(derived_onnx), str(out_fpath))
    elif not out_fpath.exists():
        raise FileNotFoundError(
            f"DEIMv2 export reported success but neither {derived_onnx} nor "
            f"{out_fpath} exist on disk."
        )

    policy = _read_policy(workdir)
    write_modelspec(
        out_fpath,
        input_hw=(H, W),
        postprocess_score_thresh=float(score_thresh),
        variant=policy.get("variant", "deimv2"),
        category_names=category_names,
        candidate_kind=policy.get("candidate_kind", "real"),
        model_id=policy.get("candidate_id"),
        extra_meta={"opset": int(opset)},
    )
    return out_fpath
