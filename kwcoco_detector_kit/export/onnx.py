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

import scriptconfig as scfg

from kwcoco_detector_kit.export.modelspec import write_modelspec


DEFAULT_OPSET = 18


def _embed_onnx_metadata(model, *, category_names, score_thresh, H, W):
    """Embed inference params into ONNX model.metadata_props in-place.

    Keyed as: category_names (comma-joined), score_thresh, input_hw ("H,W").
    OnnxPredictor reads these as a fallback when the .modelspec.json sidecar
    is absent via session.get_modelmeta().custom_metadata_map.
    """
    for key, value in [
        ("category_names", ",".join(category_names)),
        ("score_thresh", str(score_thresh)),
        ("input_hw", f"{H},{W}"),
    ]:
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value


def _read_policy(workdir: Path) -> dict:
    p = workdir / "policy.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _checkpoint_fingerprint(ckpt) -> Optional[dict]:
    """Content fingerprint of the source ``.pth`` for provenance.

    Returns ``{name, path, sha256, size_bytes}`` or ``None`` if the
    checkpoint can't be read. Hashing a few-hundred-MB checkpoint is a
    one-time export cost.
    """
    import hashlib
    try:
        ckpt = Path(ckpt)
        if not ckpt.is_file():
            return None
        h = hashlib.sha256()
        with open(ckpt, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return {
            "name": ckpt.name,
            "path": str(ckpt),
            "sha256": h.hexdigest(),
            "size_bytes": ckpt.stat().st_size,
        }
    except Exception:
        return None


def _modelspec_provenance(*, trainer, workdir: Path, ckpt=None,
                          category_names, imputed):
    """Assemble (category_names, source_checkpoint, provenance, imputed) for
    the modelspec. Never fabricates class names: if none are supplied, the
    absence is recorded as imputed/unknown rather than written as truth.
    """
    from kwcoco_detector_kit._provenance import provenance_dict
    if ckpt is None:
        try:
            ckpt = trainer.find_checkpoint(workdir)
        except Exception:
            ckpt = None
    fingerprint = _checkpoint_fingerprint(ckpt) if ckpt is not None else None
    prov = provenance_dict()
    names = [str(n).strip() for n in (category_names or []) if str(n).strip()]
    imp = {k: v for k, v in dict(imputed or {}).items() if v}
    if not names:
        imp.setdefault(
            "category_names",
            "no class names supplied to the exporter; ONNX class indices are "
            "undecodable until re-exported with --category-names",
        )
    return names, fingerprint, prov, imp


def export_onnx(
    *,
    trainer,
    workdir,
    input_hw: Tuple[int, int],
    out_fpath: Optional[Path] = None,
    opset: int = DEFAULT_OPSET,
    score_thresh: float = 0.30,
    category_names: Optional[Sequence[str]] = None,
    category_names_source: Optional[str] = None,
    imputed: Optional[dict] = None,
    force: bool = False,
) -> Path:
    """Dispatch to the trainer-appropriate ONNX exporter.

    mock_tiny exports in-process via torch.onnx.export.
    DEIMv2 exports via ``tools/deployment/export_onnx.py`` subprocess.
    Other trainer plugins should implement their own export path; this
    function falls back to the mock_tiny path if the trainer has the
    expected structure.

    ``category_names`` is no longer defaulted to a placeholder — passing
    ``None``/empty records the absence as imputed/unknown in the modelspec
    instead of fabricating class names. ``category_names_source`` is a note
    on where the names came from; ``imputed`` maps any inferred-not-derived
    metadata field to a reason string (see ``write_modelspec``).
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
            category_names_source=category_names_source,
            imputed=imputed,
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
        category_names_source=category_names_source,
        imputed=imputed,
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
    category_names: Optional[Sequence[str]],
    category_names_source: Optional[str] = None,
    imputed: Optional[dict] = None,
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

    names, fingerprint, prov, imp = _modelspec_provenance(
        trainer=trainer, workdir=workdir,
        category_names=category_names, imputed=imputed,
    )

    try:
        import onnx as _onnx
        _m = _onnx.load(str(out_fpath), load_external_data=True)
        _embed_onnx_metadata(_m, category_names=names,
                             score_thresh=score_thresh, H=H, W=W)
        _onnx.save(_m, str(out_fpath), save_as_external_data=False)
    except ImportError:
        pass

    write_modelspec(
        out_fpath,
        input_hw=(H, W),
        postprocess_score_thresh=float(score_thresh),
        variant=variant,
        category_names=names,
        candidate_kind=candidate_kind,
        model_id=policy.get("candidate_id"),
        category_names_source=category_names_source,
        source_checkpoint=fingerprint,
        provenance=prov,
        imputed=imp,
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
    category_names: Optional[Sequence[str]],
    category_names_source: Optional[str] = None,
    imputed: Optional[dict] = None,
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

    # Read policy early — needed for both the simplify decision and write_modelspec.
    policy = _read_policy(workdir)

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
    # onnxsim cannot simplify dinov3-based models: the RoPE embedding subgraph
    # references tensors that onnxsim's C extension can't resolve, producing
    # "Input .../rope_embed/... is undefined!". Other backbones (hgnetv2, etc.)
    # are fine with simplify. Skip it only for dinov3.
    _variant = policy.get("variant", "")
    if importlib.util.find_spec("onnxsim") is not None and "dinov3" not in _variant:
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
        repacked = False
        try:
            import onnx
            model = onnx.load(str(derived_onnx),
                              load_external_data=True)
            _embed_onnx_metadata(model, category_names=category_names,
                                 score_thresh=score_thresh, H=H, W=W)
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
            repacked = True
        except ImportError:
            # No onnx module — fall back to shutil.move below.
            pass
        except Exception:
            # onnx.load failed: malformed bytes (test fixtures), corrupt
            # export, or other parse issue. Fall back to shutil.move so
            # the kit's move-after-success contract still holds. If the
            # caller's downstream step needs valid bytes, it will surface
            # the real error there (and the test fixtures don't reach it).
            pass
        if not repacked:
            import shutil
            shutil.move(str(derived_onnx), str(out_fpath))
    elif not out_fpath.exists():
        raise FileNotFoundError(
            f"DEIMv2 export reported success but neither {derived_onnx} nor "
            f"{out_fpath} exist on disk."
        )

    names, fingerprint, prov, imp = _modelspec_provenance(
        trainer=trainer, workdir=workdir, ckpt=ckpt,
        category_names=category_names, imputed=imputed,
    )
    write_modelspec(
        out_fpath,
        input_hw=(H, W),
        postprocess_score_thresh=float(score_thresh),
        variant=policy.get("variant", "deimv2"),
        category_names=names,
        candidate_kind=policy.get("candidate_kind", "real"),
        model_id=policy.get("candidate_id"),
        category_names_source=category_names_source,
        source_checkpoint=fingerprint,
        provenance=prov,
        imputed=imp,
        extra_meta={"opset": int(opset)},
    )
    return out_fpath


class ExportOnnxConfig(scfg.DataConfig):
    """Export a trained checkpoint to ONNX.

    Reads variant and input size from the workdir's policy.json.
    category_names defaults to policy.json when present (written for
    workdirs generated after this fix); pass --category-names explicitly
    for older workdirs that lack it.
    """

    workdir = scfg.Value(None, position=1, required=True,
                         help="trained workdir (contains policy.json + checkpoint)")
    category_names = scfg.Value(
        None,
        help="comma-separated category names; defaults to policy.json when present",
    )
    category_names_source = scfg.Value(
        None,
        help=(
            "note recorded in the modelspec on where category_names came from "
            "(e.g. 'train_kwcoco', 'cli', 'imputed:class_schemes.yaml:pup_vs_nonpup'). "
            "Defaults to 'cli' or 'policy.json' based on resolution."
        ),
    )
    category_names_imputed = scfg.Value(
        False, isflag=True,
        help=(
            "mark category_names as imputed (inferred from a secondary artifact, "
            "not a clean data-driven source) so downstream systems distrust them"
        ),
    )
    force = scfg.Value(False, isflag=True, help="re-export even if .onnx already exists")
    score_thresh = scfg.Value(0.30, help="score threshold written into the modelspec")
    opset = scfg.Value(DEFAULT_OPSET, help="ONNX opset version")

    @classmethod
    def main(cls, argv=1, **kwargs):
        import kwcoco_detector_kit.trainers  # noqa: F401 — register plugins
        from kwcoco_detector_kit.trainers._registry import get_trainer

        config = cls.cli(argv=argv, data=kwargs, strict=True)
        workdir = Path(str(config.workdir)).expanduser().resolve()
        policy = _read_policy(workdir)

        # Infer trainer from variant prefix (e.g. "deimv2_dinov3_x" → "deimv2").
        variant = policy.get("variant", "")
        trainer_name = variant.split("_")[0] if variant else "deimv2"
        trainer = get_trainer(trainer_name)

        H = int(policy.get("export_input_h", 640))
        W = int(policy.get("export_input_w", 640))

        # Resolve category_names: CLI arg takes precedence, then policy.json.
        raw = config.category_names
        names_source = config.category_names_source
        if raw is None:
            names_from_policy = policy.get("category_names") or []
            if not names_from_policy:
                raise ValueError(
                    "--category-names is required: policy.json in this workdir "
                    "predates the category_names fix. "
                    "Pass e.g. --category-names pup,nonpup_sealion"
                )
            category_names = list(names_from_policy)
            names_source = names_source or "policy.json"
        elif isinstance(raw, (list, tuple)):
            category_names = [str(n).strip() for n in raw if str(n).strip()]
            names_source = names_source or "cli"
        else:
            category_names = [s.strip() for s in str(raw).split(",") if s.strip()]
            names_source = names_source or "cli"

        imputed = None
        if bool(config.category_names_imputed):
            imputed = {"category_names": f"imputed via {names_source}"}

        out = export_onnx(
            trainer=trainer,
            workdir=workdir,
            input_hw=(H, W),
            category_names=category_names,
            category_names_source=names_source,
            imputed=imputed,
            score_thresh=float(config.score_thresh),
            opset=int(config.opset),
            force=bool(config.force),
        )
        print(f"[export-onnx] wrote {out}")


def run(config):
    ExportOnnxConfig.main(argv=False, **{k: v for k, v in config.items()})


__cli__ = ExportOnnxConfig
