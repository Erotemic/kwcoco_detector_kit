"""DEIMv2 ONNX export args must match upstream's CLI surface (lesson #27).

`tools/deployment/export_onnx.py` accepts only `--config`/`-c`, `--resume`/`-r`,
`--opset`, `--check`, `--simplify`. No `-o`/`--output`. The kit's wrapper
drops `-o` and moves the derived artifact (``<ckpt>.replace('.pth', '.onnx')``)
to the canonical `<workdir>/export/<name>.onnx` slot afterwards.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest


def test_export_args_do_not_include_o_flag(tmp_path, monkeypatch):
    """Capture the subprocess args to verify -o is not passed."""
    from kwcoco_detector_kit.trainers._registry import get_trainer
    from kwcoco_detector_kit.export.onnx import export_onnx

    trainer = get_trainer("deimv2")

    # Fake DEIMv2 repo with both train.py + tools/deployment/export_onnx.py.
    fake_repo = tmp_path / "fake_deimv2"
    (fake_repo / "tools" / "deployment").mkdir(parents=True)
    (fake_repo / "train.py").write_text("# stub\n")
    (fake_repo / "tools" / "deployment" / "export_onnx.py").write_text("# stub\n")
    monkeypatch.setenv("KCD_DEIMV2_REPO_DPATH", str(fake_repo))

    # Fake workdir with a checkpoint where `find_checkpoint` will resolve.
    workdir = tmp_path / "wd"
    workdir.mkdir()
    ckpt_fpath = workdir / "best_stg2.pth"
    ckpt_fpath.write_bytes(b"x")
    cfg_dpath = workdir / "generated_configs"
    cfg_dpath.mkdir()
    (cfg_dpath / "train.yml").write_text("# stub\n")
    (workdir / "policy.json").write_text("{}")

    import subprocess as _sp
    _real_run = _sp.run
    captured = {}

    def _fake_run(args, **kwargs):
        # Only intercept the DEIMv2 export subprocess; defer auxiliary calls
        # (e.g. the git provenance probe) to the real subprocess.run.
        if "export_onnx.py" not in " ".join(map(str, args)):
            return _real_run(args, **kwargs)
        captured["args"] = list(args)
        # Materialise the "derived" .onnx so the kit's move-after-success
        # path has a file to move.
        derived = Path(str(ckpt_fpath).replace(".pth", ".onnx"))
        derived.write_bytes(b"fake_onnx_bytes")
        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr(
        "kwcoco_detector_kit.export.onnx.subprocess.run",
        _fake_run,
    )

    out_fpath = export_onnx(
        trainer=trainer, workdir=workdir, input_hw=(256, 256),
    )

    args = captured["args"]
    assert "-o" not in args, (
        f"DEIMv2 export must not pass -o; upstream rejects it. Got args: {args!r}"
    )
    assert "--output" not in args, f"likewise: no --output. Got args: {args!r}"
    # -c, -r, --check, --opset, --simplify (when onnxsim present) only.
    assert "-c" in args
    assert "-r" in args
    assert "--check" in args
    assert "--opset" in args
    # The kit-canonical artifact must exist after export.
    assert Path(out_fpath).exists(), f"out_fpath {out_fpath} must exist after export"


def test_derived_onnx_moves_to_canonical_path(tmp_path, monkeypatch):
    """When upstream writes to <ckpt>.replace('.pth', '.onnx'), the kit
    moves it to <workdir>/export/<name>.onnx."""
    from kwcoco_detector_kit.trainers._registry import get_trainer
    from kwcoco_detector_kit.export.onnx import export_onnx

    trainer = get_trainer("deimv2")

    fake_repo = tmp_path / "fake_deimv2"
    (fake_repo / "tools" / "deployment").mkdir(parents=True)
    (fake_repo / "train.py").write_text("# stub\n")
    (fake_repo / "tools" / "deployment" / "export_onnx.py").write_text("# stub\n")
    monkeypatch.setenv("KCD_DEIMV2_REPO_DPATH", str(fake_repo))

    workdir = tmp_path / "wd"
    workdir.mkdir()
    ckpt_fpath = workdir / "best_stg1.pth"  # exercise the fallback name too
    ckpt_fpath.write_bytes(b"x")
    cfg_dpath = workdir / "generated_configs"
    cfg_dpath.mkdir()
    (cfg_dpath / "train.yml").write_text("# stub\n")
    (workdir / "policy.json").write_text("{}")

    import subprocess as _sp
    _real_run = _sp.run

    def _fake_run(args, **kwargs):
        # Only intercept the DEIMv2 export subprocess; defer auxiliary calls
        # (e.g. the git provenance probe) to the real subprocess.run.
        if "export_onnx.py" not in " ".join(map(str, args)):
            return _real_run(args, **kwargs)
        # Upstream writes <ckpt>.replace('.pth', '.onnx').
        derived = Path(str(ckpt_fpath).replace(".pth", ".onnx"))
        derived.write_bytes(b"fake_onnx_bytes")
        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr(
        "kwcoco_detector_kit.export.onnx.subprocess.run",
        _fake_run,
    )

    out_fpath = export_onnx(
        trainer=trainer, workdir=workdir, input_hw=(256, 256),
    )

    # Derived path must have moved (not exist anymore).
    derived = workdir / "best_stg1.onnx"
    assert not derived.exists(), (
        f"derived path {derived} should have been moved, not left behind"
    )
    # Canonical kit path must exist.
    assert Path(out_fpath).exists()
    # And it must live under workdir/export/.
    assert "export" in Path(out_fpath).parts


def test_recovery_works_when_simplify_crashes(tmp_path, monkeypatch):
    """When the upstream subprocess crashes mid --simplify but the .onnx is on
    disk, the kit recovers (failure #10 pattern preserved)."""
    import subprocess as _sp
    from kwcoco_detector_kit.trainers._registry import get_trainer
    from kwcoco_detector_kit.export.onnx import export_onnx

    trainer = get_trainer("deimv2")

    fake_repo = tmp_path / "fake_deimv2"
    (fake_repo / "tools" / "deployment").mkdir(parents=True)
    (fake_repo / "train.py").write_text("# stub\n")
    (fake_repo / "tools" / "deployment" / "export_onnx.py").write_text("# stub\n")
    monkeypatch.setenv("KCD_DEIMV2_REPO_DPATH", str(fake_repo))

    workdir = tmp_path / "wd"
    workdir.mkdir()
    ckpt_fpath = workdir / "best_stg2.pth"
    ckpt_fpath.write_bytes(b"x")
    cfg_dpath = workdir / "generated_configs"
    cfg_dpath.mkdir()
    (cfg_dpath / "train.yml").write_text("# stub\n")
    (workdir / "policy.json").write_text("{}")

    _real_run = _sp.run

    def _fake_run_crashes_after_writing_onnx(args, **kwargs):
        # Only intercept the DEIMv2 export subprocess; defer auxiliary calls
        # (e.g. the git provenance probe) to the real subprocess.run.
        if "export_onnx.py" not in " ".join(map(str, args)):
            return _real_run(args, **kwargs)
        # Upstream writes the .onnx before --simplify, then --simplify crashes.
        derived = Path(str(ckpt_fpath).replace(".pth", ".onnx"))
        derived.write_bytes(b"unsimplified_onnx")
        raise _sp.CalledProcessError(returncode=1, cmd=args)

    monkeypatch.setattr(
        "kwcoco_detector_kit.export.onnx.subprocess.run",
        _fake_run_crashes_after_writing_onnx,
    )

    out_fpath = export_onnx(
        trainer=trainer, workdir=workdir, input_hw=(256, 256),
    )
    assert Path(out_fpath).exists()
    assert Path(out_fpath).read_bytes() == b"unsimplified_onnx"
