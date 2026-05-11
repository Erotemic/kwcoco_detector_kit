"""DEIMv2 launch must always go through torch.distributed.run (lesson #24).

Bare `python train.py` works on torch <= 2.9 but breaks on torch 2.10+ when
DEIMv2's hgnetv2 backbone calls `torch.distributed.get_rank()` without an
initialized process group. The kit's launch() unconditionally prepends
``-m torch.distributed.run --nproc_per_node N``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest


def test_launch_command_is_torchrun_even_for_single_gpu(tmp_path, monkeypatch):
    """Capture the subprocess args to verify torchrun is always used."""
    from kwcoco_detector_kit.trainers._registry import get_trainer

    trainer = get_trainer("deimv2")

    # Fake repo with a train.py marker so launch() doesn't raise FileNotFound.
    fake_repo = tmp_path / "fake_deimv2"
    fake_repo.mkdir()
    (fake_repo / "train.py").write_text("# stub\n")
    monkeypatch.setenv("KCD_DEIMV2_REPO_DPATH", str(fake_repo))

    # Fake generated cfg path so the parent.parent => workdir layout works.
    workdir = tmp_path / "wd"
    cfg_dir = workdir / "generated_configs"
    cfg_dir.mkdir(parents=True)
    cfg_fpath = cfg_dir / "train.yml"
    cfg_fpath.write_text("# stub\n")

    captured = {}

    def _fake_run(args, **kwargs):
        captured["args"] = list(args)
        captured["kwargs"] = kwargs
        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr(
        "kwcoco_detector_kit.trainers.deimv2.subprocess.run",
        _fake_run,
    )

    # Single-GPU launch — must still go through torchrun.
    trainer.launch(cfg_fpath, num_gpus=1, distributed=False)

    args = captured["args"]
    assert sys.executable in args[0], f"first arg should be the python exe; got {args!r}"
    assert "-m" in args, f"launch should use -m torch.distributed.run; got {args!r}"
    assert "torch.distributed.run" in args, f"launch should call torch.distributed.run; got {args!r}"
    assert "--nproc_per_node" in args
    nproc_idx = args.index("--nproc_per_node")
    assert args[nproc_idx + 1] == "1", f"nproc_per_node should be 1; got {args[nproc_idx + 1]!r}"
    assert str(cfg_fpath) in args, f"config path should be in args; got {args!r}"


def test_launch_command_scales_nproc_per_node_with_num_gpus(tmp_path, monkeypatch):
    from kwcoco_detector_kit.trainers._registry import get_trainer

    trainer = get_trainer("deimv2")
    fake_repo = tmp_path / "fake_deimv2"
    fake_repo.mkdir()
    (fake_repo / "train.py").write_text("# stub\n")
    monkeypatch.setenv("KCD_DEIMV2_REPO_DPATH", str(fake_repo))

    workdir = tmp_path / "wd"
    cfg_dir = workdir / "generated_configs"
    cfg_dir.mkdir(parents=True)
    cfg_fpath = cfg_dir / "train.yml"
    cfg_fpath.write_text("# stub\n")

    captured = {}

    def _fake_run(args, **kwargs):
        captured["args"] = list(args)
        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr(
        "kwcoco_detector_kit.trainers.deimv2.subprocess.run",
        _fake_run,
    )
    trainer.launch(cfg_fpath, num_gpus=4, distributed=True)
    args = captured["args"]
    nproc_idx = args.index("--nproc_per_node")
    assert args[nproc_idx + 1] == "4"
