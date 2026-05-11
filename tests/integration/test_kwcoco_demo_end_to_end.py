"""End-to-end smoke: synth -> tile -> train mock_tiny -> export -> eval -> bench -> manifest.

Mirrors what `examples/kwcoco_demo/run_smoke.sh` runs, but as a single
pytest. Acceptance: <30 s on a 1-CPU laptop, produces a manifest row
with `eligibility_class=HOST_PROMISING` of `candidate_kind=smoke`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.mark.requires_torch
def test_kwcoco_demo_end_to_end_smoke(synthetic_kwcoco_factory, tmp_path, monkeypatch):
    monkeypatch.setenv("KCD_ROOT", str(tmp_path))

    raw = synthetic_kwcoco_factory("raw", num_images=4, image_size=(128, 128))

    # 1. Tile (multiscale)
    from kwcoco_detector_kit.data.tile import TileConfig, run as tile_run

    tiles_fpath = tmp_path / "tiles.kwcoco.zip"
    tile_cfg = TileConfig.cli(
        argv=False,
        data={
            "src": str(raw), "dst": str(tiles_fpath),
            "mode": "multiscale", "category_name": "widget",
            "tile_size": 64, "source_scales": "1.0",
            "stride_frac": 0.5, "min_gt_area_frac": 0.001,
            "min_source_scale_long_side": 32,
            "keep_negative": True, "progress": False,
        },
    )
    tile_run(tile_cfg)
    assert tiles_fpath.exists()

    # 2. Sweep + eligibility (mock_tiny single cell)
    from kwcoco_detector_kit.orchestration.pareto_sweep import (
        SweepConfig, run as sweep_run,
    )
    from kwcoco_detector_kit.orchestration.eligibility import (
        EligibilityConfig, run as elig_run, HOST_PROMISING,
    )

    sweep_cfg = SweepConfig.cli(
        argv=False,
        data={
            "train_kwcoco": str(tiles_fpath),
            "vali_kwcoco": str(tiles_fpath),
            "test_kwcoco": str(raw),
            "kcd_root": str(tmp_path),
            "trainer": "mock_tiny", "variant": "mock_tiny",
            "input_hw": [64, 64], "train_policy": "fixed",
            "num_epochs": 1, "batch_size": 2, "val_batch_size": 2,
            "scale_tier": "S", "category_name": "widget",
            "lr": 1e-2, "backbone_lr": 1e-2, "use_amp": False,
        },
    )
    sweep_run(sweep_cfg)

    elig_cfg = EligibilityConfig.cli(
        argv=False,
        data={
            "auto": True, "kcd_root": str(tmp_path),
            "out": str(tmp_path / "manifest.tsv"),
            "out_json": str(tmp_path / "manifest.json"),
            "max_desktop_ms": 5000.0,
            "include_smoke_models": True,
            "allow_missing_desktop_bench": False,
            "print_winner": False,
        },
    )
    rows = elig_run(elig_cfg)

    assert (tmp_path / "manifest.tsv").exists()
    assert (tmp_path / "manifest.json").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest, "manifest must have at least one row"

    # At least one HOST_PROMISING smoke candidate.
    promising = [
        r for r in manifest
        if r["eligibility_class"] == HOST_PROMISING and r["candidate_kind"] == "smoke"
    ]
    assert promising, (
        f"expected HOST_PROMISING smoke candidate; got "
        f"{[(r['candidate_id'], r['eligibility_class'], r['candidate_kind']) for r in manifest]}"
    )
