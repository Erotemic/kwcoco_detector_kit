"""
Top-level scriptconfig CLI.

Subcommands::

  demo-data       Generate a synthetic kwcoco bundle (vidshapes-style).
  tile            kwcoco -> tile-augmented kwcoco (3 modes).
  merge           positive + negative -> training kwcoco for one round.
  mine            offline hard-negative mining.
  train           Run one trainer-plugin cell.
  sweep           Run a Pareto sweep over a matrix of cells.
  round-loop      Round-based hard-negative-mining loop.
  export-onnx     Export a trained checkpoint to ONNX.
  parity          Check torch <-> ONNX parity on the exported model.
  eval            Run kwcoco eval against a trained checkpoint.
  bench           Run desktop ONNX bench.
  package-build   Build a portable package from a trained workdir.
  predict         Run packaged detector inference over kwcoco data.
  export-labelme  Export prediction kwcoco to LabelMe JSON sidecars.
  segmenter-train Fine-tune a SAM2 segmenter from kwcoco splits.
  pseudo-label    Generate pseudo-label kwcoco from a teacher model.
  manifest        Aggregate sweep outputs into an eligibility manifest.
  check-env       Probe transitive runtime deps; --install to fix.
  config-init     Write editable environment + dataset YAML configs.
  config-inspect  Show config values with introspected suggestions.
  config-edit     Modify config YAML via a text UI or --set overrides.
  run-all         The kwcoco_demo end-to-end smoke driver.
"""
from __future__ import annotations

import sys
from pathlib import Path

import scriptconfig as scfg

# Force trainer-plugin import-side registration.
import kwcoco_detector_kit.trainers  # noqa: F401


# Each subcommand wraps a module's __cli__ DataConfig + run().


class ModalKit(scfg.ModalCLI):
    """kwcoco-detector-kit — domain-agnostic detector training on kwcoco."""


@ModalKit.register
class DemoDataCLI(scfg.DataConfig):
    """Generate a synthetic kwcoco bundle for smoke tests."""

    __command__ = "demo-data"

    dst = scfg.Value(None, position=1, required=True, help="output kwcoco path")
    num_images = scfg.Value(16, help="number of images to synthesize")
    num_categories = scfg.Value(1, help="number of categories")
    image_size = scfg.Value([256, 256], help="image size [H, W]")
    seed = scfg.Value(0)
    category_name = scfg.Value("widget", help="single-category convenience name")

    @classmethod
    def main(cls, argv=1, **kwargs):
        import numpy as np
        import kwcoco
        import kwimage

        config = cls.cli(argv=argv, data=kwargs, strict=True)
        dst = Path(str(config.dst)).expanduser().resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        bundle_dpath = dst.parent
        asset_dpath = bundle_dpath / (dst.stem.replace(".kwcoco", "") + "_assets")
        asset_dpath.mkdir(parents=True, exist_ok=True)

        rng = np.random.RandomState(int(config.seed))
        H, W = int(config.image_size[0]), int(config.image_size[1])
        dset = kwcoco.CocoDataset()
        dset.fpath = str(dst)
        cid = dset.add_category(name=str(config.category_name))

        for k in range(int(config.num_images)):
            img = (rng.rand(H, W, 3) * 255).astype(np.uint8)
            bw = int(rng.randint(20, max(21, W // 4)))
            bh = int(rng.randint(20, max(21, H // 4)))
            bx = int(rng.randint(0, W - bw))
            by = int(rng.randint(0, H - bh))
            img[by:by + bh, bx:bx + bw] = (255, 50, 50)
            fpath = asset_dpath / f"demo_{k:04d}.jpg"
            kwimage.imwrite(str(fpath), img)
            gid = dset.add_image(
                file_name=str(fpath.relative_to(bundle_dpath)),
                width=W, height=H, name=f"demo_{k:04d}",
            )
            dset.add_annotation(
                image_id=gid, category_id=cid,
                bbox=[float(bx), float(by), float(bw), float(bh)],
                area=float(bw * bh), iscrowd=0,
            )
        dset.dump()
        print(f"wrote {dset.fpath} ({int(config.num_images)} images)")


def _register_module(name, module):
    """Wrap a module's __cli__ DataConfig as a ModalKit subcommand."""
    inner_cls = module.__cli__
    bases = (inner_cls,)
    # Copy attributes and inject the __command__ field.
    attrs = {"__command__": name, "__doc__": inner_cls.__doc__}
    new_cls = type(inner_cls.__name__, bases, attrs)
    ModalKit.register(new_cls)


# Register each module's CLI under its kebab-case command name.
def _register_subcommands():
    import kwcoco_detector_kit.data.tile as _tile
    import kwcoco_detector_kit.data.merge as _merge
    import kwcoco_detector_kit.data.mine as _mine
    import kwcoco_detector_kit.data.tile_store as _tile_store
    import kwcoco_detector_kit.data.stats as _stats
    import kwcoco_detector_kit.orchestration.pareto_sweep as _sweep
    import kwcoco_detector_kit.orchestration.round_loop as _round
    import kwcoco_detector_kit.orchestration.eligibility as _elig
    import kwcoco_detector_kit.orchestration.setup_audit as _audit
    import kwcoco_detector_kit.configs as _configs
    import kwcoco_detector_kit.export.package as _package
    import kwcoco_detector_kit.export.labelme as _labelme
    import kwcoco_detector_kit.trainers.sam2 as _sam2
    import kwcoco_detector_kit.data.distill as _distill
    import kwcoco_detector_kit.predict as _predict

    _register_module("tile", _tile)
    _register_module("merge", _merge)
    _register_module("mine", _mine)
    _register_module("convert-store", _tile_store)   # Phase 3
    _register_module("stats", _stats)                # Phase 3
    _register_module("sweep", _sweep)
    _register_module("round-loop", _round)
    _register_module("manifest", _elig)
    _register_module("check-env", _audit)
    _register_module("package-build", _package)
    _register_module("predict", _predict)
    _register_module("export-labelme", _labelme)
    _register_module("segmenter-train", _sam2)
    _register_module("pseudo-label", _distill)
    _register_module("config-init", type("ConfigInitModule", (), {"__cli__": _configs.ConfigInitConfig}))
    _register_module("config-inspect", type("ConfigInspectModule", (), {"__cli__": _configs.ConfigInspectConfig}))
    _register_module("config-edit", type("ConfigEditModule", (), {"__cli__": _configs.ConfigEditConfig}))


_register_subcommands()


@ModalKit.register
class RunAllCLI(scfg.DataConfig):
    """End-to-end smoke driver — synth data, tile, train, export, eval, bench, manifest.

    Designed to run on a 1-CPU laptop in <90 s using ``trainer=mock_tiny``.
    """

    __command__ = "run-all"

    train_kwcoco = scfg.Value(None, required=True, help="training kwcoco")
    vali_kwcoco = scfg.Value(None, help="validation kwcoco (defaults to train_kwcoco)")
    test_kwcoco = scfg.Value(None, help="test kwcoco (defaults to train_kwcoco)")
    workdir = scfg.Value(None, required=True, help="workspace root (sets KCD_ROOT)")
    category_name = scfg.Value("widget")
    trainer = scfg.Value("mock_tiny")
    variant = scfg.Value("mock_tiny")
    tier = scfg.Value("S")
    input_hw = scfg.Value([256, 256])
    num_epochs = scfg.Value(2)
    batch_size = scfg.Value(2)

    @classmethod
    def main(cls, argv=1, **kwargs):
        import os
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        workdir = Path(str(config.workdir)).expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        os.environ["KCD_ROOT"] = str(workdir)

        vali = config.vali_kwcoco or config.train_kwcoco
        test = config.test_kwcoco or config.train_kwcoco

        from kwcoco_detector_kit.orchestration.pareto_sweep import SweepConfig, run as sweep_run
        from kwcoco_detector_kit.orchestration.eligibility import EligibilityConfig, run as elig_run

        sweep_cfg = SweepConfig.cli(
            argv=False,
            data={
                "train_kwcoco": str(config.train_kwcoco),
                "vali_kwcoco": str(vali),
                "test_kwcoco": str(test),
                "kcd_root": str(workdir),
                "trainer": str(config.trainer),
                "variant": str(config.variant),
                "input_hw": list(config.input_hw),
                "train_policy": "fixed",
                "num_epochs": int(config.num_epochs),
                "batch_size": int(config.batch_size),
                "val_batch_size": int(config.batch_size),
                "scale_tier": str(config.tier),
                "category_name": str(config.category_name),
                "lr": 1e-2,
                "backbone_lr": 1e-2,
                "use_amp": False,
            },
        )
        sweep_run(sweep_cfg)

        # Aggregate
        elig_cfg = EligibilityConfig.cli(
            argv=False,
            data={
                "auto": True,
                "kcd_root": str(workdir),
                "out": str(workdir / "manifest.tsv"),
                "out_json": str(workdir / "manifest.json"),
                "max_desktop_ms": 500.0,        # generous for CPU smoke
                "include_smoke_models": True,
                "allow_missing_desktop_bench": False,
                "print_winner": True,
            },
        )
        elig_run(elig_cfg)


def main():
    ModalKit.main()


if __name__ == "__main__":
    main()
