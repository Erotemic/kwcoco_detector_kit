"""
Recipe runner — one YAML file drives an end-to-end run.

A "recipe" is a single self-contained YAML describing **what to train,
on what data, into what workspace**. Running the recipe executes (in
order):

    1. check-env --runtime              (skippable with --skip-checks)
    2. sweep over the recipe's matrix
    3. eligibility manifest aggregation

The recipe is the unit of reproducibility for the shitspotter mobile
quality push (v6, v7, v8, ...). Each experiment commits a recipe YAML
before the run and an EVAL.md after; the manifest TSV is the durable
record bound to that recipe.

Schema (``schema: recipe.v1``)
------------------------------

::

    schema: recipe.v1
    name: <kebab-case-identifier>           # required, used for run tagging
    description: <human-readable one-liner>
    data:
        train_kwcoco: <path>                # required
        vali_kwcoco:  <path>                # required
        test_kwcoco:  <path>                # required
        tiled: true                         # optional; declares the training
                                            #   data is pre-tiled (e.g. a
                                            #   tile-corpus). Drives eval.mode:auto.
        expect:                             # optional guard (KCD-DATA-01): assert
            n_images: 10671                 #   the train bundle's true contents
            n_annots: 22600                 #   before any GPU time. A filename is
            categories: [poop]              #   not a contract; this is.
        expect_mode: fail                   # fail (default) | warn
        tile_store: kwcoco_jpeg | webdataset   # default: kwcoco_jpeg
        train_wds_shards: <path>            # required when tile_store: webdataset
                                            #   output of kwcoco_dataloader's
                                            #   build_detection_webdataset CLI
        train_wds_epoch_length: <int>       # optional; 0 = drain shards once
        train_wds_source_to_target:         # optional mapping {raw: target}
            <raw_category>: <target_category>
                                            #   default: identity map over
                                            #   sweep.category_names
                                            # Vali always reads the kwcoco
                                            # path; see ADR-0001.
    workspace:
        kcd_root: <path>                    # required, becomes $KCD_ROOT
    sweep:
        trainer: deimv2 | opengroundingdino | mock_tiny
        matrix:                             # list of {variant,input_hw,train_policy}
          - variant: deimv2_pico
            input_hw: [416, 416]
            train_policy: fixed
        # The remaining fields are passed through verbatim to SweepConfig:
        num_epochs: 80
        batch_size: 16
        val_batch_size: 8
        category_names: poop
        lr: 0.0004
        backbone_lr: 0.0004
        use_amp: true
        scale_tier: M
        num_gpus: 1
        distributed: false
        do_export: true
        do_eval: true
        do_bench: true
    eval:                                   # first-class eval mode (KCD-EVAL-01)
        mode: auto                          # whole_image | tiled | auto (default)
                                            #   auto -> tiled iff training is
                                            #   tiled (data.tiled or a
                                            #   multiscale matrix train_policy).
        window: null                        # optional window px (default: model
                                            #   eval_spatial_size = train tile)
        overlap: 0.25                       # fractional window overlap
        nms_thresh: 0.5                     # cross-window NMS IoU
        keep_full: true                     # also run a whole-image pass + merge
        batch: 64                           # windows per forward pass
        device: cpu                         # cpu | cuda (cuda recommended)
    eligibility:
        max_desktop_ms: 80.0
        min_device_fps: 1.0                 # walk-and-scan use case
        include_smoke_models: false
    reproducibility:
        seed: 0
        deterministic: false                # opt-in stricter mode
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import scriptconfig as scfg
import yaml


SUPPORTED_SCHEMAS = {"recipe.v1"}


class RecipeRunConfig(scfg.DataConfig):
    """Run a recipe.yaml end-to-end (sweep + manifest)."""

    recipe = scfg.Value(None, position=1, required=True,
                        help="path to a recipe.v1 YAML file")
    skip_checks = scfg.Value(False, isflag=True,
                             help="skip the leading check-env --runtime probe")
    dry_run = scfg.Value(False, isflag=True,
                         help="parse + validate the recipe, print the resolved "
                              "sweep + eligibility configs, but do not run")
    force_train = scfg.Value(False, isflag=True,
                             help="re-train cells even if best_stg2.pth exists")
    force_export = scfg.Value(False, isflag=True)
    force_eval = scfg.Value(False, isflag=True)
    force_bench = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


def _load_recipe(fpath: Path) -> Dict[str, Any]:
    if not fpath.exists():
        raise FileNotFoundError(f"recipe file not found: {fpath}")
    # Expand ${VAR}/${VAR:-default} so recipes can be host-portable (KCD-CFG-01).
    from kwcoco_detector_kit.configs import expand_env_vars
    text = expand_env_vars(fpath.read_text(), source=str(fpath))
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"recipe must be a YAML mapping; got {type(parsed).__name__}")
    schema = str(parsed.get("schema", ""))
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(
            f"unknown recipe schema {schema!r}; supported: {sorted(SUPPORTED_SCHEMAS)}"
        )
    return parsed


_VALID_TILE_STORES = ("kwcoco_jpeg", "webdataset")


def _validate_required(recipe: Dict[str, Any]) -> None:
    """Fail fast with a clear message if required keys are missing."""
    if not recipe.get("name"):
        raise ValueError("recipe.name is required")

    data = recipe.get("data") or {}
    for key in ("train_kwcoco", "vali_kwcoco", "test_kwcoco"):
        if not data.get(key):
            raise ValueError(f"recipe.data.{key} is required")
        # Don't require the file to *exist* in dry_run mode; we validate
        # existence in the live path so containerized runs that mount
        # paths late still get a clear error.

    tile_store = str(data.get("tile_store") or "kwcoco_jpeg")
    if tile_store not in _VALID_TILE_STORES:
        raise ValueError(
            f"recipe.data.tile_store must be one of {_VALID_TILE_STORES}; "
            f"got {tile_store!r}"
        )
    if tile_store == "webdataset" and not data.get("train_wds_shards"):
        raise ValueError(
            "recipe.data.train_wds_shards is required when "
            "tile_store: webdataset. Build the shards with "
            "`kwcoco_dataloader build_detection_webdataset` first."
        )

    workspace = recipe.get("workspace") or {}
    if not workspace.get("kcd_root"):
        raise ValueError("recipe.workspace.kcd_root is required")

    sweep = recipe.get("sweep") or {}
    if not sweep.get("trainer"):
        raise ValueError("recipe.sweep.trainer is required")
    matrix = sweep.get("matrix")
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("recipe.sweep.matrix must be a non-empty list")
    for i, cell in enumerate(matrix):
        if not isinstance(cell, dict):
            raise ValueError(f"recipe.sweep.matrix[{i}] must be a mapping")
        for key in ("variant", "input_hw"):
            if key not in cell:
                raise ValueError(f"recipe.sweep.matrix[{i}].{key} is required")
        if (not isinstance(cell["input_hw"], list)
                or len(cell["input_hw"]) != 2):
            raise ValueError(
                f"recipe.sweep.matrix[{i}].input_hw must be a list of "
                f"[H, W]; got {cell['input_hw']!r}"
            )


def _apply_reproducibility(recipe: Dict[str, Any]) -> None:
    """Set the env vars the trainer plugins read for seed/determinism.

    This is best-effort: the trainer subprocess is responsible for honoring
    these. We set them at the recipe-runner level so a child sweep + child
    trainer process inherits them.
    """
    repro = recipe.get("reproducibility") or {}
    seed = repro.get("seed")
    if seed is not None:
        os.environ["KCD_SEED"] = str(int(seed))
        os.environ["PYTHONHASHSEED"] = str(int(seed))
    if bool(repro.get("deterministic", False)):
        os.environ["KCD_DETERMINISTIC"] = "1"
        # Required by torch.use_deterministic_algorithms() when CUBLAS is
        # in use. Set defensively here even though the trainer plugins
        # are the ones that read KCD_DETERMINISTIC.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _build_sweep_data(recipe: Dict[str, Any], cli_overrides: Dict[str, Any]) -> Dict[str, Any]:
    data = recipe["data"]
    workspace = recipe["workspace"]
    sweep = recipe["sweep"]
    repro = recipe.get("reproducibility") or {}

    sweep_data = {
        "train_kwcoco": str(data["train_kwcoco"]),
        "vali_kwcoco": str(data["vali_kwcoco"]),
        "test_kwcoco": str(data["test_kwcoco"]),
        "kcd_root": str(workspace["kcd_root"]),
        "trainer": str(sweep["trainer"]),
        "matrix": sweep["matrix"],  # passed through as a Python list
    }

    # All remaining sweep fields are forwarded if present. Skipping the
    # known wrapper keys avoids leaking 'matrix' / 'trainer' twice.
    passthrough_keys = {
        "num_epochs", "batch_size", "val_batch_size",
        "category_names", "lr", "backbone_lr", "use_amp", "scale_tier",
        "num_gpus", "distributed", "keep_going",
        "do_export", "do_eval", "do_bench",
        "init_checkpoint",
        # Tiled-eval knobs under sweep: (back-compat; the `eval:` block is the
        # preferred interface — see _resolve_eval_block). These were silently
        # dropped before KCD-EVAL-01, which is why shitspotter v13's
        # `sweep.tiled_eval: true` never engaged.
        "tiled_eval", "tiled_eval_window", "tiled_eval_overlap",
        "tiled_eval_nms_thresh", "tiled_eval_keep_full", "tiled_eval_batch",
        "eval_device", "eval_read_workers",
    }
    for k in passthrough_keys:
        if k in sweep:
            sweep_data[k] = sweep[k]

    # First-class eval mode (KCD-EVAL-01). The `eval:` block is the clean
    # interface and, when present, overrides any legacy sweep.tiled_eval*.
    _resolve_eval_block(recipe, sweep_data)

    # Webdataset training-input plumbing. Recipe authors set the
    # storage backend declaratively under data.tile_store; the
    # SweepConfig consumer (and DEIMv2 trainer plugin) cares about the
    # concrete paths/params. SweepConfig expects source_to_target as a
    # JSON string, so encode if a YAML mapping was provided. If absent,
    # auto-derive an identity mapping over the recipe's category_names
    # so authors don't have to type it for the typical case where
    # source == target.
    if str(data.get("tile_store") or "kwcoco_jpeg") == "webdataset":
        import json
        sweep_data["train_wds_shards_dpath"] = str(data["train_wds_shards"])
        if "train_wds_epoch_length" in data:
            sweep_data["train_wds_epoch_length"] = int(
                data["train_wds_epoch_length"]
            )
        src_to_tgt = data.get("train_wds_source_to_target")
        if src_to_tgt is None:
            cats_field = sweep.get("category_names", "")
            cats = (
                [c.strip() for c in cats_field.split(",") if c.strip()]
                if isinstance(cats_field, str) else list(cats_field)
            )
            src_to_tgt = {c: c for c in cats}
        sweep_data["train_wds_source_to_target"] = json.dumps(src_to_tgt)

    # CLI force flags override recipe-level booleans.
    for k in ("force_train", "force_export", "force_eval", "force_bench"):
        if cli_overrides.get(k):
            sweep_data[k] = True

    # Seed env var is read by trainer plugins; not part of SweepConfig.
    return sweep_data


_EVAL_MODES = ("whole_image", "tiled", "auto")


def _training_is_tiled(recipe: Dict[str, Any]) -> Optional[str]:
    """Return a human reason string if training is tiled, else None.

    Used to resolve ``eval.mode: auto``. Signals (any one is sufficient):
      * ``data.tiled: true`` — explicit author declaration (e.g. a pre-built
        tile-corpus, where train_policy is `fixed` but the DATA is multi-scale).
      * any matrix cell whose ``train_policy`` starts with ``multiscale``.
    """
    data = recipe.get("data") or {}
    if bool(data.get("tiled")):
        return "data.tiled is set"
    for cell in recipe.get("sweep", {}).get("matrix", []) or []:
        policy = str(cell.get("train_policy", ""))
        if policy.startswith("multiscale"):
            return f"matrix cell train_policy={policy!r}"
    return None


def _resolve_eval_block(recipe: Dict[str, Any], sweep_data: Dict[str, Any]) -> None:
    """Resolve the recipe ``eval:`` block into SweepConfig tiled_eval* keys.

    First-class eval mode (KCD-EVAL-01). ``eval.mode`` is one of:
      * ``whole_image`` — resize each test image to the model input (the
        classic protocol; understates small-object AP for tile-trained models).
      * ``tiled`` — slide native-resolution windows and merge (TiledPredictor).
      * ``auto`` (default) — ``tiled`` iff training is tiled (see
        ``_training_is_tiled``), else ``whole_image``.

    The resolution is logged loudly so eval mode is never a silent surprise.
    When the ``eval:`` block is absent, a legacy ``sweep.tiled_eval`` (already
    passed through) is honored as-is for back-compat.
    """
    eval_block = recipe.get("eval")
    sweep = recipe.get("sweep") or {}

    if eval_block is None:
        # No eval: block. Honor legacy sweep.tiled_eval (already in sweep_data)
        # but still record the effective mode for the loud log + downstream.
        if "tiled_eval" in sweep:
            mode = "tiled" if bool(sweep.get("tiled_eval")) else "whole_image"
            print(f"[recipe] eval mode = {mode} (legacy sweep.tiled_eval; "
                  f"prefer an `eval:` block)")
        return

    if not isinstance(eval_block, dict):
        raise ValueError("recipe.eval must be a mapping")
    mode = str(eval_block.get("mode", "auto"))
    if mode not in _EVAL_MODES:
        raise ValueError(
            f"recipe.eval.mode must be one of {_EVAL_MODES}; got {mode!r}")

    if mode == "auto":
        reason = _training_is_tiled(recipe)
        resolved = "tiled" if reason else "whole_image"
        print(f"[recipe] eval.mode=auto resolved to {resolved!r} "
              f"({reason or 'no tiled-training signal'})")
    else:
        resolved = mode
        print(f"[recipe] eval.mode = {resolved!r}")

    sweep_data["tiled_eval"] = (resolved == "tiled")
    # Optional tuning fields map onto the SweepConfig knobs.
    _eval_field_map = {
        "window": "tiled_eval_window",
        "overlap": "tiled_eval_overlap",
        "nms_thresh": "tiled_eval_nms_thresh",
        "keep_full": "tiled_eval_keep_full",
        "batch": "tiled_eval_batch",
        "device": "eval_device",
        "read_workers": "eval_read_workers",
    }
    for src_key, sweep_key in _eval_field_map.items():
        if src_key in eval_block:
            sweep_data[sweep_key] = eval_block[src_key]


def _build_eligibility_data(recipe: Dict[str, Any]) -> Dict[str, Any]:
    workspace = recipe["workspace"]
    elig = recipe.get("eligibility") or {}
    kcd_root = Path(str(workspace["kcd_root"]))
    out = kcd_root / "manifest.tsv"
    out_json = kcd_root / "manifest.json"
    elig_data = {
        "auto": True,
        "kcd_root": str(kcd_root),
        "out": str(out),
        "out_json": str(out_json),
        "print_winner": True,
    }
    for k in ("max_desktop_ms", "min_device_fps", "include_smoke_models",
              "smoke_only", "allow_missing_desktop_bench", "device_index"):
        if k in elig:
            elig_data[k] = elig[k]
    return elig_data


def _check_input_paths(recipe: Dict[str, Any]) -> None:
    """Fail before the first GPU minute if a kwcoco path is wrong."""
    data = recipe["data"]
    for key in ("train_kwcoco", "vali_kwcoco", "test_kwcoco"):
        p = Path(str(data[key])).expanduser()
        if not p.exists():
            raise FileNotFoundError(
                f"recipe.data.{key} = {p} does not exist. "
                f"Check the bind-mount or dataset path."
            )
    if str(data.get("tile_store") or "kwcoco_jpeg") == "webdataset":
        shards = Path(str(data["train_wds_shards"])).expanduser()
        if not shards.exists():
            raise FileNotFoundError(
                f"recipe.data.train_wds_shards = {shards} does not exist. "
                "Run `kwcoco_dataloader build_detection_webdataset` to "
                "produce the shards bundle first."
            )
        # Footer files are how kwcoco_dataloader.readers.detection
        # discovers shards; if none are present the bundle is missing
        # or the writer crashed before close.
        if not any(shards.rglob("__footer__.json")):
            raise FileNotFoundError(
                f"recipe.data.train_wds_shards = {shards} contains no "
                "__footer__.json files. Re-run build_detection_webdataset; "
                "an earlier run likely crashed before closing the writer."
            )


def _assert_data_expectations(recipe: Dict[str, Any]) -> None:
    """Assert the train bundle matches a declared ``data.expect:`` block (KCD-DATA-01).

    Guards the "filenames lie" class of bug: a recipe author who knows the
    training set should have N images declares ``data.expect.n_images: N`` and
    the run fails loudly before any GPU time if the bundle disagrees. No-op when
    no ``expect`` block is present. Set ``data.expect_mode: warn`` to downgrade a
    mismatch to a warning.
    """
    data = recipe.get("data") or {}
    expect = data.get("expect")
    if not expect:
        return
    from kwcoco_detector_kit.data.manifest import compute_manifest, assert_expected
    strict = str(data.get("expect_mode", "fail")) != "warn"
    man = compute_manifest(data["train_kwcoco"])
    mismatches = assert_expected(
        man, expect, source="recipe.data.expect", strict=strict)
    if mismatches and not strict:
        print("[recipe] WARNING: data.expect mismatch(es) (expect_mode=warn):")
        for msg in mismatches:
            print(f"  - {msg}")
    else:
        print(f"[recipe] data.expect OK (train: {man.get('n_images')} images, "
              f"{man.get('n_annots')} annots, hash {man.get('content_hash')})")


def run(config) -> None:
    recipe_path = Path(str(config.recipe)).expanduser().resolve()
    recipe = _load_recipe(recipe_path)
    _validate_required(recipe)

    name = str(recipe["name"])
    print(f"[recipe] {name} ({recipe_path})")
    if recipe.get("description"):
        print(f"[recipe] {recipe['description']}")

    cli_overrides = {
        "force_train": bool(config.force_train),
        "force_export": bool(config.force_export),
        "force_eval": bool(config.force_eval),
        "force_bench": bool(config.force_bench),
    }

    sweep_data = _build_sweep_data(recipe, cli_overrides)
    elig_data = _build_eligibility_data(recipe)

    if bool(config.dry_run):
        print("\n[recipe] dry-run — would run sweep with:")
        print(yaml.safe_dump(sweep_data, sort_keys=True))
        print("\n[recipe] then eligibility manifest with:")
        print(yaml.safe_dump(elig_data, sort_keys=True))
        return

    _check_input_paths(recipe)
    _assert_data_expectations(recipe)
    _apply_reproducibility(recipe)

    # Create the workspace before either subcommand touches it.
    Path(str(recipe["workspace"]["kcd_root"])).expanduser().mkdir(
        parents=True, exist_ok=True,
    )

    if not bool(config.skip_checks):
        from kwcoco_detector_kit.orchestration.setup_audit import (
            CheckEnvConfig, run as audit_run,
        )
        audit_cfg = CheckEnvConfig.cli(
            argv=False,
            data={
                "groups": "core,onnx,deimv2",
                "runtime": True,
                # Don't auto-fail the recipe on a missing GPU when the
                # trainer is mock_tiny; otherwise require_gpu=True.
                "require_gpu": str(recipe["sweep"]["trainer"]) != "mock_tiny",
            },
        )
        rc = audit_run(audit_cfg)
        if rc != 0:
            print(
                f"[recipe] check-env --runtime returned {rc}. "
                f"Re-run with --skip_checks to bypass at your own risk.",
                file=sys.stderr,
            )
            sys.exit(2)

    from kwcoco_detector_kit.orchestration.pareto_sweep import (
        SweepConfig, run as sweep_run,
    )
    sweep_cfg = SweepConfig.cli(argv=False, data=sweep_data)
    sweep_run(sweep_cfg)

    from kwcoco_detector_kit.orchestration.eligibility import (
        EligibilityConfig, run as elig_run,
    )
    elig_cfg = EligibilityConfig.cli(argv=False, data=elig_data)
    elig_run(elig_cfg)

    print(f"\n[recipe] {name} complete.")
    print(f"[recipe] manifest -> {elig_data['out']}")


__cli__ = RecipeRunConfig
