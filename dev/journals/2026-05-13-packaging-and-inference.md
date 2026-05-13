# 2026-05-13 — Packaging and Inference Work

Goal: make trained detector outputs packageable after the fact, with robust
metadata/provenance, optional artifacts, archive support, and a common
`predict` command that can consume a package directory or archive.

Context from user:

- Successful full OpenGroundingDINO VIAME run completed on arisia.
- Reported metrics tail:
  - AP@[0.50:0.95] = 0.499
  - AP@0.50 = 0.772
  - AP@0.75 = 0.563
  - Training time = 5:44:35
- User rsynced raw results under:
  `/data/users/jon.crall/dvc-repos/viame_sealions_2026_expt`
- Need result paths to encode username and hostname as separate fields/components
  for rsync across machines.
- Need package creation to be post-hoc and robust to optional/missing artifacts.
- Need packages to support archive inputs (`.zip`, `.tar`, `.tar.gz`, etc.) as
  first-class objects with a YAML manifest referencing package members.
- Need efficient inference examples across model backends.

Findings so far:

- OpenGroundingDINO trainer currently has `supports_onnx_export = False`.
- The successful run was launched with `--no-do_export --no-do_bench`, so it is
  checkpoint/config based, not ONNX/modelspec packaged.
- Existing `kwcoco_detector_kit/export/package.py` was ONNX-centric; replacing
  it with a general package manifest builder.
- `/data/users/jon.crall/dvc-repos/viame_sealions_2026_expt` is not visible in
  this local workspace, so final package creation for the real sealion model may
  need to be provided as a ready-to-run command unless the path becomes visible.

Implementation notes:

- Started replacing `export/package.py` with:
  - `build_model_package(...)`
  - `open_package(...)`
  - `materialize_workdir(...)`
  - `PackageBuildConfig` CLI
- Manifest schema name: `kwcoco_detector_kit.package.v1`.
- Manifest backend for OGDino/checkpoint packages: `trainer_checkpoint`.
- Package layout being targeted:
  - `package.yaml`
  - `labels.json`
  - `weights/checkpoint.pth`
  - `training_config/<config>`
  - `training_config/datasets.json`
  - `training_config/policy.json`
  - optional `logs/train.log`
  - optional `eval/detect_metrics.json`
  - optional `exports/*.onnx`
  - `inference_examples/*`

Next steps:

1. Add a `predict` module/CLI that accepts a package directory/archive/YAML.
2. Register `package-build` and `predict` in the top-level CLI.
3. Add unit tests for package directory + archive load.
4. Add a mock_tiny package/predict round-trip test.
5. Try to build the real sealion package if result paths are visible; otherwise
   print exact command for arisia.

Progress:

- Reopened the tree after context compaction. Current dirty state is limited to
  package work plus the journal; `_slurm_monitor_show.py` and `logs.txt` are
  untracked user/reference files and should stay out of commits.
- Confirmed the existing eval driver already has the right kwcoco prediction
  pattern: copy image rows, rewrite `file_name` to absolute source paths, run
  `trainer.build_predictor(workdir).predict_image(...)`, and emit a pred kwcoco.
- Next implementation step is a reusable `kwcoco_detector_kit.predict` module
  that materializes package artifacts into the minimal workdir expected by
  trainer plugins.
- Added `kwcoco_detector_kit.predict` with `predict_kwcoco(...)` and a
  `PredictConfig` CLI. It accepts a package directory, archive, or manifest
  file and writes a kwcoco prediction bundle.
- Registered top-level CLI commands `package-build` and `predict`.
- Added a CPU test that trains `mock_tiny`, builds a `.zip` package, opens the
  archive manifest, and runs package-backed prediction over the synthetic
  kwcoco fixture.
- Verified:
  - `uv run --extra dev pytest -q tests/unit/test_package_predict.py`
  - `uv run --extra dev pytest -q tests/unit/test_mock_tiny.py tests/unit/test_package_predict.py`
  - `uv run kwcoco-detector-kit --help | head -80`
- Rechecked for the real arisia result path:
  `/data/users/jon.crall/dvc-repos/viame_sealions_2026_expt/...`; it is not
  mounted/visible in this local session. Need to run the final real package
  creation command on arisia or wherever that result tree is mounted.
- Added automatic package output path construction for the no-`@` convention:
  `<out_root>/<dataset>/<experiment>/users/<username>/hosts/<hostname>/<run_id>/<variant>.zip`.
  The path helper is covered in the package/predict unit test.

Arisia command to build the real sealion package once on the machine with the
results mounted:

```bash
kwcoco-detector-kit package-build \
    --workdir /data/users/jon.crall/dvc-repos/viame_sealions_2026_expt/reproduce/viame_sealions_2026/ogdino_swint_full/run/runs/opengroundingdino_swint_800x800_fixed \
    --out-root /data/users/jon.crall/dvc-repos/viame_sealions_2026_expt/packages \
    --trainer opengroundingdino \
    --variant opengroundingdino_swint_800x800_fixed \
    --category_name sealion \
    --dataset_slug viame_sealions_2026 \
    --experiment_slug ogdino_swint_full \
    --run_id 20260513T010928 \
    --train_kwcoco /data/users/jon.crall/dvc-repos/viame_sealions_2026/training_ready_v1/train.kwcoco.zip \
    --vali_kwcoco /data/users/jon.crall/dvc-repos/viame_sealions_2026/training_ready_v1/vali.kwcoco.zip \
    --test_kwcoco /data/users/jon.crall/dvc-repos/viame_sealions_2026/training_ready_v1/test.kwcoco.zip
```

Expected default package path:

```text
/data/users/jon.crall/dvc-repos/viame_sealions_2026_expt/packages/viame_sealions_2026/ogdino_swint_full/users/<username>/hosts/<hostname>/20260513T010928/opengroundingdino_swint_800x800_fixed.zip
```
