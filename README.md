# kwcoco-detector-kit

Domain-agnostic object-detector training pipeline on **kwcoco** datasets. Scales from a single 12 GB GTX 1080 Ti to multi-GPU A100 / H100 / Blackwell clusters. Ships two trainer plugins (DEIMv2, OpenGroundingDINO) plus a mock CPU detector for CI smoke.

## Quick start

```bash
git clone https://github.com/Erotemic/kwcoco-detector-kit.git
cd kwcoco-detector-kit
git submodule update --init --recursive          # tpl/DEIMv2 + tpl/Open-GroundingDino
pip install -e ".[dev]"
bash examples/kwcoco_demo/run_smoke.sh
```

`run_smoke.sh` exercises the full pipeline (synth kwcoco → tile → train mock → ONNX export → eval → eligibility manifest) in <90 s on a 1-CPU laptop.

## What's in the box

- **`data/`** — kwcoco tile augmentation (three modes: full-only, quadrant grid, multi-scale fixed-size), positive + hard-negative merging, offline hard-negative mining, kwcoco → MSCOCO export.
- **`trainers/`** — pluggable trainer interface; `deimv2` covers 12 variants (HGNetv2 Atto/Femto/Pico/N/S/M/L/X + DINOv3 S/M/L/X); `opengroundingdino` covers DINOv2 + BERT + DETR; `mock_tiny` is a CPU smoke detector.
- **`predictors/`** — trained-checkpoint inference adapters used by the eval + hard-neg mining paths.
- **`export/`** — ONNX export + modelspec sidecar, torch ↔ ONNX parity guard, deployment package YAML.
- **`eval/`** — kwcoco eval driver, checkpoint shortlist sweep, ONNX desktop benchmark.
- **`orchestration/`** — Pareto sweep state machine, round-based hard-negative mining driver, eligibility manifest, setup-time `--check-env` probe.
- **`config-init` / `config-inspect` / `config-edit`** — editable environment + dataset YAML configs with host and kwcoco introspection; see [`docs/configs.md`](docs/configs.md).

All CLIs are [scriptconfig](https://gitlab.kitware.com/utils/scriptconfig)-based; `python -m kwcoco_detector_kit --help` or `kwcoco-detector-kit --help`.

## Scale tiers

| Tier | Hardware | Recommended variants |
|---|---|---|
| **S** | 1× 12–16 GB (GTX 1080 Ti / Titan X) | DEIMv2 HGNetv2 Atto / Femto / Pico |
| **M** | 1× 24 GB (RTX 3090 / 4090) | DEIMv2 HGNetv2 N / S, DEIMv2 DINOv3-S |
| **L** | 1× 48 GB (L40S / RTX 6000 Ada) | DEIMv2 DINOv3-M, OGDino-Swin-Tiny |
| **XL** | 1× 80 GB (A100 / H100) | DEIMv2 DINOv3-L/X, OGDino-Swin-Base |
| **2-4×L / 4×XL** | DDP cluster | full sweep, larger effective batches |
| **cloud** | SLURM / k8s | tier-L/XL × N nodes with cloud-mount kwcoco |

See [`docs/scale_tiers.md`](docs/scale_tiers.md).

## Engineering memory

This kit ships agent-readable engineering memory in [`dev/`](dev/) — 19 documented failure modes from the prior prototype + 4 distilled benchmark candidates. New >1 h-debug bugs land in [`dev/journals/lessons_learned.md`](dev/journals/lessons_learned.md) above the seed divider.

## Project status

Phase 1 in flight (port + RGB + tier M/L single-GPU). See [`CHANGELOG.md`](CHANGELOG.md) and [`PLAN.md`](PLAN.md) for the full roadmap.

Phase-3 (webdataset / multispectral / cloud) ship-vs-deferred status is tracked in [`docs/phase3_status.md`](docs/phase3_status.md).

## `examples/` vs `projects/`

- **`examples/`** — minimal, self-contained demonstrations of one capability (e.g. `kwcoco_demo/` the 90 s CPU smoke). Not maintained as live campaigns. The `sealion_aerial/` and `viame_sealions_2026/` example dirs are **superseded** historical references (banners at their top).
- **`projects/`** — real, ongoing experimental campaigns with their own scripts, class schemes, and run registry. `projects/viame_sealions_2026/` is the live sea-lion project and the reference shape new projects copy.

## License

Apache-2.0.
