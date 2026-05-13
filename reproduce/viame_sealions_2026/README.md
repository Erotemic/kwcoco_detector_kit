# Reproduce: VIAME Sea Lions 2021-2024 OpenGroundingDINO

End-to-end real-data run for the VIAME Steller sea lion aerial imagery on
top of OpenGroundingDINO Swin-T. Designed for a 4x A6000 Slurm host with
the kwcoco-detector-kit Docker image, following the smoke ladder under
[smoketests/dino_v2_4x/](../../smoketests/dino_v2_4x/) once stage 03
passes.

Compared to the smoke (`03_ogdino_viame_subset_4gpu.sh`):

- Uses the full prepared splits (no `make_subset_abs` truncation).
- `INPUT_SIZE=800` (the original 3090 tiling target).
- Positive-only multiscale tiling at scales `1.0,0.5,0.25` with the
  shared tile cache under `$KCD_CACHE_ROOT`.
- `BATCH_SIZE=16` per GPU — sized for A6000 headroom (47 GB; smoke saw
  ~13.6 GB at b=8 input=800 on a 3090).

## Layout

```
reproduce/viame_sealions_2026/
├── README.md
├── run_ogdino_full.sh        # payload (executed inside the docker image)
└── slurm/
    ├── submit.sh             # sbatch submit (no-arg "real run")
    └── run_in_docker.sh      # Slurm job script — starts Docker
```

## Prerequisites

1. Prepared splits under `$DATA_DPATH/training_ready_v1/`
   (`train.kwcoco.zip`, `vali.kwcoco.zip`, `test.kwcoco.zip`) — produced
   by [examples/viame_sealions_2026/prepare_training_kwcoco.py](../../examples/viame_sealions_2026/prepare_training_kwcoco.py).
2. The Docker image already built on this host (default tag
   `kwcoco-detector-kit:ogdino-cu132-arisia`) — see
   [docker/opengroundingdino/](../../docker/opengroundingdino/).
3. Smoke stage `03_ogdino_viame_subset_4gpu.sh` has passed under
   [smoketests/dino_v2_4x/](../../smoketests/dino_v2_4x/) — confirms that
   Docker, Slurm, GPU access, and the path-rewrite plumbing all work
   end-to-end on the host.

## Submit

```bash
cd /path/to/kwcoco_detector_kit
bash reproduce/viame_sealions_2026/slurm/submit.sh
```

Common overrides:

```bash
BATCH_SIZE=24 NUM_EPOCHS=16 \
KCD_REPRODUCE_TAG=ogdino_swint_full_b24_e16 \
bash reproduce/viame_sealions_2026/slurm/submit.sh
```

`KCD_REPRODUCE_TAG` segregates run roots so multiple configurations can
live side by side under
`$KCD_EXPT_DPATH/reproduce/viame_sealions_2026/<tag>`.

## What it does

`run_ogdino_full.sh`, executed inside the container:

1. **Restage parent kwcoco** — the prepared splits encode the laptop's
   absolute image paths (`/media/joncrall/raid/...`). Inside the
   container the same data is mounted under `$DATA_DPATH`, so the
   script writes rewritten copies to `$KCD_REPRODUCE_ROOT/staged/`
   using `$KCD_PATH_REWRITE_PREFIXES`. (Same idea as the smoke's
   `make_subset_abs`, but without the subset step. The reuse-check
   verifies the *verbatim* paths in the staged file resolve here — if
   the rewrite map changes, the staged file is rebuilt.)
2. **Tile** the staged splits into `$KCD_CACHE_ROOT/tiles_t800_...`,
   positive-only multiscale. Cached across runs.
3. **Sweep OpenGroundingDINO Swin-T** on the tiles, 4-GPU distributed,
   `--use_amp true`, no export/bench, eval gated by `KCD_DO_EVAL`.

## Knobs

| env var | default | notes |
|---|---|---|
| `BATCH_SIZE` | `16` | per GPU at input=800; bump as memory allows |
| `VAL_BATCH_SIZE` | `$BATCH_SIZE` | |
| `NUM_EPOCHS` | `12` | upstream LR schedule isn't epoch-aware; revisit if scaling |
| `NUM_GPUS` | `4` | |
| `INPUT_SIZE` | `800` | also drives tile size |
| `SOURCE_SCALES` | `1.0,0.5,0.25` | multiscale tile pyramid |
| `STRIDE_FRAC` | `0.75` | tile sliding-window stride |
| `LR` / `BACKBONE_LR` | `1e-4` / `1e-5` | |
| `SCALE_TIER` | `2-4xL` | matches 4-GPU dispatch |
| `KCD_DO_EVAL` | `0` | set `1` to run held-out eval after train |
| `KCD_REPRODUCE_TAG` | `ogdino_swint_full` | per-run output subdir |
| `IMAGE_TAG` | `kwcoco-detector-kit:ogdino-cu132-arisia` | Docker image |

Slurm-side knobs in `slurm/submit.sh`: `GPUS`, `CPUS_PER_TASK`, `MEM`,
`TIME_LIMIT`, `SLURM_PARTITION`, `ACCOUNT`.

## Memory headroom

Observed on the first 4x A6000 run at `INPUT_SIZE=800`:

| batch (per GPU) | max mem | epoch wall clock | step time |
|---|---|---|---|
| 16 | ~25 GB / 47 GB | ~27 min (539 steps) | ~3.0 s |
| 24 (default) | est. ~38 GB | est. ~22 min (359 steps) | TBD |

Push higher (28-32) only after confirming the val pass also stays
within budget — eval steps can carry more boxes than train. Single-3090
reference at input=800, b=8 hit ~13.6 GB (see the
[bring-up journal](../../dev/journals/2026-05-12-viame-sealions-ogdino.md)).

## Tuning notes

- **Data-loader contention.** OGDino's `train_dist.sh` calls `main.py`
  with the upstream default `--num_workers=8`, so 4 GPUs × 8 = 32
  worker processes contend for the 12 CPUs the Slurm job requests.
  If GPU utilization looks "mixed", try `CPUS_PER_TASK=16` on submit
  (Slurm partition permitting) to give the dataloaders breathing room.
  Lowering per-GPU `num_workers` would require patching
  `tpl/Open-GroundingDino/train_dist.sh` to forward a `NUM_WORKERS` env
  var — not done here yet.
- **LR vs effective batch.** The journal's b=8 (single 3090) reference
  reached AP50=0.78 at `LR=1e-4`. With `BATCH_SIZE=24` and 4 GPUs the
  effective batch is 96 (12x the journal reference). Loss is still
  decreasing at the current `LR=1e-4`, so we are not bumping it by
  default. Try `LR=1.5e-4` (sqrt scaling) if convergence plateaus
  early.
- **Tokenizer warning.** `TOKENIZERS_PARALLELISM=false` is exported by
  default to silence the fork warning the BERT text encoder emits when
  dataloader workers fork.
