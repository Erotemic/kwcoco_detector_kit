# A latent aug-policy collision, caught on the fish project (2026-08-23)

Cross-project note. Nothing here broke a sea-lion run, but it would have broken
the next one, and the fix changes the augmentation schedule every run in this
project will get from now on.

## What happened elsewhere

The fish project's `gen002` (slurm job 489) went irrecoverably NaN partway
through epoch 1 and then trained 11 more epochs to AP 0.000 before anyone
noticed. Root cause, in full, is in
[the fish journal](../../../viame_fish_2026/docs/journals/2026-08-23_stop_epoch_collision_and_gen003.md).

The short version: `train_policy=fixed` pins `collate_fn.stop_epoch = 1`
(`deimv2.py:287`), and at `epoch == stop_epoch` DEIMv2 reloads model,
optimizer, **GradScaler** and EMA state from `best_stg1.pth`
(`det_solver.py:83-86`). Commit `2b3a911` — which scaled the augmentation
policy to the schedule length, fixing a genuine bug — clamps `e0 = max(1, ...)`,
which lands on **1 for every schedule shorter than ~80 epochs**. So the
optimizer and loss scale get reset to their pre-augmentation values at the
exact step augmentation switches on.

## Why no sea-lion run was harmed

Timing, and nothing else:

| | |
|---|---|
| last sea-lion slurm run | 2026-06-27 |
| collision introduced (`2b3a911`) | 2026-08-18 |

Every sea-lion run to date emitted upstream's raw `[4, 78, 148]`, so `e0` was
4 and never coincided with `stop_epoch`. The bug was purely latent here.

Note the flip side, which the fish project paid for: those raw boundaries mean
**stages 3 and 4 were unreachable** on a 30- or 45-epoch schedule. Every
sea-lion run so far entered the Mosaic stage at epoch 4 and stayed there, never
getting the clean fine-tuning epochs the recipe is built around. That is the
bug `2b3a911` was fixing.

## What changes for the next run

With `a5cee5b`, both of this project's schedules get a proper four-stage policy
with no collision:

| schedule | pre-`2b3a911` | `2b3a911` (collides) | now |
|---|---|---|---|
| 30 epochs | `[4, 78, 148]` | `[1, 16, 29]` | **`[2, 16, 29]`** |
| 45 epochs | `[4, 78, 148]` | `[1, 23, 44]` | **`[2, 23, 44]`** |

So the next sea-lion run is the first here to get a NoAug warmup, a real
Mosaic stage that ends, a mid stage, and a genuine clean final epoch. Expect
the loss curve to look different from every previous run in this project —
specifically a step up when augmentation engages at epoch 2, which is normal
and not a regression.

## Also inherited

- **Training aborts on non-finite `pred_boxes`** (DEIMv2 fork `32c3cd7`)
  instead of dumping an 819 MB `NaN.pth` every step and training on. Once the
  weights are NaN every loss reads exactly `0.0000`, so gradients are 0 and
  *finite* — `GradScaler` skips nothing and upstream's
  `math.isfinite(loss_value)` guard never fires.
- **`run-health`** (`09298ff`) diagnoses a run from its slurm log — NaN
  zombies, mid-epoch stalls, AP collapse, NCCL watchdog aborts, OOM. Copy
  `projects/viame_fish_2026/scripts/run_health.sh` into this project's
  `scripts/` (it is a straight copy; it only reads `KCD_SLURM_LOG_DPATH` from
  the local `paths.sh`), or call the module by path:

  ```bash
  python3 kwcoco_detector_kit/monitoring/log_health.py <slurm-log> --num_epochs 45
  ```

- **AMP is fp16**, unchanged. bf16 was tried on fish and reverted: DEIMv2 is an
  fp16 recipe and bf16 finished below both fp16 runs.

All of it requires an image rebuild to take effect — the kit is baked.
