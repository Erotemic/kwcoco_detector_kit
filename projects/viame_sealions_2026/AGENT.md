# AGENT.md — viame_sealions_2026

Project-scoped context for agents working in this subtree. The
kit-wide conventions live in the kit-root [CLAUDE.md](../../CLAUDE.md);
this file holds the sea-lion-specific details an agent needs to make
informed changes here.

## Active run

- **Slurm job 2477** on arisia, `pup_vs_nonpup`, 4x A6000, 30 epochs.
  Submitted around 2026-05-21/2026-05-22 (a timezone discrepancy may
  put the recorded start date one day off — verify on arisia with
  `squeue` / `sacct` before treating either value as authoritative).
- **Variant**: `deimv2_dinov3_s` (DEIMv2 + DINOv3 backbone).
- **Per-iter ETA at start**: ~3:50/epoch — multiply by 30 to estimate
  walltime risk against the 72h sbatch limit.
- **Registry row**: `2026-05-21-arisia-deimv2_dinov3_s-pup_vs_nonpup`
  in [docs/training_runs.yaml](docs/training_runs.yaml).

## Class scheme convention

`target_order` in [docs/class_schemes.yaml](docs/class_schemes.yaml)
defines the trained model's class-index assignment (i-th name → class
index i). Downstream code reads this list, not the mapping's
value-iteration order. **Reordering `target_order` invalidates every
checkpoint already trained against that scheme** — don't do it without
rebuilding the bundles and retraining.

Schemes (P0 → P2):

- `single_sealion` (1 cls) — P0 baseline.
- `pup_vs_nonpup` (2 cls) — P1, currently training.
- `lifestage_6cls` (6 cls) — P2 operational.

## Data location

- **arisia**: `/data/users/jon.crall/dvc-repos/viame_sealions_2026/`
  (matches the `KCD_DATA_DPATH` default in `scripts/paths.sh`).
- **namek**: the user has the data mounted at
  `/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026/`
  and treats it as read-only. Override `KCD_DATA_DPATH` in your shell
  rc on namek.

The project tree (scripts/docs/tests) lives in the kit and is
transferred via git; data artifacts (kwcoco bundles, unpacked imagery)
are NOT versioned and stay in `KCD_DATA_DPATH`.

## Workflow

1. **Build a per-scheme kwcoco bundle**

   ```bash
   python3 scripts/build_scheme_kwcoco.py --scheme <name>
   ```

   Writes `$KCD_SCHEMES_DIR/<name>/{train,vali,test}.kwcoco.zip` plus a
   `scheme_report.json` the expensive tests compare against.

2. **Fetch the pretrained init checkpoint**

   ```bash
   bash scripts/fetch_pretrained.sh
   ```

   Downloads + converts the DEIMv2+DINOv3 COCO checkpoint to a
   DEIMv2-loadable `.pth` at `$KCD_DEIMV2_DINOV3_S_COCO_PTH`.

3. **Submit a slurm training run**

   ```bash
   bash scripts/submit_pup_vs_nonpup.sh
   ```

   `sbatch`-submits [scripts/sbatch_pup_vs_nonpup.sh](scripts/sbatch_pup_vs_nonpup.sh)
   and tails the log via the kit's `smoketests/dino_v2_4x/slurm/follow_job.py`.
   The job runs [scripts/launch_pup_vs_nonpup_arisia.sh](scripts/launch_pup_vs_nonpup_arisia.sh)
   inside the kit's docker image.

4. **Record the result**

   ```bash
   python3 scripts/training_registry.py update <run-id> \
       --status done \
       --metric vali_map=... --metric vali_map50=... \
       --artifact detect_metrics_json=...
   ```

## Open items

- **Walltime ETA monitoring** for job 2477: ~3:50/epoch × 30 epochs is
  close to the 72h sbatch limit if epochs slow down. Check `sacct -j 2477`
  for elapsed and queue a re-submit if it cuts close.
- **Disk monitoring** on arisia: the previous run died after the
  filesystem hosting `$KCD_TRAINING_ROOT` filled. `launch_pup_vs_nonpup_arisia.sh`
  now fails fast if less than `KCD_MIN_FREE_GB` (default 30) is
  available — but watch headroom during training.
- **Next scheme to schedule**:
  - `single_sealion` (P0 baseline) — establishes the
    single-class detection AP floor for comparison.
  - `lifestage_6cls` (P2 operational) — full age-sex classifier.

  [docs/research_plan.md](docs/research_plan.md) has the phase-gate
  criteria.
- **`find_unused_parameters=True` warning in DDP**: a minor perf
  cleanup; not load-bearing.

## Investigating remote issues

If something breaks on arisia (job death, disk fill, OOM), check
**before** attributing it to our work:

- `sacct -j <jobid> --format=JobID,State,Reason,ExitCode,Elapsed,MaxRSS`
- `df -h` on the affected mount
- `dmesg | tail` for kernel-side OOM kills, virtio storms, etc.
- arisia logs in `/var/log/`

It's a shared machine — failure causes are not always ours. (See
[[feedback-no-blame-self-first]] in agent memory.)
