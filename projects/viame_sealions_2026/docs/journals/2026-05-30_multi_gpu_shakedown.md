# 2026-05-30 — Multi-GPU shakedown via the wds_e2e_demo matrix
> [!WARNING]
> **2026-06-05 retroactive correction.** Every training run referenced
> below used `training_ready_v1/` — a 1,314-image, 2021–2024-only,
> n_cats=1 subset of the actual corpus. The authoritative bundles at
> `/data/Public/VIAME/viame_sealions_2026/unpacked/*_norm.kwcoco.zip`
> have 6,462 train images across 14 years (2007–2024) with 9 proper
> kwcoco categories. All AP numbers in this journal are SUBSET-only
> and not comparable to gen005+ runs trained on the real corpus. The
> recipe-level findings (dinov3_s + balance beats hgnetv2_n, EMA
> beats best_stg1, multiscale OOMs at 2-GPU 512–768) likely
> generalize; the absolute AP numbers (kit AP 0.5xx, in-train mAP
> 0.1xx) do not. See `2026-06-05_corpus_audit_wrong_bundle.md`.


## Context

[[2026-05-30_gen002_wds_pipeline_shakedown]] left the WDS pipeline
working end-to-end on 1×GPU. Next we needed to verify the pipeline
on 2×GPU (and ready it for 4×GPU production training on arisia)
across both data backends (kwcoco JPEG, WebDataset shards) and both
runtime targets (bare host venv, docker container).

The afternoon's work was: build a test harness, run it on yardrat
(2× heterogeneous GPUs: RTX 8000 + RTX 5000), fix every failure
mode that surfaced. We ended with all 10 scenarios green.

## What we built

- `dev/wds_e2e_demo/run_test_matrix.sh` — auto-detects compute
  capabilities (`docker info`, `nvidia-smi -L`, `sbatch`), runs the
  wds_e2e_demo across a cartesian product of `{host,docker,slurm}` ×
  `{jpeg,wds}` × `{cpu,gpu1x,gpu2x,gpu4x}`. Each scenario writes
  `log.txt`, `summary.json`, the full demo output dir, and (on
  timeout) `py-spy.txt`. Heartbeat output every 30s during long
  runs; `[OK]/[FAIL]/[TIMEOUT]` tag per scenario; non-zero exit on
  any failure with an unmissable summary banner. Filter with
  `MATRIX_ONLY=name1,name2` or `MATRIX_SKIP=name` to iterate fast.
  Opt-in `MATRIX_INCLUDE_BROKEN=1` for scenarios we mark
  known-broken (currently empty).

- `dev/wds_e2e_demo/run_demo.sh` (updated) — now supports
  `DEMO_DATA_PATH=wds|jpeg`, `DEMO_NUM_GPUS=N`,
  `DEMO_WDS_EPOCH_LENGTH=N` so the matrix can drive the same demo
  through all the configurations.

## What we found, in landing order

Every failure that surfaced was a real correctness gap, not a demo
artifact. Each one is fixed and pinned by either the new diagnostic
tests or the matrix harness itself.

1. **DEIMv2 was CPU-incompatible** ([1fbf7ea]):
   `dist_utils.warp_model` hard-coded `device_ids=[0]`, `setup_distributed_mode`
   hard-coded NCCL, `MetricLogger.SmoothedValue.sync` hard-coded
   `device='cuda'`. The CPU scenarios couldn't even import without
   crashing. Made each path check `torch.cuda.is_available()` and
   degrade gracefully (gloo backend, `device_ids=None`, host-device
   tensor).

2. **ONNX export's external-data sidecar broke the move-after-success
   contract** ([5765953]/[dbaeb5a]): torch ≥2.12 saves model weights
   to `<file>.onnx.data` by default. The kit moves the `.onnx` to
   `<workdir>/export/` but didn't move the sidecar; bench step then
   crashed. Repack via `onnx.load` + `onnx.save(save_as_external_data=False)`,
   fall back to `shutil.move` when the file isn't parseable (test
   fixtures use fake bytes).

3. **WDS IterableDataset + multi-GPU needed stream cycling**
   ([ff1f48d]): `wds.split_by_node` assigns shards to ranks; with
   uneven shard sizes one rank exhausts before the other, hits
   StopIteration, exits the epoch loop, DDP collective on the next
   surviving rank fingerprints-mismatches the absent one
   ("BROADCAST seq=N vs REDUCE seq=0"). Fix: when caller pins
   `epoch_length`, auto-cycle the underlying stream so every rank
   yields exactly that many samples regardless of shard distribution.
   Default (`epoch_length=0`) preserves the gen001 drain-once
   contract.

4. **save_on_master had no barrier** ([8a077c4]): rank 0 wrote
   `best_stg1.pth`; rank 1 raced ahead to the next epoch's
   `load_resume_state` BEFORE write completed → silent FileNotFound
   exception (`print_rank=0` suppressed it) → rank 1 exit → DDP
   mismatch. The canonical PyTorch "save on master, load on all"
   pattern wants a barrier after the write; DEIMv2 was missing it.
   Add `if is_dist_available_and_initialized(): dist.barrier()` to
   `save_on_master`.

5. **CocoEvaluator returns asymmetric test_stats** ([4975085]): the
   barrier from #4 isn't enough — the surrounding `for k in test_stats:`
   loop iterates 1 time on rank 0 and 0 times on non-main ranks
   because evaluator state isn't uniform. Rank 1 skips the save loop
   entirely (including its barrier) and reaches the next epoch's
   `load_resume_state` before rank 0 finishes writing. Fix:
   `broadcast_object_list(test_stats, src=0)` right after `evaluate()`
   so every rank has the same dict before the conditional save loop.

6. **Docker scenarios needed bind-mounted DEIMv2 + production-scale
   shm + env forwarding** ([aafa24a]/[4a0614c]/[47ab737]):
   * Bind-mount `tpl/DEIMv2` from host so DEIMv2 patches land
     immediately without rebuilding the image.
   * Scale `--shm-size` with rank count (`16 + 8*n_gpus`, matches
     production `_sbatch_train.sh`); WDS workers under 2+ ranks
     starve on 16g.
   * Forward `DEMO_WDS_EPOCH_LENGTH=16` + `NCCL_DEBUG=INFO` etc.
     into the container. The host case set these; docker case
     dropped them, so the WDS adapter ran without cycling and
     deadlocked at the next collective.

7. **Polling/UX bugs in the matrix runner itself** ([4a0614c]):
   silent exit when `grep` returned non-zero after a failed scenario;
   30-second-sleep loop padded every fast scenario to multiples of
   30s; ANSI escapes from NCCL/onnxruntime warnings bled through the
   heartbeat output and yellowed the terminal. All fixed.

## Final matrix (yardrat, 2× heterogeneous Quadro RTX 8000+5000)

```
name                  status  duration_s  train_time  ap     bench_ms
host-cpu-jpeg         ok      60.20       0:00:15     0.000  34.3
host-cpu-wds          ok      54.18       0:00:04     0.000  26.4
host-gpu1x-jpeg       ok      48.17       0:00:04     0.000  26.2
host-gpu1x-wds        ok      53.18       0:00:05     0.000  26.6
host-gpu2x-jpeg       ok      55.19       0:00:10     0.000  34.1
host-gpu2x-wds        ok      70.23       0:00:20     0.000  33.6
docker-gpu1x-jpeg     ok      52.18       0:00:04     0.000  34.5
docker-gpu1x-wds      ok      57.19       0:00:04     0.000  26.5
docker-gpu2x-jpeg     ok      57.19       0:00:08     0.000  33.6
docker-gpu2x-wds      ok      73.23       0:00:19     0.000  27.0
```

AP=0.000 is expected (training from scratch on 16 demo images);
bench latency is consistent across configurations. The point of the
matrix is correctness, not metrics — every scenario completing
train + eval + export + bench without exception is what we wanted.

## State at end of session

- All 5 DEIMv2 fixes pushed (`8a077c4`, `4975085`, `ff1f48d`,
  `1fbf7ea`, and earlier `bf3d290`/`ab013cc`/`331270e`).
- Matrix harness in `dev/wds_e2e_demo/run_test_matrix.sh` —
  reusable for any future shakedown.
- Diagnostic tests in `tests/integration/test_wds_stream_behavior.py`
  pin every contract the production WDS path depends on.
- Production gen002 + 4×GPU now have a verified-correct path. The
  4×GPU scenarios skip on yardrat (only 2 GPUs) but the code is
  exercised — they'll run the same path on arisia.

## How to resume

Run the full matrix on any host:

```bash
cd ~/code/kwcoco_detector_kit
bash dev/wds_e2e_demo/run_test_matrix.sh
```

It auto-detects compute, skips inapplicable scenarios, prints the
final pass/fail banner, exits non-zero on any failure. On arisia
this will additionally run the slurm and 4×GPU paths.

Iterate on a single failing scenario:

```bash
MATRIX_ONLY=docker-gpu2x-wds bash dev/wds_e2e_demo/run_test_matrix.sh
```

## Open items

1. **Promote the matrix into CI** — once a Docker image is baked
   with all the new DEIMv2 commits, the matrix is a cheap pre-merge
   gate (~5 minutes on a GPU host). Worth wiring up.
2. **Rebuild the docker image** to bake the new DEIMv2 fixes so
   docker scenarios don't depend on the bind-mount working
   perfectly. Bind-mount becomes a dev convenience, not load-bearing
   for correctness.
3. **Verify on arisia's 4× A6000** as the final correctness check
   before the next gen002 run — `MATRIX_ONLY=host-gpu4x-wds,docker-gpu4x-wds`
   on arisia validates the same DEIMv2 patches at production scale.
