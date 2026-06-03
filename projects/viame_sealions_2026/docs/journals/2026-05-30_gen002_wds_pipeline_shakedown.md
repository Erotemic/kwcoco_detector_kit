# 2026-05-30 — gen002 WebDataset pipeline shakedown

## Context

[[2026-05-29_per_checkpoint_rescoring_results]] picked v6 (pup_vs_nonpup, hgnetv2_n,
320×320, fixed-policy) as the gen001 baseline. The gen002 direction is
"keep the model + resolution, swap the data pipeline to WebDataset shards so we
can train on the FULL universal tile bundle without choking the kit's
in-memory CocoDataset path." The full path needed wiring through:
prep job builds shards once → train jobs consume shards via a new
`WebDatasetCocoDetection` adapter inside DEIMv2.

This day was the gen002 first-end-to-end shakedown. Lots of small
bugs in series; the WDS path is now verified end-to-end on a CPU
demo, with the production gen002 job on arisia training healthily.

## What we ran

- **arisia jobs 2540–2552**: iterating on the gen002 submit script
  + DEIMv2 adapter. Most failed early. Final job (`pup_vs_nonpup_v2_hgnetv2_n_1gpu_arisia_gen002`) trained
  through epoch 1 before we pivoted to a shorter run config (see Next steps).
- **arisia job 2545 (single_sealion_v5)**: gen001-style baseline,
  resubmitted after the DDP-on-CPU fix. Training healthy: 0.25s/iter, ETA ~25h.
- **namek**: `dev/wds_e2e_demo/run_demo.sh` — new end-to-end demo that
  exercises the WDS pipeline on kwcoco demo data (shapes16 + shapes4).
  Runs in 17 sec on CPU. Will run on yardrat (3090) for the GPU smoke.

## What we learned (and the bugs we fixed)

Every bug found and shipped this session, in landing order:

1. **WDS `__len__` double-counted samples** ([bf3d290] in DEIMv2):
   `*.tar.index.json` lists every file in the shard, including the
   `<key>.json` sidecar alongside `<key>.jpg`. Summing `len(fnames)`
   reported 2x the real sample count. gen002 epoch nominal was 95,757
   iters when the true count was 47,878. **Fix**: count only entries
   with image extensions.

2. **WDS `__len__` re-decoded every index file per iter** ([ab013cc]):
   DEIMv2's `MetricLogger.log_every()` calls `len(dataloader)` once
   per iter for progress display. That cascaded to our `__len__`
   walking every `*.tar.index.json` + json-decoding on each call —
   multiple seconds at gen002 scale. **Fix**: memoize.

3. **Slurm `CUDA_VISIBLE_DEVICES` remap inside `--gpus=device=N`**
   ([03dfa2e] in kit, [_sbatch_train.sh]):
   `docker run --gpus=device=1` exposes only physical GPU 1 to the
   container, but the container remaps it to logical index 0. We were
   forwarding `CUDA_VISIBLE_DEVICES=1` into the container, which then
   pointed at a non-existent logical device 1. Model stayed on CPU,
   DDP refused with `module parameters {device(type='cpu')}`.
   **Fix**: container-side `CUDA_VISIBLE_DEVICES` is `0,1,...,n-1`
   counted from the slurm-assigned set, regardless of host indices.

4. **WDS shuffle buffer caused worker OOM-kill** ([331270e]):
   Default `shuffle_buffer=1024` × 4 workers = up to 12 GB of decoded
   PIL images sitting in heap. Kernel silently OOM-killed workers
   without logging anything in-process. DataLoader then hung
   forever waiting on dead workers' queues. **Fix**: cap defaults
   to `shuffle_buffer=128, shardshuffle=8`; plumb `stream_kwargs`
   through for caller override.

5. **DEIMv2 had three hard-coded `'cuda'` references** ([1fbf7ea]):
   surfaced when running the CPU demo on namek (whose GPU is broken
   ATM). `dist_utils.warp_model` always called DDP with
   `device_ids=[0]`; `dist_utils.setup_distributed_mode` always
   initialised NCCL; `MetricLogger.SmoothedValue.sync` built tensors
   on CUDA. **Fix**: each call checks `torch.cuda.is_available()`
   and degrades gracefully (gloo backend, `device_ids=None`, host
   device for tensors). No GPU-side behavior change.

6. **ONNX external-data sidecar dropped during move** ([5765953]):
   torch 2.12 saves `<file>.onnx.data` alongside the `.onnx` by
   default, even for small models. The kit moved the `.onnx` from
   the checkpoint dir to `<workdir>/export/` but not the `.data`
   sidecar; bench step then crashed with
   `ONNXRuntimeError: External data path does not exist`.
   **Fix**: load via `onnx.load` (follows external data), save back
   with `save_as_external_data=False` so the destination is
   self-contained. Best-effort cleanup of source-side `.data`.

## What the data showed (don't panic next time)

Looking at gen002 2552's epoch 0 trace, I initially thought the
training had truncated to ~5000 of 47878 iters. **It hadn't.** The
log's per-iter `time:` field is a moving-window estimate skewed by
the 6-min warmup (workers spin up, shuffle buffers fill, first
batches are slow). The epoch-end summary
`Total time: 0:31:48 (0.0399 s/it)` is the real average — 47878
iters at 40ms each, consistent. There IS a divergence between
`len(ds)` and the actual iter-yield when source samples have empty
annotations (our `__iter__` skips them), but at gen002 scale this
is small. See `tests/integration/test_wds_stream_behavior.py`
for the pinned contract.

Memory `max mem: 2403 MiB` was constant across all 3500 iters —
much lower than v5's 10.5 GB at the same batch/resolution. Real
reason: gen002 uses `use_amp=true` (v5 doesn't), which halves
activation memory, and during epochs 0-3 Mosaic/MixUp/CopyBlend
haven't kicked in yet (they fire from epoch 4 per the policy).
Not a bug.

## Verification artifacts

- `tests/integration/test_wds_stream_behavior.py` — 14 diagnostic
  tests pinning `__len__` memoization, multi-worker yield stability,
  empty-skip behavior, `stream_kwargs` plumbing, etc. All pass.
- `dev/wds_e2e_demo/run_demo.sh` — end-to-end demo, runs the kit's
  actual sweep CLI against kwcoco demo data. 17 sec on CPU.
  Self-detects python (host venv → docker image → system) so it
  runs anywhere.

## State at end of session

- **arisia v5 (single_sealion, 2545)**: training healthy on GPU 0,
  iter time ~0.25s, ETA ~25h total.
- **arisia gen002 (2552)**: was healthy at ~5000 iters into epoch 1
  before user asked to scancel + shorten. Run dir is intact in case
  we want to inspect.
- **Pending resubmit on arisia**: gen002 with `KCD_WDS_EPOCH_LENGTH=14351`
  (match v5's nominal per-epoch sample count, keep 30 epochs, ~5h total).
  Command in section below.
- **Push pending from namek**: `tpl/DEIMv2` (5 commits) and the kit
  (7 commits including this journal). User pushes manually since
  Claude can't auth gitlab/github.

## How to resume (verbatim commands)

**From namek**, push all the staged work:

```bash
cd /home/joncrall/code/kwcoco_detector_kit/tpl/DEIMv2 && git push origin main
cd /home/joncrall/code/kwcoco_detector_kit && git push origin main
```

**On arisia**, sync and resubmit gen002 with the shortened-epoch config:

```bash
cd /home/local/KHQ/jon.crall/code/kwcoco_detector_kit && git pull && git submodule update --init --recursive
scancel 2552  # if still running
rm -rf /data/users/jon.crall/kcd_sealion/runs/pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_gen002/runs
KCD_DEV_MOUNT_DEIMV2=1 \
    KCD_DEV_MOUNT_DATALOADER=1 \
    KCD_TRAIN_NUM_WORKERS=4 \
    KCD_WDS_EPOCH_LENGTH=14351 \
    bash /home/local/KHQ/jon.crall/code/kwcoco_detector_kit/projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_gen002.sh
```

**On yardrat** (GPU smoke before trusting any new gen002 change):

```bash
cd ~/code/kwcoco_detector_kit && git pull && git submodule update --init --recursive
DEMO_BATCH=8 DEMO_AMP=true DEMO_EPOCHS=5 DEMO_INPUT_HW="[320, 320]" \
    bash dev/wds_e2e_demo/run_demo.sh
```

If yardrat's venv lacks deps, use the docker image (header in demo script).

## Open items for next session

1. **gen002 resubmit results**: when the shortened-epoch run completes,
   compare detection AP (NFS-excluded) against v5/v6. If gen002 ≥ v5,
   the WDS path is operationally validated and we can promote it.
2. **`distractor_classes` not yet wired into gen002 scheme YAML for
   `pup_vs_nonpup`** — left default empty. NFS handling in this scheme
   is via the `drop:` list at the kit-side, so this is correct, but
   worth re-verifying when we move on to lifestage_6cls gen002 where
   the distractor list is non-trivial.
3. **The `__len__` vs iter-yield divergence** is currently expected
   behavior (empty-annotation skip). If we end up wanting
   `KCD_WDS_EPOCH_LENGTH` to mean "exact number of samples consumed
   per epoch" rather than "nominal length", the empty-skip behavior
   needs to change too. Today's contract is documented in
   `test_len_vs_actual_yield`.
4. **ONNX repack consumes some CPU on every export** — fine for now,
   but if the export step becomes a bottleneck consider just moving
   the `.data` sidecar alongside instead.
