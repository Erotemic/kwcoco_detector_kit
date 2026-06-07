# gen005 scale-up + eval/infra tooling (2026-06-07)

State of the gen005 round and the tooling built to support it.

## Results so far

**pup_vs_nonpup, dinov3_s, 640, tiled eval** — converged, triple-confirmed
(class-agnostic AP@0.5):

| source | overall | pup | nonpup |
|--------|---------|-----|--------|
| arisia mid-training rescore | 0.857 | 0.838 | 0.879 |
| arisia FINAL (ep29) rescore | 0.857 | 0.838 | 0.879 |
| aiq-gpu S (ep29, 4-GPU)     | 0.858 | 0.838 | 0.881 |

Pup AP is settled at **0.838** with tiled eval (vs 0.123 whole-image —
see [2026-06-06_gen005_small_object_floor.md](2026-06-06_gen005_small_object_floor.md)).
Hardware-independent and train-length-independent (plateaus early).

**single_sealion** (arisia): still training (~epoch 10/30).

**Scale-up: dinov3_x (50.3M, COCO 57.8)** — launched on aiq-gpu via slurm
(job 3), 640 tiles, tiled eval. One lever changed: backbone S->X. Training
clean on Blackwell (sm_120 deform ops fine), loss 60->29 by epoch 2, ETA
~6.5h. The question: does 5x capacity beat S's 0.858 overall / 0.838 pup,
or is the data (not the model) the ceiling? pup already converged at S, so
expect gains mostly in the harder cases if any.

## VRAM: huge headroom, and the "growth" is a watermark

dinov3_x at per_gpu_batch=4 on a 96GB Blackwell card uses **~5-8 GB
(~8%)**. We are massively under-utilizing; batch could go 16-32+ (scale LR
with it) for far faster epochs. Deferred to keep one-lever-at-a-time.

The "VRAM grows over a run" worry is the `max mem` field = a monotonic
high-water mark (`torch.cuda.max_memory_allocated`), not a leak — it
climbs to the single worst-case batch's peak (object-count + Mosaic
variability) and sticks. Fixed the readability: DEIMv2 MetricLogger now
also prints **`cur mem`** (`memory_allocated`, the current working set),
so steady-state footprint is visible per log line. DEIMv2 commit 3f5c5ef
(branch ddp-loss-key-alignment); kit bump 33080a4. Full analysis:
[2026-06-04_deimv2_training_internals.md](2026-06-04_deimv2_training_internals.md) sec 6.

## Tooling built this round

* **Tiled eval is now the project default** (paths.sh KCD_TILED_EVAL=True,
  KCD_EVAL_DEVICE=cuda). We train on tiles; whole-image eval measures the
  wrong thing.
* **Larger backbones wired** (dinov3 m/l/x): checkpoint paths, resolver,
  fetch_pretrained cases.
* **Auto-fetch** missing pretrained checkpoints in the host pre-flight
  (KCD_NO_AUTO_FETCH=1 to disable).
* **Host pre-flight** for all training inputs (checkpoint + corpus bundles
  + exact tile-cache hash via kcd_tile_hash) — fails on the host before
  docker, with fetch hints.
* **Standalone (no-slurm) launch path** (KCD_NO_SLURM=1 -> _run_standalone.sh)
  AND **slurm on aiq-gpu** (KCD_DOCKER_GPU_MODE=gpus for --gpus all;
  --gres=gpu:4 confirmed against typed gres gpu:rtxpro6000:4).
* **Symlinked tile cache auto-mounted** in both launch paths (aiq's SSD is
  symlinked into /data; the target wasn't mounted -> dangled in container).
* **Eval efficiency**: fp16 autocast (~2x on tensor cores), window
  batching (batch=64), threaded decode prefetch, per-phase timing
  (decode/predict/nms + window_infer/nms_merge), post-scoring phase
  progress (dump/filter/AP were silent over ~16k dets/image).
* **rsync_from_aiq.sh** (mirrors rsync_from_arisia; pulls slurm .out +
  standalone tee logs; excludes the large reproducible tile cache).

## Open / next

* **Multi-GPU eval** — eval still uses 1 of 4 GPUs (3 idle post-training).
  The ~4x win, but build it single-GPU-safe (default 1, opt-in count,
  auto-clamp) so namek rescoring is untouched. Gate on the timing
  breakdown confirming inference is the bottleneck.
* **Tiled eval emits ~16-18M detections** over the test set (~16k/image),
  making dump + AP slow. A per-image top-K cap or higher tiled score floor
  would speed it with negligible AP@0.5 impact — opt-in, tradeoff TBD.
* Decide whether X's result justifies keeping the capacity, or S is the
  efficient operating point and the data is the ceiling.
