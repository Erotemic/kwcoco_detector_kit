# DEIM gen001: 13 good epochs, then a DDP deadlock (2026-08-17)

First DEIMv2 training run on FishTrack23. It produced a usable model and then
hung, alive but making no progress, for two days. Both halves are worth
recording.

Run: `fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001`, slurm job **293**, started
2026-08-14 17:31 EDT (21:31 UTC) on aiq-gpu.

## Outcome

**13 of 20 epochs completed. Best checkpoint: epoch 12, vali box AP@50:95 =
0.5440**, saved as `best_stg2.pth` (819 MB) and intact on disk.

Full epoch-12 evaluation on the held-out, sequence-disjoint vali split:

| metric | value |
|---|---|
| AP@50:95 (all) | **0.5440** |
| AP@50 | 0.8060 |
| AP@75 | 0.5877 |
| AP small / medium / large | 0.184 / 0.252 / 0.657 |
| AR@100 (all) | 0.631 |
| AR small / medium / large | 0.304 / 0.388 / 0.736 |
| AR@50 | 0.900 |

Training curve — steady, with one violent excursion:

| epoch | AP | AP50 | AP75 | train_loss |
|---|---|---|---|---|
| 0 | 0.5280 | 0.7855 | 0.5692 | 35.98 |
| 1 | 0.5318 | 0.7889 | 0.5726 | 32.62 |
| 2 | 0.5382 | 0.7977 | 0.5784 | 31.59 |
| 3 | 0.5388 | 0.7994 | 0.5784 | 30.79 |
| **4** | **0.0000** | **0.0000** | **0.0000** | **27.02** |
| 5 | 0.5363 | 0.8044 | 0.5780 | 36.66 |
| 6 | 0.5393 | 0.8050 | 0.5819 | 35.57 |
| 7 | 0.5422 | 0.8070 | 0.5844 | 35.09 |
| 8 | 0.5416 | 0.8042 | 0.5842 | 34.71 |
| 9 | 0.5335 | 0.7997 | 0.5766 | 36.78 |
| 10 | 0.5388 | 0.8016 | 0.5821 | 35.60 |
| 11 | 0.5409 | 0.8007 | 0.5842 | 35.06 |
| **12** | **0.5440** | 0.8060 | 0.5877 | 34.69 |

Pace was ~1.4 h/epoch (10,464 steps at ~0.42 s/step), so 20 epochs would have
finished in ~29 h.

### This number is NOT comparable to RF-DETR's 0.4429

Both are single-class `fish` box AP@50:95, and it is tempting to read
0.544 > 0.443 as a win. It is not a like-for-like comparison:

- **Ours** is measured on 46 whole sequences held out by deployment, 35,111
  images, that the model never saw.
- **RF-DETR's** was measured on 4,000 720px chips carved frame-level out of its
  own training sequences, i.e. near-duplicates of training data (and its
  `valid/` and `test/` files are byte-identical).

Different denominators, and the baseline's is the easier one. The only honest
comparison is both models scored on the corpus's own `Test/` split, which is
still to do.

## The failure: a spinning DDP collective, not a crash

Last progress line: **epoch 13, step 1500/10464, 2026-08-15 16:43 UTC**. Then
nothing on disk anywhere for two days. Epoch 12's eval had completed cleanly
minutes earlier (`best_stat: {'epoch': 12, 'coco_eval_bbox': 0.5440}`).

State at 2026-08-17 09:19 EDT, ~2d16h into the job:

```
squeue    293  R  2-15:48:53   aiq-gpu
docker    kcd-293-...  Up 2 days
nvidia-smi  all 4 GPUs: 100% util, 78-81W / 300W, ~16.5 GB each
```

**100% utilization at a quarter of the power cap is the signature of a spinning
synchronization primitive, not computation.** NCCL busy-waits on collectives, so
a deadlocked all-reduce pegs the SM occupancy counter while doing no work. A
genuinely training GPU sits near the power cap. No traceback, no OOM, no NCCL
watchdog abort — the process is alive and waiting forever.

### Why the watchdog never fired

`_sbatch_train.sh` sets both:

```
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
TORCH_NCCL_BLOCKING_WAIT=1
```

These fight each other. `TORCH_NCCL_BLOCKING_WAIT=1` makes collectives block
synchronously in the calling thread, which is precisely the case the async
watchdog cannot preempt — so the 600 s heartbeat timeout that should have
aborted the job after ten minutes never got the chance, and a ten-minute
failure became a two-day one. Worth revisiting for every project using this
launcher, not just fish.

### py-spy: all four ranks stopped in the same frame

Caught live before the job was touched. `py-spy dump` against each rank's host
PID (visible from `nvidia-smi`; the container's PID 1 is host PID 4077820):

```
Thread 210 (active): "MainThread"
    _engine_run_backward (torch/autograd/graph.py:913)
    backward (torch/autograd/__init__.py:395)
    backward (torch/_tensor.py:633)
    train_one_epoch (engine/solver/det_engine.py:68)
    fit (engine/solver/det_solver.py:88)
```

**Identical on all four ranks.** Rank 0 additionally shows four unnamed
`(active)` native threads — the spinning NCCL/CUDA threads that produce the
100%-utilization-at-low-power reading — and eight idle `QueueFeederThread`s,
one per dataloader worker.

That is conclusive, and it eliminates the candidates worth eliminating:

- **Not a rank divergence.** All four are in the same collective, at the same
  call site. A divergence would show ranks in different frames.
- **Not a dataloader deadlock.** Every rank is past data loading and inside
  backward; the feeder threads are idle because there is nothing to feed a
  process that never asks for the next batch.
- **Not a dead rank.** All four processes are alive and spinning.

What remains is a genuine NCCL collective stall: every rank entered the same
gradient all-reduce during backward and it never returned. Given the stack —
Blackwell RTX PRO 6000, driver 610.43.02, CUDA UMD 13.3, torch nightly cu132 —
a transport- or driver-level hang is the most plausible explanation, and not a
surprising one on hardware and drivers this new.

`py-spy dump --native`, which would have named the exact NCCL call, fails with
`UNW_EBADREG: bad register number` — it cannot unwind torch's optimized
binaries. Not worth chasing.

### The fix bounds the damage, it does not prevent the stall

We cannot fix an NCCL/driver hang from here. We can stop it costing two days.

`_sbatch_train.sh` already sets `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` and
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600`, which exist to abort precisely this
failure. They were defeated by its `TORCH_NCCL_BLOCKING_WAIT=1` default.
The knob is parameterized, so the fish submit script now sets
`KCD_NCCL_BLOCKING_WAIT=0`; a future stall should abort after ten minutes,
leaving a resumable per-epoch checkpoint instead of a dead weekend.

Scoped to this project on purpose. The same default sits in the shared
launcher and the sea-lion runs carry the same exposure, but changing shared
infrastructure should be a deliberate act, not a side effect of this run.

## Two configuration findings for the next run

**Memory was massively over-provisioned.** Actual usage is **~16.5 GB of 96 GB
per GPU** at batch 6 / 1024px. The submit script predicted ~49 GB by
area-scaling the sea-lion 1280px measurements — off by 3x, because that model
was fed dense aerial tiles rather than whole frames. There is room for batch
16-24 per GPU, which would cut epoch time substantially, though it needs an LR
rescale and the epoch-4 event below argues for care.

**Epoch 4 diverged to NaN and recovered.** Tensor dumps of all-NaN outputs run
03:35 -> 05:47 UTC on 2026-08-15; AP collapsed to exactly 0.0000, train_loss
dipped to 27.02 (loss on garbage), then epoch 5 came back at 0.5363 and the run
improved steadily to epoch 12. So the optimizer recovered on its own, but
**LR = 1e-3 is at the edge of stable here.** This is the same LR that made
sea-lion gen006 X@640 diverge, where it did *not* recover. Do not raise it when
increasing batch size; if anything the sqrt-scaling rule that produced 1e-3
deserves re-derivation for this corpus.

## Provenance caveat

Job 293 launched 13 minutes after job 292 failed, which is not enough time for
an image rebuild — so **this run used the stale image**, baked kit
`e48ca1dab2a4-dirty` from 2026-06-21, 44 commits behind. That image predates
the kwconf migration, which is why its CLI works at all (old kit code +
scriptconfig).

Consequences: the result is reproducible only against that image, not against
the current source tree, and the `-dirty` tag means it is not reproducible from
git at all. The `/out` gitignore fix (6e81493) and the Dockerfile kwconf fix
(21b897f) are both in place, so a rebuild now yields a clean, traceable SHA.
**Rebuild before gen002.**

Also worth noting the startup log records no kit SHA — `IMAGE=` is printed but
not the baked commit. The Dockerfile bakes `kcd.kit_sha` as a label, so the
launcher could echo `docker inspect` output and make every run log
self-describing. Cheap, and it would have removed all doubt here.

## The held-out number

Scored the epoch-12 checkpoint against the corpus's own `Test/` split — 69
sequences, 33,434 images, 84,694 annotations that **neither** this model nor the
RF-DETR baseline has ever seen. This is the first honest generalization measure
any fish detector on this corpus has had.

| metric | value |
|---|---|
| **AP @ IoU=0.5 (`fish`)** | **0.7272** |
| AUC | 0.8504 |
| true positives | 84,694 |
| predictions scored | 10,041,803 |

Note the threshold: the kit's eval runs `--iou_thresh 0.5`, so this is an
**AP50**, not COCO's AP@50:95. Compare like with like:

| model | split | AP50 | what the split is |
|---|---|---|---|
| DEIMv2 e12 | vali | 0.8060 | held-out sequences, deployment-grouped |
| **DEIMv2 e12** | **test** | **0.7272** | held-out sequences, never seen by either model |
| RF-DETR | its own "val" | 0.7166 | 4,000 chips carved frame-level from its own training sequences |

Two things worth reading carefully here.

**The vali -> test drop (0.806 -> 0.727) is the honest generalization gap** and
is exactly what a sequence-disjoint protocol is supposed to expose. A
frame-level split would have hidden it.

**DEIMv2 is ahead of RF-DETR's number while being measured on strictly harder
data**, which is suggestive but still not a clean head-to-head: 0.7166 was
computed on near-duplicates of that model's own training frames. The comparison
only becomes real when RF-DETR is run over the same `Test/` bundle, which needs
a VIAME inference pass and a reader for its output CSV. Until then, report the
DEIMv2 figure on its own terms and cite RF-DETR's with its caveat attached.

### The eval nearly cost another 31 minutes

The first scoring attempt ran inference over all 33,434 images in 31 minutes
(22.7 Hz, 56 ms/img), dumped a valid 223 MB `pred_boxes.kwcoco.zip`, and then
died in the last 5% — inside the *pred-side* bbox filter, where the `safer`
atomic-write helper called `shutil.copymode` on a temp file that no longer
existed. Not disk space (469 GB free), not the model: a post-processing bug.

The GPU work was recoverable because the predictions file was already complete
and intact (verified: 33,434 images, 10,030,200 annotations, zip not corrupt).
Scoring finished from it directly, skipping the failing filter — safe because
that filter only drops annotations lacking a length-4 bbox and every detection
this model emits has one (0 of 200,000 sampled without).

`_launch_export_score.sh` now checks for a complete predictions file before
doing anything expensive and scores directly when one exists, so the same bug
cannot cost the time twice.

Also measured in passing: **exactly 300 predictions per image**, median score
0.0107, only 0.5% above 0.05. That is `run_kwcoco_eval`'s `score_thresh=0.001`
default filling a 300/image cap — correct for faithful AP, and the reason the
artifact is 223 MB and this write path is fragile at all.

## Where things stand

- `best_stg2.pth` (epoch 12, AP 0.5440) is safe on the NVMe.
- The job's 72 h walltime expires 2026-08-17 17:30 EDT, so slurm resolves the
  hang on its own today if nothing is done sooner.
- The curve is flat from epoch 7 (0.5422) to epoch 12 (0.5440), so the 7 missing
  epochs were unlikely to move the number much. Resuming is lower value than
  scoring what we have.

## Next

1. ~~Score epoch 12 on the held-out `Test/` split.~~ **Done: AP50 0.7272.**
2. ~~Reconcile `TORCH_NCCL_BLOCKING_WAIT` against the heartbeat timeout.~~
   **Done:** `KCD_NCCL_BLOCKING_WAIT=0` in the gen001 submit script.
3. **Rebuild the image before exporting.** The baked kit (`e48ca1dab2a4`,
   2026-06-21) has no `export-onnx`, `bench` or `parity` subcommands
   registered — they appear in its `--help` text but were never wired up. So
   the current image can score and package a checkpoint, but cannot produce an
   ONNX. A rebuild also picks up the kwconf fix and a clean, non-`-dirty` SHA.
4. **Package and hand off** (see below). Weights + metrics package first; add
   the ONNX after the rebuild.
5. **Resume epochs 13-19** via `submit_resume_*.sh` — optional, since the curve
   was flat from epoch 7.
6. Score RF-DETR on the same `Test/` bundle to make the comparison real. Needs
   a VIAME inference pass and a reader for its alternating class/score output.
7. Capture `py-spy dump` if the deadlock recurs — the one thing we still cannot
   explain is why the all-reduce stalled.

## Handoff package

`_launch_export_score.sh` gained a fourth stage that calls the kit's
`package-build`, producing a self-describing directory: the checkpoint under
`weights/`, `labels.json`, the provenance block, and — via `--metrics` — the
`detect_metrics.json` above, so a recipient can read what it scored without
being told separately. The ONNX is copied in only when one exists, which is
what makes this useful on an image that cannot export: the package is
weights-plus-evidence today and gains a deploy graph after the rebuild.

What to state when handing it over, because none of it is visible from the
files alone:

- Single class `fish`, box-only, folded through the corpus's own
  `Train/labels.txt` — the same file the RF-DETR model used, so the two are
  label-compatible by construction.
- Input 1024x1024, whole frames, no tiling.
- **Trained 13 of 20 planned epochs**, stopped by an infrastructure deadlock,
  not by convergence or divergence. Flat from epoch 7 (0.5422) to 12 (0.5440),
  so close to converged but not the full recipe.
- **AP50 0.7272 on 69 sequences it has never seen.** Cite the split, not just
  the number.
- Built from a stale image; reproducible against that image, not from git.
