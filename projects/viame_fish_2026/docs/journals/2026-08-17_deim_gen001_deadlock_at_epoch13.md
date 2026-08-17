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

Root cause of the deadlock itself is not yet established. It is not
reproducible from the logs alone; the next occurrence should be caught live
with `py-spy dump` on the container's rank-0 PID before anything is killed.
Candidates, in rough order of suspicion: a dataloader worker death under
`num_workers=8` + `persistent_workers` + `prefetch_factor=4`; a rank divergence
in batch count; or an NCCL transport stall.

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

## Where things stand

- `best_stg2.pth` (epoch 12, AP 0.5440) is safe on the NVMe.
- The job's 72 h walltime expires 2026-08-17 17:30 EDT, so slurm resolves the
  hang on its own today if nothing is done sooner.
- The curve is flat from epoch 7 (0.5422) to epoch 12 (0.5440), so the 7 missing
  epochs were unlikely to move the number much. Resuming is lower value than
  scoring what we have.

## Next

1. **Score epoch 12 on the held-out `Test/` split** (69 sequences, 33,434
   images, 84,694 annotations). That is the number that means something, and it
   is the same protocol RF-DETR can be scored under.
2. Capture `py-spy dump` if the deadlock recurs.
3. Reconcile `TORCH_NCCL_BLOCKING_WAIT` against the heartbeat timeout.
4. Rebuild the image; consider larger batch given the 6x memory headroom.
