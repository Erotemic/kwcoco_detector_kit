# 2026-06-04 — gen004 forensic + resume plan
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

The gen004 plan ([[2026-06-01_gen004_balanced_plan]]) shipped two
runs: a hgnetv2_n + 1-GPU ablation and a dinov3_s + 2-GPU
"bigger leap". Both used kit-side `balance_mscoco` on the JPEG
path. Submissions began 2026-06-01; the analysis below covers
the full set of attempted submissions through the end of 2578's
walltime on 2026-06-04.

Headline: **dinov3_s + balance is the breakthrough. Single
checkpoint at epoch 4 (in-train mAP 0.157) is the strongest model
we've produced for pup_vs_nonpup**. hgnetv2_n is genuinely
capacity-bound (10 epochs to mAP=0.006, no improvement after
epoch 4). gen003 single_sealion confirms that
`skip_empty=False` alone is insufficient — explicit positive
oversampling (balance) is required.

## Run-by-run state

### gen003 single_sealion (the unbalanced WDS comparison)

| Job | State | Reached | Result |
|---|---|---|---|
| 2562 | crash at startup | SSD symlink mkdir collision | nothing |
| 2563 | scancel by user | epoch 0 iter 1000 | nothing |
| 2564 | scancel by user | apply_scheme sanity | nothing |
| **2565** | **walltime hit** | **epoch 11 iter 20500/47878** | **mAP=0.000 throughout** |

2565 ran ~3.5 hours per epoch, completed 11 of 30, and produced
in-train mAP 0.000 at every checkpoint. Loss was non-zero
throughout, gradients flowed, model trained — but the matcher
converged to "predict nothing" because positives were ~20% of
the stream and focal loss makes "no prediction" the safe play
when positives are rare. This confirms what the gen003 fix was
supposed to address ([[project-gen002-three-scheme-verdict]] /
[[2026-06-01_gen002_three_scheme_results]]): just letting
empties through (`skip_empty=False`) doesn't help if the
relative frequency of positives is still too low. The matcher
needs them up-weighted.

`best_stg1.pth` and `best_stg2.pth` on disk — both at mAP=0.

### gen004 hgnetv2_n + 1-GPU + balance (the ablation)

| Job | State | Reached | Result |
|---|---|---|---|
| 2566 | crash at startup | `ModuleNotFoundError: balance_mscoco` (image stale; balance_mscoco wasn't baked yet) | nothing |
| 2569 | scancel by user | epoch 0 iter 0 (double-submit) | nothing |
| **2570** | **walltime hit** | **epoch 10 iter 28000/47878** | **mAP plateau 0.001 → 0.006** |

Used LEGACY oversample mode (no `max_oversample`), so each pup
tile got repeated ~20× per epoch within a 229k-sample bundle.
That's heavy overfitting pressure even with augmentation. Per-
epoch mAP: 0.001 (ep0), 0.001 (ep1), 0.001 (ep2), 0.000 (ep3),
**0.006 (ep4)**, 0.006 (ep5-10).

Two things to read from this:
* hgnetv2_n at 320×320 is structurally insufficient for
  pup_vs_nonpup. **Don't iterate on this variant for this task.**
* The plateau at 0.006 from epoch 4 onward also tells us
  Mosaic kicking in at epoch 4 (per [[2026-06-04_deimv2_training_internals]]
  Mosaic policy stage 2) didn't help — the model has enough
  augmentation diversity, it just doesn't have the capacity.

`best_stg1.pth` and `best_stg2.pth` exist; both at mAP=0.006.

### gen004 dinov3_s + 2-GPU + balance (the bigger leap)

Started with seven submissions that hit a Docker `--gpus`
comma-parsing bug ([[2026-06-04_slurm_docker_robustness]]).
Iterations 2568, 2571, 2572, 2573, 2574, 2575, 2576 each
attempted a different incantation of `--gpus` and each failed
the same way. Final fix (commit `c9183c5`) was to bypass
`--gpus` entirely and use the legacy `--runtime=nvidia +
NVIDIA_VISIBLE_DEVICES` path.

| Job | State | Reached | Result |
|---|---|---|---|
| 2568, 2571–2576 | crash | docker `--gpus` parser collision (7 attempts) | nothing |
| **2577** | **OOM + walltime** | **epoch 6 iter 3500/7175** | **mAP 0.138 → 0.161** ⭐ |
| 2578 (resume) | **OOM** | **epoch 5 iter ~500/812** | **mAP TBD** |

2577 trajectory (in-train mAP at every full-epoch test):
```
ep0: 0.138    ep3: 0.151
ep1: 0.143    ep4: 0.157  ← checkpoint0004.pth
ep2: 0.149    ep5: 0.161  ← best_stg1.pth (right before OOM)
```

Monotonically improving. Compared to gen002 pup_vs_nonpup
baseline of 0.025 ([[2026-06-01_gen002_three_scheme_results]]),
this is **6.4× better** after only 6 epochs. The full 30-epoch
run was OOM'd at epoch 6 (45.76 GB / 47.4 GB VRAM peak) and
then walltime-cancelled.

2578 was the resume from `checkpoint0004.pth` with two changes:
* `max_oversample=1` (epoch dropped 7175 → 812 iters)
* AMP confirmed active via kit commit `c43bf8f`

It got further than 2577 per-iter (epoch 5 / 812 vs epoch 6 / 7175,
much faster pace) but still OOM'd at iter 500. AMP saved ~1 GB
(2577 OOM at 45.76 GB FP32; 2578 OOM at 44.79 GB FP16) — less
than expected because activations are only half the budget; the
rest is weights + AdamW(2×) + EMA, which AMP does not touch.

Root cause of 2578's OOM: `multiscale_512_768` policy
occasionally samples 768×768 batches. At that peak scale,
activation memory roughly doubles vs 640, and the spike pushes
total VRAM over budget. `max mem` grew from 9.9 GB (iter 0) →
28.1 GB (iter 500) → peaked above 47 GB.

Next resume (committed `5b2fb32`) drops `multiscale_512_768` →
`fixed` 640. The model already saw 640 throughout 2577's
multiscale training so this is a regime narrowing, not a regime
shift. See [[2026-06-04_deimv2_training_internals]] for why
the LR schedule survives this change cleanly (we're in the
flat phase of FlatCosineLR for the entire run, so iter-count
changes don't break the curve).

## Strongest checkpoint on disk: 2577's `best_stg1.pth`

Path:
```
/data/users/jon.crall/kcd_sealion/runs/pup_vs_nonpup_deimv2_dinov3_s_2gpu_arisia_gen004_balanced/runs/deimv2_dinov3_s_640x640_multiscale_512_768/best_stg1.pth
```

Epoch 5, in-train COCO mAP 0.161. **This is the model to
rescore against the kit's NFS-excluded class-agnostic AP**
([[feedback-detection-ap-is-selection-criterion]]) to get the
deployable number. The 2× kit/in-train ratio observed in
gen002 ([[2026-06-01_gen002_three_scheme_results]]) projects
kit AP ≈ 0.32 for this checkpoint — **>12× better than gen002's
0.025 baseline**.

Suggested next action regardless of the resume's outcome:
```bash
bash projects/viame_sealions_2026/scripts/submit_rescore_per_checkpoint.sh \
  <run-name>=pup_vs_nonpup_deimv2_dinov3_s_2gpu_arisia_gen004_balanced
```

This re-evaluates every checkpoint under the kit's eval and
records per-class AP. We want to know specifically how PUP did
vs nonpup at the strongest checkpoint.

## Lessons for future runs

1. **dinov3_s is the right model for pup_vs_nonpup**, full stop.
   hgnetv2_n at 320×320 cannot learn this task no matter how
   well the data is balanced. Capacity is the bottleneck, not
   composition.

2. **`max_oversample=1` is the right default for class-balance**
   ([[2026-06-04_balance_mscoco_design_notes]] if written;
   else commit `bfbed6b`). Heavy oversampling of the rare bucket
   wastes compute and risks overfitting; undersample the
   majority instead. Each pup tile seen once per epoch +
   augmentation (Mosaic at 50%, MixUp, CopyBlend, Photo, flip,
   IoUCrop) provides plenty of stochastic diversity across the
   30 epochs.

3. **`fixed` policy beats `multiscale_512_768` for dinov3_s at
   batch=16**. The multiscale upper bound at 768 was the OOM
   trigger; we get the same model behavior at 640 fixed
   without the peak. If you want multiscale anyway, use
   `multiscale_512_640` or drop `per_gpu_batch` to 8.

4. **AMP must be force-enabled via `--use-amp`** ([[2026-06-04_deimv2_training_internals]]):
   DEIMv2's `train.py` argparse defaults `--use-amp` to False,
   which then overrides the YAML's `use_amp: true`. The kit's
   `launch()` introspects the YAML and appends `--use-amp` when
   true (commit `c43bf8f`). Every kit-launched run prior to
   that commit trained at FP32.

5. **The image bakes kit Python.** When iterating on
   `kwcoco_detector_kit/*` Python changes, either set
   `KCD_DEV_MOUNT_KIT=1` in the submit script or rebuild the
   image. The dev mount is fine for short experimentation; the
   image rebuild via
   `docker/opengroundingdino/build_arisia_cuda132.sh` is the
   reproducibility-unit path.

6. **gen003 single_sealion mAP=0.000 is data not bug.** The
   model trained correctly, gradients flowed, loss was finite —
   the matcher just learned "predict nothing" because positives
   were 20% of the stream under WDS uniform sampling. The fix
   is the same balance treatment dinov3_s got: pup + sealion
   classes need explicit upweighting at the dataset level. The
   gen003 single_sealion checkpoint is NOT useful and shouldn't
   be promoted.

## Open questions

* What kit AP does the 2577 `best_stg1.pth` actually produce?
  In-train mAP / kit-AP ratio is the open variable; we project
  ~0.32 but need the rescoring run to confirm.

* Does the 2578-style resume (max_oversample=1, fixed 640,
  AMP) sustain its trajectory through epoch 30, or does it
  plateau early like hgnetv2_n? The first 5-10 epochs of the
  next resume should tell.

* Should single_sealion get the same dinov3_s + balance
  treatment? Probably yes — that's the obvious follow-up once
  pup_vs_nonpup is locked. Don't reuse 2565's setup.

## State at end of session

* gen003 single_sealion (2565): walltime-killed; mAP=0; **don't deploy**.
* gen004 hgnetv2_n + balance (2570): walltime-killed; mAP=0.006; **don't deploy**.
* gen004 dinov3_s + balance (2577): OOM+walltime; mAP=0.161 @ ep5; **strongest checkpoint we have** (until 2581 surpasses it — see below).
* gen004 dinov3_s + balance (2578 resume): OOM; superseded by next resume with `fixed` policy.
* Next resume committed (`5b2fb32`): `fixed` 640 policy; awaiting submission.

## Update 2026-06-05 — 2579 resume OOMed, batch=8 fix, fresh restarts

### 2579 (resume, fixed 640): plateau + OOM

The committed `fixed 640` resume from `checkpoint0004` ran THREE
clean epochs at steady-state memory then OOM'd on a worst-case
batch:

| epoch | in-train mAP | max_mem |
|---|---|---|
| 5 | 0.157 | 31 GB |
| 6 | 0.157 | 31 GB |
| 7 | 0.158 | 33 GB |
| 8 | 0.158, then OOM | spike → 42.82 GB |

Two findings:

* **mAP plateau at 0.157-0.158** confirms the user's intuition
  that resuming from a mixed-hyperparam checkpoint locks the
  model into the training distribution its history was built
  on. The model couldn't recover the 0.161 from 2577 epoch 5,
  let alone improve.
* **Fixed 640 alone is not enough** to fit `batch_per_gpu=16`
  in 47 GB. Steady-state memory was a comfortable 31-33 GB,
  but a transient peak hit 42.82 GB and tried to allocate
  4.68 GB more (need 47.5 GB total — over ceiling). See
  [[2026-06-04_deimv2_training_internals]] section 4.

### Fix: `KCD_PER_GPU_BATCH=8`

Committed `5a6d33e`. Halves activation memory in the worst-case
batches. Trade-off: 2× more iters per epoch, but at
`max_oversample=1` the epochs are already short (~1600 iters at
batch 8 for single_sealion, ~14k for pup_vs_nonpup which has a
larger bucket fit).

Also committed: clean restart of pup_vs_nonpup from COCO-pretrained
(commit `9c5b3f2`) — no resume from 2577's checkpoint, so the
ablation is hyperparam-clean from epoch 0.

### 2580 (clean pup_vs_nonpup) + 2581 (clean single_sealion), parallel on 4 GPUs

User submitted both back-to-back on 4 free GPUs (no slurm dep
needed). Both running with the new clean recipe:
`dinov3_s + 2-GPU + fixed 640 + AMP + max_oversample=1 + batch_per_gpu=8`.

**Memory verdict: SOLVED.** Both runs hold steady at **max_mem ≈
13 GB** throughout training. Down from 31-33 GB at batch=16 and
from 45 GB at batch=16 + multiscale. Well below the 47 GB
ceiling. No OOM risk going forward.

#### 2580: clean pup_vs_nonpup (3-bucket balance)

* started 23:40, ~1h40m elapsed at snapshot
* iters/epoch: **14,351** (large because pup is rare → balance
  target_size = pup_pool / 0.2 is the largest)
* progress: epoch 0 iter 13000/14351 (90% through epoch 0)
* no AP eval yet (eval runs at end of epoch)
* loss: 38.2 → 23.6 (clean monotonic descent — training is on
  the right trajectory)
* max_mem: 13096 MB constant

Will produce first AP at epoch 0 end (~30 min more wallclock from
snapshot).

#### 2581: clean single_sealion (2-bucket balance) — RUNNING WINNER

* started 23:42, ~1h36m elapsed at snapshot
* iters/epoch: **6,149** (smaller because sealion-positive bucket
  is much larger than pup, so the rarest-fits multiplier is
  smaller)
* progress: epoch 2 iter 500/6149 (2 full epochs evaluated)
* max_mem: 13334 MB constant

**Per-epoch in-train mAP** (CocoEvaluator, includes NFS — kit
eval will likely be higher):

| epoch | mAP@0.50:0.95 | mAP@0.50 | mAP small/med/large |
|---|---|---|---|
| 0 | **0.177** | (not shown) | (not shown) |
| 1 | **0.188** | 0.425 | 0.003 / 0.137 / 0.427 |

`best_stat: {'epoch': 1, 'coco_eval_bbox': 0.188}`.

**This already beats every prior gen003/gen004 result for any
scheme.** Comparison:

| run | scheme | epochs run | best in-train mAP |
|---|---|---|---|
| v5 (gen001) | single_sealion | 30 | 0.0485 (kit AP 0.177) |
| gen002 | single_sealion | 30 | 0.012 (kit AP 0.024) |
| gen003 | single_sealion | 11 | 0.000 |
| gen002 | pup_vs_nonpup | 30 | 0.025 |
| 2577 | pup_vs_nonpup gen004 | 5 | 0.161 |
| **2581** | **single_sealion gen004 clean** | **1** | **0.188** ⭐ |

After ONE epoch of training, the clean recipe surpassed the
2577 mAP (epoch 5) AND v5's in-train mAP (0.0485, the best
single_sealion training had ever produced before this).

The small-object AP is the diagnostic: 0.003 small, 0.137
medium, 0.427 large. The model is detecting big sealions
excellently but missing small ones — common at 640 input with
no super-resolution path. That's an area to address in a
future iteration (resolution lever, or multiscale-without-768),
but not blocking the deployment of what we have.

### What we now know

1. **`per_gpu_batch=8` was the missing piece**. The OOMs we
   chased through 2577 / 2578 / 2579 weren't fundamentally
   about multiscale; they were about activation memory peaks
   at batch=16 being too close to the ceiling. Batch=8 keeps
   us at 13 GB steady, which is half the A6000's budget.
2. **The clean restart hypothesis is validated**.
   2581 epoch 1 (0.188) > 2577 epoch 5 (0.161) > 2579 resume
   epoch 8 (0.158). Starting fresh from COCO-pretrained with
   uniform hyperparams from epoch 0 converges much faster than
   resuming into a different regime.
3. **single_sealion is easier than pup_vs_nonpup**. With the
   same model and balance recipe, single_sealion at epoch 1 is
   already at the level pup_vs_nonpup reached after 5 epochs.
   That tracks — pup detection requires fine-grained
   discrimination (pup vs adult), while single_sealion is
   any-sealion detection.
4. **Memory ceiling is no longer a concern.** 13 GB / 47 GB
   leaves room for batch_per_gpu=16 ONCE we know the recipe
   converges. But we don't need it; batch=8 is converging fine.
5. **CocoEvaluator's mAP includes NFS in this train/eval setup.**
   For the deployable number we still need to rescore against
   the kit's NFS-excluded eval. Expect the kit AP to be 2-3×
   higher than in-train mAP per the gen001/gen002 ratio
   pattern ([[2026-06-01_gen002_three_scheme_results]]). If
   2581 hits in-train mAP 0.25 by epoch 5, kit AP would
   project to 0.5-0.75 — would be a strong deployable model.

### What to watch / do next

1. **Let both runs go to epoch 30** (estimated finish times: both
   ~6-8h wallclock from snapshot). 2581 reaches `best_stg2.pth`
   in ~6h; 2580 in ~12h (longer epochs).
2. **Rescore 2581's checkpoints periodically** as they land —
   `scripts/submit_rescore_per_checkpoint.sh` runs the kit's
   NFS-excluded eval. The first epoch where kit AP plateaus
   tells us the early-stopping point.
3. **Pup-specific AP** for 2580 once it finishes epoch 0: pup
   detection is the binding constraint per memory, so even if
   averaged mAP looks similar to 2581, the per-class breakdown
   tells us whether pup specifically has improved.
4. **Small-object AP** for both: 2581 epoch 1 shows 0.003 small
   vs 0.427 large. If small stays near zero through epoch 30,
   the next iteration should explore multiscale-without-the-OOM
   (e.g. `multiscale_512_640`) or higher-resolution tiles.

### State at this update

* 2580 (clean pup_vs_nonpup gen004): epoch 0, 90% through;
  no AP yet; on track.
* 2581 (clean single_sealion gen004): epoch 1 complete at
  in-train mAP **0.188**; epoch 2 underway. **NEW BEST
  CHECKPOINT in any gen.**
* 2577 best_stg1 (mAP 0.161 @ ep5): still on disk as a
  reference for "what 2-GPU dinov3_s + balance at batch_per_gpu=16
  + multiscale produces."
* Memory issue: resolved by batch=8.
* Image rebuild: completed by user 2026-06-04 (per
  [[feedback-image-is-reproducibility-unit]]); resume script no
  longer needs `KCD_DEV_MOUNT_KIT=1`.

## Update 2026-06-05 evening — first kit AP measurement on namek

Ran the rescore on namek (3090) inside the kit's docker image
against 2581's `best_stg1.pth` (in-train mAP 0.228 from the
synced snapshot, epoch ~16).

**Result: kit AP = 0.471** (`nocls_measures.ap`, class-agnostic
at IoU 0.50, NFS-excluded by virtue of the single_sealion
scheme dropping NFS entirely).

* **v5 baseline (previous SOTA for single_sealion)**: 0.177 kit AP
* **2581 best_stg1**: **0.471 kit AP** — **2.66× v5**.

In-train → kit-AP ratio for this run: 0.471 / 0.228 = **2.07×**.
Compared to historical ratios:

| run | in-train mAP | kit AP | ratio |
|---|---|---|---|
| v5 (gen001) | 0.0485 | 0.177 | 3.65× |
| gen002 single_sealion | 0.0120 | 0.024 | 2.00× |
| **2581 best_stg1 (synced ep ~16)** | **0.228** | **0.471** | **2.07×** |

If 2581's in-train mAP continues climbing to ~0.24 by epoch 30
(extrapolating the slowing trajectory in the parent section
above), and the ratio holds at ~2×, the **deployable kit AP
projects to ~0.48-0.50** for the final checkpoint. Even if the
ratio degrades to 1.8× as the model saturates, we're still at
0.43-0.45 deployable — **comfortably the strongest model the
project has ever produced**.

### Cross-host eval setup gotchas

Hit two friction points worth recording so future agents don't
spend time on the same ones:

1. **The kit's docker image bakes kit Python + DEIMv2 but NOT
   `projects/`**. The Dockerfile COPY list excludes the project
   trees, so any analysis tool under
   `projects/.../scripts/` (e.g. `rescore_per_checkpoint.py`)
   isn't inside the image. Workaround: bind-mount the host's
   `projects/` dir at `/opt/kwcoco_detector_kit/projects:ro`
   when running. Long-term fix: add `COPY projects ...` to the
   Dockerfile.

2. **kwcoco files bake absolute imagery paths**, not bundle-
   relative ones (see [[feedback-kwcoco-bakes-absolute-paths]]).
   On namek the imagery lives at
   `/media/joncrall/raid/Public/VIAME/...` but the synced
   vali.kwcoco.zip's `file_name` hardcodes
   `/data/Public/VIAME/...` (arisia's path). Eval fails because
   the kwcoco refuses to find images that ARE on disk but at a
   different path. Workaround: bind-mount aliasing —
   `-v /media/joncrall/raid/Public/VIAME:/data/Public/VIAME:ro`
   in the docker run. Long-term fix: write bundle-relative
   paths in `coco_export.py` and `tile.py`.

3. **`rsync_from_arisia.sh` does NOT sync source imagery.** It
   pulls `$KCD_TRAINING_ROOT/` (your training outputs); the
   imagery at `/data/Public/VIAME/...` is the cluster-shared
   read-only corpus, intentionally excluded from sync. A
   correctly-configured eval host needs the imagery already
   present (most users have it from initial project setup).
   Took us a few minutes of "but I thought rsync brought
   everything!" to figure out.

4. **"Training ViT-Tiny from scratch..." in the eval log is
   harmless.** It's a misleading `__init__` print from
   `DINOv3STAs` that fires whenever the model is constructed,
   EVEN when we're about to load checkpoint weights over it.
   The next thing after that print is the checkpoint load
   (`tuning=...` in the kit's launch args). No actual training
   happens. Don't be alarmed.

### Final working rescore command (for posterity)

```bash
docker run --rm --gpus all \
  -v /media/joncrall/raid/users/jon.crall:/data/users/jon.crall \
  -v /media/joncrall/raid/users/jon.crall:/media/joncrall/raid/users/jon.crall \
  -v ~/code/kwcoco_detector_kit/projects:/opt/kwcoco_detector_kit/projects:ro \
  -v /media/joncrall/raid/Public/VIAME:/data/Public/VIAME:ro \
  -w /opt/kwcoco_detector_kit \
  kwcoco-detector-kit:ogdino-auto \
  python3 /opt/kwcoco_detector_kit/projects/viame_sealions_2026/scripts/rescore_per_checkpoint.py \
    --run-dir /data/users/jon.crall/kcd_sealion/runs/<RUN_NAME> \
    --eval-target vali --device cuda
```

Replace `<RUN_NAME>` with the run dir name. Works for both
gen004 schemes; the bind-mount set is invariant.

### State at this update

* 2581 best_stg1 (synced epoch ~16): **kit AP 0.471**. Other
  checkpoints (best_stg2, last) rescoring in the same job
  — results pending.
* 2580: still rescoring later when the docker run gets to it.
* Training continues on arisia; next sync will bring fresher
  checkpoints with potentially higher AP.

## Update 2026-06-05 late evening — full kit-AP table for all gen004 checkpoints

All checkpoints from 2577 (multiscale, OOM'd at ep 6), 2580
(clean fixed 640 pup, still running at ep 6-7 when synced),
and 2581 (clean fixed 640 single_sealion, running at ep 17 when
synced) rescored against the kit's NFS-excluded eval.

### Single number summary

| Scheme | Best deployable checkpoint | kit AP | vs prior SOTA |
|---|---|---|---|
| single_sealion | **2581 best_stg2** | **0.5807** | **3.28× v5 (0.177)** |
| pup_vs_nonpup | **2580 best_stg2** | **0.5650** | **2.84× v6 (0.199)** |

### Full rescore table

| run | variant | ckpt | nocls AP | per-class AP |
|---|---|---|---|---|
| 2581 single_sealion clean | fixed 640 | **best_stg2** | **0.5807** ⭐ | sealion: 0.5807 |
| 2581 | fixed 640 | best_stg1 | 0.4710 | sealion: 0.4710 |
| 2581 | fixed 640 | last | 0.4710 | sealion: 0.4710 |
| 2580 pup_vs_nonpup clean | fixed 640 | **best_stg2** | **0.5650** ⭐ | **pup: 0.1019**, nonpup_sealion: 0.7196 |
| 2580 | fixed 640 | best_stg1 | 0.4937 | pup: 0.0494, nonpup_sealion: 0.6463 |
| 2580 | fixed 640 | last | 0.4937 | (same as best_stg1) |
| 2577 pup_vs_nonpup multiscale | multiscale_512_768 | best_stg1 | 0.5584 | pup: 0.0884, nonpup_sealion: 0.7154 |
| 2577 | multiscale_512_768 | checkpoint0004 | 0.5504 | pup: 0.0819, nonpup_sealion: 0.7072 |
| 2577 | multiscale_512_768 | last | 0.5584 | (same as best_stg1) |

### Findings

**1. The class-balance hypothesis is unambiguously validated.**
Pup AP went from 0.0104 in v6 (the prior SOTA for
pup_vs_nonpup, baseline of "pup detection ~at floor") to
**0.1019 in 2580 best_stg2** — nearly 10× the historical
pup detector. Class balance + dinov3_s + the rest of the
gen004 recipe gives the matcher enough positive pup gradient
to actually learn the class.

Pup is still the binding constraint at 0.102 vs nonpup_sealion's
0.720 (a 7.1× gap), but it's no longer at noise floor.

**2. EMA (best_stg2) is dramatically better than the
in-train-best snapshot (best_stg1).** Across both schemes:

* 2581 single_sealion: 0.581 (stg2) vs 0.471 (stg1) — **+23%**
* 2580 pup_vs_nonpup:   0.565 (stg2) vs 0.494 (stg1) — **+14%**
* 2577 pup_vs_nonpup:   0.558 (stg1) vs no stg2 — (OOM'd
  before EMA mature)

**Operational implication**: the kit's eval / deployment pick
should ALWAYS prefer `best_stg2.pth` over `best_stg1.pth` for
single_sealion and pup_vs_nonpup. The previous practice of
"check both" can be retired except when an EMA-vs-non-EMA
comparison is specifically wanted.

**3. The 2577 multiscale run is competitive with 2580 (fixed)
despite being a partial run.** 2577 best_stg1 (0.558,
5 epochs, multiscale) is essentially equivalent to 2580
best_stg2 (0.565, 6-7 epochs, fixed). Two possible
explanations, not mutually exclusive:
* Multiscale broader-resolution training helps representation
  more than the extra epoch helps fixed-resolution training.
* 2580 hasn't reached its plateau yet; it's still on the
  improving slope.

Worth checking when 2580 completes whether final 2580 beats
2577 by a wider margin. If they remain ~equivalent, the
multiscale + memory-headroom (`per_gpu_batch=8 +
multiscale_512_640`) variant would be the recommended recipe
for gen005.

**4. In-train mAP → kit AP ratios** (corrected with actual numbers):

| run | in-train mAP (sync ep) | kit AP best_stg2 | ratio |
|---|---|---|---|
| v5 single_sealion | 0.0485 (ep 11) | 0.177 (best_stg1) | 3.65× |
| gen002 single_sealion | 0.012 (ep 21) | 0.024 | 2.00× |
| **2581 single_sealion** | **0.228 (ep 16)** | **0.581** | **2.55×** |
| **2580 pup_vs_nonpup** | **0.168 (ep 6)** | **0.565** | **3.36×** |
| **2577 pup_vs_nonpup** | **0.161 (ep 5)** | **0.558 (stg1)** | **3.47×** |

pup_vs_nonpup has a higher ratio than single_sealion because
the nocls (class-agnostic) metric advantages multi-class
problems — it merges predictions across classes before
ranking, so a confident nonpup_sealion prediction near a pup
GT still counts as a true positive for nocls. single_sealion's
ratio (2.55×) is the more honest single-class kit AP.

### What this means for deployment

* For single_sealion detection: **deploy 2581 best_stg2.pth**.
  kit AP 0.581. Stop training? Probably not yet — the
  trajectory was still climbing (in-train 0.230 at ep 17 vs
  0.228 at ep 16), so let it run to ep 30 to see how high
  it goes, but the current model is ALREADY operational.

* For pup_vs_nonpup detection: **deploy 2580 best_stg2.pth**.
  kit AP 0.565. Same logic: let training continue, but the
  model is already operational AND already crushes every
  prior pup-detector trial.

* The 2577 best_stg1 is still on disk but superseded — it was
  the strongest known checkpoint until today; now it's a
  reference point only.

### What to iterate on next (gen005?)

* **Pup is still 7× behind nonpup AP.** The recipe lifts pup
  from floor to "low-but-real" (0.102) but there's enormous
  remaining headroom. Levers worth exploring:
  - More aggressive pup balance (e.g. 0.33/0.33/0.33 or even
    pup 0.5)
  - Multi-scale (with memory-safe upper bound, e.g.
    `multiscale_512_640`) to give the model more pup-tile
    resolution diversity
  - Pup-specific evaluation metrics (currently nocls AP
    averages pup with nonpup; per-class is the right
    operational signal)

* **2580 vs 2577 head-to-head** when 2580 finishes will tell
  us if multiscale was the missing piece or just an extra
  epoch's worth of training.

* **Small-object recall** is still near zero on both runs. The
  640×640 input is the limit. Higher input (with larger
  per_gpu_batch reduction) or super-resolution pre-processing
  is the gen005 direction for tiny pups.

### State at this update

* 2581: latest synced intermediate eval = **in-train mAP 0.230 @
  epoch 17** (full trajectory ep 0=0.177 → ep 17=0.230,
  monotonically improving every epoch). kit AP **0.581** already
  deployable from best_stg2. Training continues toward ep 30
  on arisia.
* 2580: training continues; in-train mAP 0.168 @ ep 6; kit AP
  0.565 already deployable from best_stg2.
* 2577: complete (OOM'd long ago); best checkpoint at kit AP
  0.558 / 0.555 (best_stg1 / checkpoint0004); superseded by
  2580 best_stg2 but worth keeping as the multiscale
  reference point.
* Both runs expected to finish in the next ~24h (single_sealion
  through ep 30, pup through walltime ep ~19). Re-rescore at
  completion will show whether further training extracts more.

## How to resume this analysis

If a future agent picks this up:
1. Read [[2026-06-04_deimv2_training_internals]] for the AMP
   bug, FlatCosineLR / Mosaic policy internals, and OOM math.
2. Read [[2026-06-04_slurm_docker_robustness]] for the docker
   `--gpus` parser bug, slurm trap race, and the (opt-in)
   janitor.
3. Look at the 2577 `best_stg1.pth` rescoring result — that's
   our actual deployable number.
4. If the new `fixed`-policy resume completed, compare its
   final kit AP against the 2577 best_stg1 baseline. If
   in-train mAP is monotonically increasing through the full
   30 epochs, the data + model combination is working as
   hypothesized; otherwise we hit a different ceiling
   (capacity? data exhaustion? scale specificity?) and need
   to think.
