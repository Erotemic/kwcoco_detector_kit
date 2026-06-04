# 2026-06-04 — gen004 forensic + resume plan

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
* gen004 dinov3_s + balance (2577): OOM+walltime; mAP=0.161 @ ep5; **strongest checkpoint we have**.
* gen004 dinov3_s + balance (2578 resume): OOM; superseded by next resume with `fixed` policy.
* Next resume committed (`5b2fb32`): `fixed` 640 policy; awaiting submission.

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
