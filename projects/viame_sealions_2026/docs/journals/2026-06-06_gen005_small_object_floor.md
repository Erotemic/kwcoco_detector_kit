# gen005: the bottleneck is small-object eval resolution, not capacity

2026-06-06. First clean gen005 runs on the corrected v2 splits
(full 2007-2024 corpus). Both trained past the PIL-truncated-image
crash (the `LOAD_TRUNCATED_IMAGES` patch held) and are progressing
normally. The mid-run COCO eval reveals a decisive, scheme-independent
pattern that should drive what we test on the aiq-gpu Blackwell box.

## Runs

- **2589** pup_vs_nonpup, dinov3_s, 2-GPU, 640px, balanced. At epoch 18:
  overall mAP 0.134 and still climbing (started 0.072). max mem 11.1 GB.
- **2590** single_sealion, same recipe. At epoch ~4: mAP 0.155.

Both healthy. Numbers are NOT comparable to gen004 (gen004 trained on
the wrong training_ready_v1 subset — 1314 imgs, 2021-2024, easier;
gen005 is the full corpus with NFS in test). See
[2026-06-05_corpus_audit_wrong_bundle.md](2026-06-05_corpus_audit_wrong_bundle.md).

## The finding: AP is gated entirely by small objects

COCO area-stratified AP at the latest evals:

| run / scheme | AP-small | AP-medium | AP-large |
|--------------|----------|-----------|----------|
| 2589 pup_vs_nonpup | **0.005** | 0.115 | 0.30–0.39 |
| 2590 single_sealion | **0.001** | 0.12–0.17 | 0.34–0.47 |

AP-small has been pinned near zero for all 18 epochs of 2589 while
AP-large is healthy and rising. The model is **not capacity-bound** —
it detects large sea lions well. It is **blind to small ones**, and
since pups are the small objects, this is the pup problem restated
in COCO terms. Capacity (a bigger backbone) is the wrong lever; the
large-object AP proves dinov3_s has plenty of representational power.

## Root cause: train/eval resolution mismatch

- **Training** runs on 640px **tiles** cut from full-resolution aerials,
  so a ~46px-median pup appears at its native size and is learnable.
- **Eval** runs **whole-image**: each multi-thousand-pixel aerial is
  `@Resize`d down to the 640px model input, then inference runs once.
  Confirmed by the batch count — `Test: [66/67]` over 1161 test images
  (~17 imgs/batch), i.e. one forward pass per whole image, not per tile.

At whole-image 640, a 46px pup in a (say) 4000px-wide aerial shrinks to
~7px — below what the detector can localize. So pups effectively vanish
at eval time even though the model saw them at training time. AP-small
≈ 0 is the direct consequence. This mismatch suppresses the headline
mAP and is almost certainly understating real operational capability.

## Implications for the aiq-gpu (Blackwell) runs

The aiq-gpu box has 4× RTX PRO 6000 Blackwell, **96 GB/GPU** (vs arisia
~40 GB). Spend that VRAM on **effective small-object resolution**, not
on a bigger model:

1. **Higher input resolution (1280, maybe 1536) for train AND eval.**
   The single highest-value lever. Closes the train/eval gap and keeps
   small objects large enough to localize at whole-image eval. 96 GB
   makes 1280px training feasible at a real batch size.
2. **Tiled / windowed inference at eval** (orthogonal, machine-agnostic).
   Evaluate on 640 windows of the full image and merge, matching the
   training tile resolution. This likely recovers a large chunk of pup
   AP for free and does not need Blackwell — worth doing regardless,
   and worth confirming whether the kit's own selection-criterion eval
   (class-agnostic, NFS-excluded `detect_metrics.json`) is also
   whole-image (if so, the selection number is suppressed too).
3. **Bigger per-GPU batch** — faster epochs, but will NOT move AP-small.
   Secondary.
4. **Larger backbone (dinov3_b/l)** — deprioritize. Large-object AP
   already ~0.4; capacity is not the constraint.

### RESULT (2026-06-07): tiled eval recovers pup AP — gap was the protocol

Re-scored the gen005 pup_vs_nonpup checkpoint (arisia 2589, mid-training)
whole-image vs tiled on the v2 test set, class-agnostic AP@0.5:

| | whole-image@640 | tiled | gain |
|---|---|---|---|
| overall | 0.542 | **0.857** | +0.32 (1.58x) |
| **pup** | 0.123 | **0.838** | **+0.72 (6.8x)** |
| nonpup_sealion | 0.689 | 0.879 | +0.19 |

The pup gap was almost entirely an **eval-protocol artifact**, not a model
deficiency. Tiling (sliding native-resolution 640 windows + per-class NMS
merge, batched) measures the model at the resolution it trained on; pups
stay ~46px instead of shrinking to ~7px. A mid-training checkpoint detects
pups at AP ~0.84 once measured correctly.

Consequences:
* **Tiled eval should be the standard selection metric** for this corpus
  (whole-image@input grossly understates small-object AP). Wire it on by
  default for sealion evals (KCD_TILED_EVAL).
* The Blackwell **1280-training experiment drops in priority**: it was
  premised on "small objects need higher *training* resolution," but the
  model already detects them — the gap was *eval* resolution. 1280 may
  still add a little, but it's polish, not the fix. Spend Blackwell on
  throughput / a capacity bump / faster iteration instead, if anything.
* Operationally: deploy with tiled/windowed inference on the full aerials.

### CONFIRMED (2026-06-07): aiq-gpu Blackwell, fully-trained, matches arisia

The 640 Blackwell shakedown (4-GPU, batch 32, 30 epochs, standalone
no-slurm) completed and ran the batched tiled eval (batch=64). Final
class-agnostic AP@0.5: overall 0.858, pup 0.838, nonpup 0.881 — within
noise of the arisia 2-GPU mid-training rescore (0.857 / 0.838 / 0.879).
Takeaways: (1) the whole Blackwell path is validated — sm_120 deform ops,
4-GPU DDP, AMP, standalone docker, symlinked SSD cache, batched tiled eval
all work; (2) cross-hardware reproducible; (3) pup AP is converged at
~0.838 (fully-trained == mid-training, so pup detection plateaus early).

### (superseded) Recommended first Blackwell experiment

pup_vs_nonpup, dinov3_s, **1280px** input (train + eval), per-GPU batch
sized to fill ~96 GB (start 8–12, scale up), LR scaled with batch,
same balanced sampler and 30-epoch budget. Compare AP-small and overall
mAP against arisia's 640px 2589 at matched epochs. If AP-small lifts
materially, resolution is confirmed as the lever and we scale further /
add tiled eval.

## RESOLVED: the official selection metric is ALSO whole-image at 640

Traced the kit's `sweep` eval path (2026-06-06):

- `sweep` → `pareto_sweep._run_eval` → `eval/kwcoco_eval.run_kwcoco_eval`.
- `kwcoco_eval.py:171-359` loops over each test image, reads the whole
  image (`coco_img.imdelay().finalize()`), and calls
  `predictor.predict_image(arr, (W, H))` **once per whole image**.
- `trainers/deimv2.py:717-750` `predict_image()` resizes the whole image
  to `eval_spatial_size` (= the train `input_hw`, currently 640×640) and
  runs a **single forward pass**. There is NO sliding-window / tiled
  inference anywhere in the predict or eval path (the only
  "sliding-window" code is in `data/tile.py`, the *training* tile builder).
- The NFS-excluded selection metric (`detect_metrics.NFS.json`, produced
  by `_rerun_eval_dropping_distractors`, kwcoco_eval.py:100-168) is
  computed from exactly these whole-image-at-640 detections.

So the **official detection-AP selection criterion is gated by the same
train/eval mismatch** — not just the DEIMv2 internal COCO mAP. Every
gen001-gen005 selection number understates small-object (pup) capability.

### Two independent, both-worth-doing fixes

1. **Tiled / windowed eval (kit change, machine-agnostic) — highest
   leverage.** Slide a 640 window over the full image at inference, merge
   with cross-tile NMS, matching the training tile resolution. Does not
   exist yet; `predict_image` would need a windowed wrapper (the geometry
   already exists in `data/tile.py:262` `_slider_positions`). This is the
   structural fix: at whole-image-640 a 46px pup is ~7px; in a 640 window
   at native scale it stays 46px.

2. **Higher-resolution train+eval (Blackwell lever).** `eval_spatial_size`
   is synchronized to train `input_hw`, so training at 1280 also evals
   whole-image at 1280 — a 46px pup becomes ~15px instead of ~7px. Helps,
   but is a *partial* fix on its own because whole-image inference of a
   multi-thousand-px aerial still downsamples. Best combined with (1).

### Rebuilding tiles on aiq-gpu: use the new `tile-corpus` builder

Commit `0d3f3e1` added `kwcoco-detector-kit tile-corpus <src> <dst>
--spec spec.yaml`, which composes multiple `tile` passes (full_only @
`full_dim`, quadrant NxN, multiscale) into one unioned training bundle
with absolute image paths. This is the clean way to build a richer,
higher-resolution training corpus on aiq-gpu (we have SSD headroom).
Tile robustness fixes `5d168d8` / `9c61218` (contiguous uint8 RGB,
`ensure_uint255`) are in the current image — a rebuild picks them up.
The current cache `b9540ace` is `multiscale, scales=1.0,0.5,
tile_size=640`; a Blackwell corpus would add a `full_only @1280` pass
and/or `quadrant` to cover the apparent-scale range a higher-res model
sees.
