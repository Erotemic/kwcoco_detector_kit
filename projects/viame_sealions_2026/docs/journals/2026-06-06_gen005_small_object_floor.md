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

### Recommended first Blackwell experiment

pup_vs_nonpup, dinov3_s, **1280px** input (train + eval), per-GPU batch
sized to fill ~96 GB (start 8–12, scale up), LR scaled with batch,
same balanced sampler and 30-epoch budget. Compare AP-small and overall
mAP against arisia's 640px 2589 at matched epochs. If AP-small lifts
materially, resolution is confirmed as the lever and we scale further /
add tiled eval.

## Open question to resolve before the big run

Does the kit's sweep eval (the one producing the official detection AP
selection metric) also run whole-image at 640? If yes, both the COCO
mAP shown here and the official selection number are gated by the same
mismatch, and tiled eval becomes the cheapest global win — independent
of the Blackwell resolution experiment.
