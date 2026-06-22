# gen006 4-GPU X@1280 results (2026-06-22)

Run: `pup_vs_nonpup_deimv2_dinov3_x_4gpu_aiq_gen006_1280`  
Slurm job: 25 (aiq-gpu, 4× RTX PRO 6000 Blackwell 96 GB)  
Script: `projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_x_4gpu_aiq_gen006_1280.sh`

---

## Training summary

| parameter | value |
|-----------|-------|
| backbone | dinov3_x |
| resolution | 1280 × 1280 |
| GPUs | 4 × 96 GB Blackwell |
| per-GPU batch | 4 |
| total batch | 16 |
| LR | 4e-4 (head), 2e-5 (backbone) |
| epochs | 45 |
| flat_epoch | 22 |
| balance | file |

**Wall time**: 2026-06-21T15:18 → 2026-06-22T01:13 = **~9.9 hours**  
(2-GPU 30-epoch reference was 20.5 hours — roughly 2× speedup from 2× more GPUs with 1.5× more epochs)

---

## Evaluation results (standard/whole-image eval)

| metric | 4-GPU 45ep | 2-GPU 30ep (job 23) | delta |
|--------|-----------|----------------------|-------|
| nocls AP@50 | 0.8819 | 0.8993 | −0.017 |
| pup AP@50 | 0.8693 | 0.8748 | −0.006 |
| nonpup AP@50 | 0.9074 | 0.9204 | −0.013 |

The 4-GPU 45-epoch run is slightly below the 2-GPU 30-epoch run on whole-image eval.  
**Both numbers are from standard (whole-image) eval** — tiled eval is the correct
comparison criterion and has not yet been run for either the 4-GPU or the 2-GPU run.

### Possible explanation: batch×LR interaction

Total batch went from **4** (2-GPU × 2/GPU) to **16** (4-GPU × 4/GPU), a 4× increase.
LR was held constant at 4e-4. Standard practice would scale LR by sqrt(4) = 2× → 8e-4,
or by linear scaling → 16e-4. Keeping LR fixed while quadrupling batch is the likely
cause of the regression.  The model trained but didn't extract as much from each
gradient step relative to the batch gradient noise.

If running another 4-GPU X@1280, try LR = 8e-4 (sqrt scaling) or 1.6e-3 (linear scaling)
and compare standard eval against the 2-GPU baseline.

---

## GPU memory profile (4-GPU × batch=4/GPU)

| epoch | peak max_mem MB | notes |
|-------|-----------------|-------|
| 0 (step 0) | 14,886 | warmup; first step |
| 0 | 18,751–19,142 | steps 500-1000 |
| 4+ (Mosaic active) | ~50,808 | steady state through ep 43-44 |

**Peak: 50,808 MB per GPU** = 49.6 GB on a 96 GB GPU (52% utilisation)

Prediction from the 2026-06-21 design journal (based on 2-GPU @batch=2/GPU = 23,824 MB):
- Linear: ~47,648 MB
- Const+linear: ~35,600 MB

Actual (50,808 MB) is ~6.6% above the fully-linear estimate — both estimates were
slightly low. Still comfortably within 96 GB; 45 GB of headroom remains.

---

## In-loop training dynamics

COCO mAP@0.5:0.95 (in-loop, whole-image; compare across epochs, not to tiled eval):

| epoch range | mAP |
|-------------|-----|
| 0–5 | rapid rise |
| 36–43 | 0.345–0.348 (flat plateau) |
| 44 | 0.252 (drop — final epoch anomaly) |

The plateau at 0.346-0.348 from ep36-43 is the effective ceiling with this schedule.
The drop at ep44 suggests the final cosine step or some numerical quirk. Since
`best_stg2.pth` tracks the global best checkpoint, the actual checkpoint used
for eval was from the 0.348 plateau, not ep44.

The 2-GPU 30ep run plateaued similarly in its final epochs (the 2026-06-21 design
journal noted "LR exhausted at epoch 29"). The 45ep flat_epoch=22 schedule gave
7 more high-LR epochs but the plateau region just extended — suggesting more epochs
alone doesn't break the ceiling at this batch/LR combination.

---

## ONNX export failure and fix

**Failure**: export killed with SIGKILL (CPU RAM OOM) during `torch.onnx.export`.
No `.onnx` file was written.

**Root cause**: `tpl/DEIMv2/tools/deployment/export_onnx.py` line 59 hardcodes
`data = torch.rand(32, 3, *img_size)` — batch=32 for the forward trace. At 1280px,
this creates 32 images × full ViT-X intermediate activations in CPU RAM during the
trace. The model is exported with `dynamic_axes={'images': {0: 'N'}}` so the batch
dimension is already dynamic; batch=32 is unnecessary and catastrophically large at
1280px.

**Fix** (committed in same session): changed `32` → `1`.

```python
# Before (OOM at 1280px dinov3_x):
data = torch.rand(32, 3, *img_size)

# After (batch dim is dynamic anyway):
data = torch.rand(1, 3, *img_size)  # batch=1: dim is dynamic; 32 OOMs at 1280px
```

This fix is added to the DEIMv2 upstream patch queue. The export must be re-run
against `best_stg2.pth` with the rebuilt image.

---

## Next steps

1. **Tiled eval**: run tiled eval on `best_stg2.pth` for the proper comparison
   to the 2-GPU 30ep run. The whole-image numbers are not the selection criterion.
2. **ONNX export**: rebuild image (includes the batch=1 fix), then re-run export
   sweep or call `export_onnx` directly.
3. **LR scaling experiment**: if running another 4-GPU X@1280, try LR=8e-4 (sqrt
   scaling for 4× batch) to test whether the regression is a batch/LR interaction.
