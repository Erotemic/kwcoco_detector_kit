# GPU memory analysis and 4-GPU X@1280 run design (2026-06-21)

Posthoc analysis of the gen006 X@1280 2-GPU run (aiq slurm job 23, pup AP
0.875). Goals: (1) measure per-epoch peak GPU memory to infer the safe batch
size for a 4-GPU scale-up, (2) assess whether results were plateauing at
epoch 29 or whether more epochs would help.

Raw data file: `data/gpu_memory_x1280_2gpu_job23.json`.

---

## How memory was measured

DEIMv2 prints per-GPU memory at every logged training step:

```
cur mem: <MB>  max mem: <MB>
```

`max mem` is `torch.cuda.max_memory_allocated()` — a monotonically-increasing
high-water mark reset at the start of each epoch.  `cur mem` is the live
allocator footprint at that step.  The analysis script
`dev/analyze_gpu_memory.py` parses both fields and reports per-epoch peaks.

Run:

```bash
python3 dev/analyze_gpu_memory.py \
  --log-file /data/users/jon.crall/slurm_logs/pup_vs_nonpup_deimv2_dinov3_x_2gpu_aiq_gen006_1280-23.out \
  --gpu-gb 96
```

---

## Memory profile — X@1280, batch=2/GPU, 2 GPUs, 96 GB RTX PRO 6000

| Phase | Epochs | Peak max_mem | Notes |
|-------|--------|-------------|-------|
| init / warmup | 0 | 10,182 MB | very first logged steps |
| transition | 1–3 | 12,393 MB | augmentation ramp |
| heavy aug (Mosaic + Mixup + CopyBlend) | 4–14 | 21,683 MB | +9.3 GB jump vs transition |
| full aug stack | 15–29 | **23,824 MB** | steady state; coincides with LR cosine decay start |
| eval | — | 23,824 MB | same as training peak |

**Global peak: 23,824 MB = 23.3 GB per GPU**
**Headroom on 96 GB GPU: 74,480 MB = 72.7 GB**

Memory jumps correspond to augmentation transitions in the DEIMv2 schedule:
- Epoch 4: `Mosaic + Mixup + CopyBlend` fully activate (policy `epoch: [4, 29, 50]`)
- Epoch 15: `mixup_epochs: [4, 29]` window causes maximum composite box count

---

## Batch size inference for 4-GPU run

Linear scaling estimate (conservative — actual headroom is larger because
model weights and optimizer state, ~12 GB, are constant across batch sizes):

| per-GPU batch | total batch | estimated peak (linear) | estimated peak (const+linear) | vs 96 GB | verdict |
|--------------|-------------|------------------------|-------------------------------|----------|---------|
| 2 (baseline) | 4 | 23,824 MB (measured) | 23,824 MB | −74.5 GB | **safe** |
| 4 | 16 | ~47,648 MB | ~35,600 MB | −50–63 GB | **safe** |
| 8 | 32 | ~95,296 MB | ~59,300 MB | −3–39 GB | risky / probably OK |

The constant+linear model estimates model overhead at ~12,000 MB (23,824 −
11,824 activation component), then scales activations linearly with batch.
At batch=4/GPU the most conservative estimate gives 47 GB — well within 96 GB.

**Decision: 4 GPUs × batch=4/GPU = 16 total samples per gradient step.**

---

## LR schedule for the 30-epoch reference run

`flat_epoch` is computed in the kit as `num_epochs // 2` (`deimv2.py:533`):

- `num_epochs=30` → `flat_epoch=15` → 15 flat epochs, 15 cosine epochs
- `lr_gamma=0.5` → minimum LR = peak_LR × 0.5 (does not decay to 0)

Observed backbone LR per epoch (from final step per epoch):

| epochs | backbone LR | phase |
|--------|-------------|-------|
| 0–15 | 2e-5 | flat (plateau) |
| 16–24 | 1.9e-5 → 1.4e-5 | cosine decay |
| 25–29 | ~1.1e-5 → 1.0e-5 | late cosine |

By epoch 29 the LR is at its minimum (1e-5 = 2e-5 × 0.5). The cosine phase
ran to completion. Any further gains from extending the schedule would require
a fresh training run with more epochs (not a resume from the frozen `last.pth`).

---

## Plateau assessment

In-loop COCO mAP (whole-image; for trend only — tiled AP is the criterion):

| epoch range | mAP gain per epoch |
|-------------|-------------------|
| 0–6 | +0.006–0.014 (fast, Mosaic active) |
| 7–14 | +0.002–0.007 (slowing) |
| 15–24 | +0.001–0.003 (slow but steady) |
| 25–29 | 0.000–0.001 (very slow) |

The last 5 epochs gained only 0.003 total.  The LR was already at its
minimum — this is not "early plateau" but rather "schedule exhausted".

**Conclusion: extending the 30-epoch run by resuming would not help (LR is
exhausted, `last.pth` is epoch 0 per DEIMv2 design). Training from scratch
with more epochs is the right lever. With 45 epochs: `flat_epoch=22` (7 more
high-LR epochs than the 30-epoch run) plus a longer cosine phase.**

---

## 4-GPU run design (gen006 4-GPU X@1280)

| parameter | value | reasoning |
|-----------|-------|-----------|
| backbone | dinov3_x | best single-run result so far (pup AP 0.875) |
| resolution | 1280px | confirmed better than 640px |
| per_gpu_batch | 4 | 2× current; proven safe by memory analysis |
| total_batch | 16 | 4× the 2-GPU reference |
| LR | 4e-4 | unchanged from 2-GPU best; historical evidence does not support linear LR scaling in our runs |
| backbone_LR | 2e-5 | unchanged |
| epochs | 45 | 50% more than 30-epoch ref; flat_epoch=22 → 7 more high-LR epochs |
| balance | file | sampler diverged (NaN in gen006 X@640); one variable at a time |
| host | aiq-gpu | 4× 96 GB Blackwell |
| gpus | 4 | KCD_NUM_GPUS=4 |

Script: `scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_x_4gpu_aiq_gen006_1280.sh`
(updated 2026-06-21: batch=4/GPU, 45 epochs).

Expected wall time: ~15 hours (estimate from 2-GPU 30-epoch = 20.5 h, adjusted
for 2× per-epoch parallelism + 4× fewer steps per epoch at 4× batch).

---

## Timed run summary

| event | time |
|-------|------|
| gen006 X@1280 2-GPU start (aiq, job 23) | 2026-06-19 18:08 |
| gen006 X@1280 2-GPU end (epoch 29 done) | 2026-06-20 14:38 |
| total wall time | ~20.5 hours |
| tiled eval: pup AP 0.875, overall AP 0.899 | 2026-06-20 ~15:00 |
| memory analysis (this journal) | 2026-06-21 |
| 4-GPU 45-epoch script committed | 2026-06-21 |
