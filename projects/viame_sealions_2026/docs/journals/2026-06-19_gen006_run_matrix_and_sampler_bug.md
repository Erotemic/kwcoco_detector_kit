# gen006 — sampler balance ablation + max_oversample bug (2026-06-19)

Generation 6 goal: test whether **dataloader-level sampler balance** improves
pup AP over the gen005 file-duplication baseline, and stack the two gen005
winners (X backbone + 1280px resolution) to see if they combine additively.

---

## Prior results (gen005 reference)

| run | scheme | backbone | res | GPUs | overall AP | pup AP |
|-----|--------|----------|-----|------|-----------|--------|
| gen005 arisia | pup_vs_nonpup | S | 640 | 2 | 0.857–0.858 | 0.838 |
| gen005 aiq (4-GPU) | pup_vs_nonpup | X | 640 | 4 | 0.892 | 0.864 |
| gen005 S@1280 | pup_vs_nonpup | S | 1280 | — | not run | — |

gen005 X@640 gave a uniform +0.03 vs S@640 (not a pup-specific unlock). Pup
AP is still the binding constraint; the hypothesis entering gen006 is that
**balanced sampling** (drawing rare-pup tiles more often without static
duplication) might help more than capacity.

---

## Gen006 run matrix

| machine | backbone | res | balance | GPUs | total batch | status (2026-06-19) |
|---------|----------|-----|---------|------|-------------|---------------------|
| namek | S | 640 | sampler | 1 | 4 | OOM at stage2/ep4 → restarted batch=4 (~16:28 UTC+local) |
| arisia | S | 640 | sampler | 2 | 8 | full_8cls scheme (running) |
| aiq | X | 640 | sampler | 2 | 16 | blocked by max_oversample bug → fix committed, pending restart |
| aiq | X | 1280 | file | 2 | 4 | 1280px tile cache building (~ETA 1h from ~2026-06-19 15:xx) |
| yardrat | S | 1280 | sampler | 1 | 2 | waiting on tile cache build |

The aiq X@640 sampler run was the first to expose the max_oversample bug
(925k-image dataset). It needs the image rebuilt with the fix before resubmit.

Tile-cache hash map:
- `b9540ace` — 640px (exists on arisia, aiq, namek)
- `0441d89e` — 1280px (building on aiq 2026-06-19; arisia has only b9540ace)

---

## Namek gen006 partial results (S@640 sampler, 1-GPU, batch=8→4)

First run (batch=8) reached stage 2 epoch 4 before OOM on the 24GB RTX 3090.
DEIMv2 stage 2 activates Mosaic augmentation, compositing 4 source images per
step → 4× bounding boxes → O(N²) Hungarian matching → memory spikes.

In-loop COCO eval (whole-image, not tiled — for trend only):

| phase | epoch | loss | mAP (in-loop) |
|-------|-------|------|----------------|
| stage 1 | 0 | 22.97 | 0.083 |
| stage 1 | 1 | 21.24 | 0.084 |
| stage 2 | 0 | 21.66 | 0.082 |
| stage 2 | 2 | 20.67 | 0.084 |
| stage 2 | 3 | 20.32 | **0.085** ← best |
| stage 2 | 4 | — | OOM |

`best_stg2.pth` (Jun 18 ~21:04) is the checkpoint to score; `last.pth` is
frozen at stage-1 epoch 0 (DEIMv2 design). Loss still trending down — model
not converged at OOM.

**Restart** (batch=4) launched 2026-06-19 ~16:28. Starts from COCO pretrained
init (clean slate, not from best_stg2.pth), which is correct: the optimizer
state from a batch=8 run would be stale, and the stage-1 warmup is cheap.
Sampler balance weights already computed (reused). Expected to clear Mosaic
stage at ~14GB/GPU (fits in 24GB).

Tiled AP via rescore still pending (submit_rescore_per_checkpoint.sh on
arisia against best_stg2.pth).

---

## Critical bug: max_oversample infinite loop (commit 7d88605)

**Symptom**: training hung for 11+ hours after "Build balanced forests
100.00% 16/16... total=0:01:05" with no further output. The forest build
itself completed in 65 seconds; the hang was in the weight post-processing.

**Root cause** in `kwcoco_detector_kit/data/balanced_sampler.py`:

```python
# OLD — broken
for _ in range(len(weights) + 1):   # = range(925160) for this dataset
    over = [w > cap for w in weights]
    if not any(over): break
    weights = [min(w, cap) for w in weights]
    total = sum(weights)
    weights = [w / total for w in weights]
```

After capping, renormalization divides by `total < 1`, pushing capped
weights marginally above the cap again due to float64 rounding. The `break`
condition never fires. Each outer iteration does 4 Python list comprehensions
over 925k items ≈ 220ms. Estimated total: 925 159 × 220ms ≈ **57 hours**.
At the 11-hour point, the loop was ~19% complete (≈178k of 925k iterations).

**Fix** (`kwcoco_detector_kit/data/balanced_sampler.py`, commit `7d88605`):

Replace all Python list comprehensions with numpy vectorized ops (each pass
~2ms) and bound the outer loop at 512 iterations. Convergence happens in
20–30 iterations (~60ms total) for any realistic class distribution.

```python
# NEW — fast
import numpy as np
w_arr = np.array(weights, dtype=np.float64)
cap = k / len(w_arr)
n_capped_total = 0
for _ in range(512):
    over = w_arr > cap
    if not over.any():
        break
    n_capped_total += int(over.sum())
    np.minimum(w_arr, cap, out=w_arr)
    total = w_arr.sum()
    if total <= 0:
        raise ValueError(...)
    w_arr /= total
weights = w_arr.tolist()
```

**Impact**: only triggered by sampler-mode balance (`KCD_BALANCE_MODE=sampler`)
on large datasets (925k images). File-mode balance is unaffected.

---

## Timing log (2026-06-19)

| time (approx) | event |
|---------------|-------|
| 2026-06-18 ~09:00 | aiq X@640 sampler run submitted; hung 11h in max_oversample loop |
| 2026-06-18 ~21:04 | namek best_stg2.pth written (mAP 0.085 in-loop) |
| 2026-06-18 ~21:xx | namek OOM at stage2/ep4 |
| 2026-06-19 morning | max_oversample bug diagnosed; numpy fix committed (7d88605) |
| 2026-06-19 morning | aiq X@640 sampler and X@1280 file scripts committed (06ce22e, 529d9b0) |
| 2026-06-19 ~15:xx | 1280px tile cache build submitted on aiq (ETA ~1h) |
| 2026-06-19 ~16:28 | namek gen006 sampler restart (batch=4) launched |

---

## Next steps

1. **Score namek best_stg2.pth** — tiled AP via rescore on arisia.
2. **Rebuild aiq image** (image bakes the numpy fix) → resubmit X@640 sampler.
3. **After 1280px tile cache** (~ETA ~16:xx+1h): submit X@1280 file on aiq.
4. **After both aiq 2-GPU jobs**: compare sampler vs file, 640 vs 1280; pick
   best for 4-GPU X@1280 follow-up.
5. **Yardrat S@1280**: build 1280px tile cache, launch S@1280 sampler.
