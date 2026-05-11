# Scale tiers

Five tiers + a `cluster` tier on top, scaled by aggregate VRAM. The trainer plugin's `memory_tier_default_batch(variant, input_hw, total_vram_gb)` returns a per-GPU batch; multi-GPU DDP multiplies for effective batch.

## Tiers

| tier | example hardware | aggregate VRAM | recommended variant family | sample (variant, input_hw, batch) |
|---|---|---|---|---|
| **S** legacy single-GPU | 1× GTX 1080 Ti / Titan X (12 GB) | 12 GB | DEIMv2 HGNetv2 Atto / Femto / Pico | (atto, 320, 32), (pico, 320, 32) |
| **M** consumer single-GPU | 1× RTX 3090 / 4090 (24 GB) | 24 GB | DEIMv2 HGNetv2 N/S, DEIMv2 DINOv3-S | (n, 320, 32), (s, 416, 16), (dinov3_s, 640, 8) |
| **L** workstation | 1× RTX 6000 Ada / L40S (48 GB) | 48 GB | DEIMv2 DINOv3-M, OGDino-Swin-Tiny | (dinov3_m, 640, 16), (ogdino_swint, 800, 4) |
| **XL** single server | 1× A100/H100 (80 GB) | 80 GB | DEIMv2 DINOv3-L/X, OGDino-Swin-Base | (dinov3_l, 800, 16), (ogdino_swinb, 1024, 4) |
| **2-4×L** small cluster | 2-4× 24–48 GB | 96–192 GB | same as L per GPU, DDP for bigger batches | DDP × 4 |
| **4×XL** mid cluster | 4× A100 80 GB / 4× H100 / 4× B200 96 GB | 320-384 GB | DINOv3-X, full OGDino sweep | DDP × 4, large batches |
| **cloud** | N×A100/H100 via SLURM/k8s | varies | same as L–XL with cloud-mount kwcoco | document SLURM submit pattern |

## Failure modes the tier system defends against

- **#16** — naïve upstream DEIMv2 batches OOM on 24 GB. The kit's per-`(variant, input_hw, tier)` table prevents the regression.
- **#17** — multi-GPU all-reduce can be SLOWER than single-GPU when peer PCIe lanes are mismatched. `_tier.py` probes `pcie.link.width.current` and warns on mismatch.
- The kit defaults `CUDA_VISIBLE_DEVICES=0` on single-host non-cluster configs — opt-in to multi-GPU explicitly.

## Auto-detection

At trainer launch on rank 0:

```python
from kwcoco_detector_kit.trainers._tier import detect_tier
tier = detect_tier()         # 'S'|'M'|'L'|'XL'|'cluster'
```

`detect_tier()` queries `torch.cuda.mem_get_info()` × world_size and looks up the closest tier. Override with `--tier`.

AMP defaults: ON for tier ≥ M, OFF for tier S.

FSDP / sharded are NOT supported in v1 — DINOv3-X (50 M params + 800×800) fits in 80 GB.
