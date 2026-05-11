# Multi-GPU + DDP

The kit's trainer plugins delegate to upstream trainers' own distributed launchers — DEIMv2's ``train.py`` runs under ``torch.distributed.run``; OpenGroundingDINO's ``train_dist.sh`` handles its own ``torchrun``. The kit's job is to pick the right knobs and warn about the configurations that are silently slower than single-GPU.

## CLI

```bash
python -m kwcoco_detector_kit sweep ... \
    --num_gpus 4 \
    --distributed
```

`--num_gpus N` sets the world size; `--distributed` opts into the upstream distributed launcher. The trainer's ``launch(num_gpus=N, distributed=True)`` does the rest.

Per-cell effective batch is ``per_gpu_batch * N``. The per-GPU default comes from the trainer's ``memory_tier_default_batch(variant, input_hw, total_vram_gb)``; ``--batch_size`` overrides.

## Tier auto-detect

```python
from kwcoco_detector_kit.trainers._tier import detect_tier
info = detect_tier()
# info.tier = 'S' | 'M' | 'L' | 'XL' | '2-4xL' | '4xXL' | 'cluster'
# info.aggregate_vram_gb summed across visible CUDA devices.
# info.pcie_warning is set if multi-GPU + PCIe < 8x somewhere.
```

`detect_tier()` queries `torch.cuda.mem_get_info()` × world_size and the GPUs' PCIe link widths via `nvidia-smi`. Override with `--tier S/M/L/XL/cluster`.

## Failure #17 — PCIe-link-width mismatch

The kit's defense:

1. `_env.default_cuda_visible_devices()` defaults `CUDA_VISIBLE_DEVICES=0` on single-host non-cluster setups (the user opts in to multi-GPU explicitly).
2. `_tier.detect_tier()` probes `nvidia-smi --query-gpu=pcie.link.width.current` and emits a warning when `num_gpus > 1` and any active GPU is below 8× lanes.

Concrete failure mode from the prior project: a 2× RTX 3090 host where GPU 1 was on a 2× PCIe link trained SLOWER in DDP than single-GPU on GPU 0. Always single-GPU on the fast peer when the slow peer is the bottleneck.

## SLURM / k8s submit pattern

The kit doesn't ship SLURM helpers (yet). A reasonable single-node submit:

```bash
#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load cuda/13.0
source $HOME/venvs/kcd/bin/activate

export KCD_ROOT=/scratch/$USER/kcd_run
export KCD_DEIMV2_REPO_DPATH=/scratch/$USER/tpl/DEIMv2

python -m kwcoco_detector_kit sweep \
    --train_kwcoco /scratch/data/train.kwcoco.zip \
    --vali_kwcoco  /scratch/data/vali.kwcoco.zip \
    --test_kwcoco  /scratch/data/test.kwcoco.zip \
    --kcd_root "$KCD_ROOT" \
    --trainer deimv2 --variant deimv2_dinov3_l \
    --input_hw 800,800 --train_policy multiscale \
    --num_gpus 4 --distributed \
    --scale_tier XL --num_epochs 30 --use_amp true
```

The cluster's NCCL config (``NCCL_IB_DISABLE``, ``NCCL_SOCKET_IFNAME``, etc.) is the user's responsibility; the kit doesn't override the cluster's defaults.

## Heterogeneous-VRAM warning

When ``num_gpus > 1`` with mixed GPUs (e.g. a 24 GB + a 16 GB peer on the same host), the effective per-GPU batch is set by the smallest peer. The kit logs the aggregate VRAM but does NOT subdivide per peer; if you have mixed VRAM, set ``CUDA_VISIBLE_DEVICES`` to expose only the matched GPUs.

## FSDP / sharded data parallel

NOT supported in v1 — DEIMv2 DINOv3-X (~50M params at 800×800) fits in 80 GB; OpenGroundingDINO Swin-Base fits in 80 GB with batch=4. Sharded data parallel is a future expansion.
