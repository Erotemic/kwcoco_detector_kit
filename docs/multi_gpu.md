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

## SLURM / k8s Submit Pattern

The kit ships a single-node 4x RTX A6000 OpenGroundingDINO template:

```bash
sbatch --export=ALL,\
TRAIN_KWCOCO=/scratch/data/train.kwcoco.zip,\
VALI_KWCOCO=/scratch/data/vali.kwcoco.zip,\
TEST_KWCOCO=/scratch/data/test.kwcoco.zip,\
CATEGORY_NAME=widget \
    examples/slurm/a6000_4x_opengroundingdino.sbatch
```

The generic template runs:

```bash
python -m kwcoco_detector_kit sweep ... \
    --trainer opengroundingdino \
    --variant opengroundingdino_swint \
    --num_gpus 4 \
    --distributed \
    --scale_tier 2-4xL
```

For the VIAME sea-lion data specifically, use:

```bash
sbatch examples/viame_sealions_2026/run_slurm_a6000_opengroundingdino.sbatch
```

That wrapper reuses the VIAME tile cache before calling the same sweep path.

The cluster's NCCL config (``NCCL_IB_DISABLE``, ``NCCL_SOCKET_IFNAME``, etc.) is the user's responsibility; the kit sets only conservative single-node defaults such as ``TORCH_NCCL_ASYNC_ERROR_HANDLING=1``.

### CUDA Driver vs Toolkit

``nvidia-smi`` reports the driver runtime capability, not the compiler ABI used
to build PyTorch extensions. For example, a machine may report:

```text
Driver Version: 595.58.03  CUDA Version: 13.2
```

That means the driver can run CUDA 13.2-era binaries. It does **not** mean the
active ``nvcc`` toolkit or the installed PyTorch wheel is CUDA 13.2.

OpenGroundingDINO builds a CUDA extension, so these must match:

```bash
python - <<'PY'
import torch
print(torch.version.cuda)
PY
nvcc --version
```

If PyTorch is a ``cu130`` wheel, load/install CUDA toolkit 13.0 for ``nvcc``
even when the driver banner says CUDA 13.2. If a ``cu132`` wheel is installed,
then load CUDA toolkit 13.2. The setup scripts intentionally fail fast when
``torch.version.cuda`` and ``nvcc --version`` disagree.

## Heterogeneous-VRAM warning

When ``num_gpus > 1`` with mixed GPUs (e.g. a 24 GB + a 16 GB peer on the same host), the effective per-GPU batch is set by the smallest peer. The kit logs the aggregate VRAM but does NOT subdivide per peer; if you have mixed VRAM, set ``CUDA_VISIBLE_DEVICES`` to expose only the matched GPUs.

## FSDP / sharded data parallel

NOT supported in v1 — DEIMv2 DINOv3-X (~50M params at 800×800) fits in 80 GB; OpenGroundingDINO Swin-Base fits in 80 GB with batch=4. Sharded data parallel is a future expansion.
