"""
GPU scale-tier auto-detection.

Six tiers (see ``docs/scale_tiers.md``):

    S   1x 12-16 GB     single legacy GPU
    M   1x 24 GB        single consumer GPU
    L   1x 48 GB        single workstation GPU
    XL  1x 80 GB        single server GPU
    2-4xL  multi-GPU 24-48 GB cluster
    4xXL   multi-GPU 80-96 GB cluster
    cloud  >= 8 GPUs aggregated through SLURM/k8s

Auto-detection asks each visible GPU for its total VRAM via
``torch.cuda.mem_get_info``, sums across world_size, and looks up the
nearest tier. Returns a string label that the trainer plugin's memory
table indexes by.

PCIe link-width probe — failure #17:
``nvidia-smi --query-gpu=pcie.link.width.current`` warns when any
active GPU is below 8x lanes and ``num_gpus > 1``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from typing import Optional


# Inclusive lower bound in GB per tier. The tier returned is the
# largest tier whose lower bound is <= aggregate VRAM.
_TIER_FLOOR_GB = [
    ("S", 0),
    ("M", 20),
    ("L", 40),
    ("XL", 70),
    ("2-4xL", 90),   # >= 2 x 48 GB or >= 4 x 24 GB
    ("4xXL", 300),
    ("cloud", 600),
]


@dataclass
class TierInfo:
    tier: str
    aggregate_vram_gb: float
    num_visible_gpus: int
    pcie_warning: Optional[str] = None


def detect_tier(*, force: Optional[str] = None) -> TierInfo:
    """Return the closest tier label given the current host's GPUs.

    Honours ``force`` (the kit's ``--tier`` CLI flag) verbatim — useful
    for CPU CI smokes where the auto-detection would otherwise pick S.
    """
    if force is not None:
        return TierInfo(tier=str(force), aggregate_vram_gb=0.0, num_visible_gpus=0)

    aggregate_gb = 0.0
    num_gpus = 0
    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                free, total = torch.cuda.mem_get_info(i)
                aggregate_gb += total / (1024 ** 3)
    except Exception as ex:  # pragma: no cover — extreme env breakage
        warnings.warn(f"_tier.detect_tier: torch.cuda probe failed: {ex}")

    tier = "S"
    for label, floor in _TIER_FLOOR_GB:
        if aggregate_gb >= floor:
            tier = label

    pcie_warning = None
    if num_gpus > 1:
        pcie_warning = _probe_pcie_link_widths(num_gpus)

    return TierInfo(
        tier=tier,
        aggregate_vram_gb=round(aggregate_gb, 2),
        num_visible_gpus=num_gpus,
        pcie_warning=pcie_warning,
    )


def _probe_pcie_link_widths(num_gpus: int) -> Optional[str]:
    """Return a warning string if any active GPU is below 8x PCIe.

    nvidia-smi is the canonical source. We don't fail when it's
    missing — that's a normal CPU-only env.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        result = subprocess.run(
            [nvidia_smi,
             "--query-gpu=index,pcie.link.width.current",
             "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=5,
        )
    except Exception as ex:
        return f"nvidia-smi pcie probe failed: {ex}"
    if result.returncode != 0:
        return None
    widths = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                widths.append((int(parts[0]), int(parts[1])))
            except ValueError:
                continue
    bad = [(idx, w) for idx, w in widths if w < 8]
    if bad and num_gpus > 1:
        return (
            f"WARNING: GPU(s) with PCIe < 8x detected ({bad}); multi-GPU "
            "all-reduce will be bottlenecked by the slowest peer. "
            "Consider CUDA_VISIBLE_DEVICES=<fast index only> or --tier "
            "single-GPU."
        )
    return None


def use_amp_for_tier(tier: str) -> bool:
    """AMP defaults: on for tier >= M, off for tier S."""
    return tier != "S"
