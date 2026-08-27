"""
Trainer plugin protocol.

Each trainer plugin lives in its own submodule under ``trainers/`` and
is registered against the kit's central registry via
``trainers._registry.register_trainer``. The kit's orchestration layer
dispatches by name (``trainer="deimv2"``, ``trainer="opengroundingdino"``,
``trainer="mock_tiny"``).

The interface is a runtime-checkable Protocol so plugins can be
ordinary classes (instances need not inherit). The Protocol is
intentionally minimal — anything that's not in the interface is
trainer-private.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from kwcoco_detector_kit.predictors._interface import DetectorPredictor


def _pin_checkpoint(trainer, workdir: Path, checkpoint) -> Path:
    """Resolve the weights file a predictor should load.

    ``None`` defers to the trainer's own autoselection. An explicit path is
    used verbatim and is required to exist -- silently falling back to
    autoselection would score a DIFFERENT checkpoint than the caller named,
    and per-epoch selection would then compare a set of results that are all
    secretly the same epoch.
    """
    if checkpoint is None:
        return Path(trainer.find_checkpoint(workdir))
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"pinned checkpoint does not exist: {ckpt}")
    return ckpt


@runtime_checkable
class DetectorTrainer(Protocol):
    name: str
    variants: dict
    supports_onnx_export: bool

    def generate_config(
        self,
        train_kwcoco_fpath: str | Path,
        vali_kwcoco_fpath: str | Path,
        workdir: str | Path,
        *,
        variant: str,
        input_hw: tuple[int, int],
        train_policy: str,
        num_classes: int,
        batch_size: int,
        val_batch_size: int,
        num_epochs: int,
        lr: float,
        backbone_lr: float,
        use_amp: bool,
        channels: str,
        scale_tier: str,
        num_gpus: int,
        data_format: str,
        extra: dict | None,
    ) -> Path:
        """Write a generated config to disk and return its path."""

    def launch(
        self,
        config_fpath: str | Path,
        *,
        init_checkpoint: str | Path | None = None,
        resume: str | Path | None = None,
        num_gpus: int = 1,
        distributed: bool = False,
    ) -> Path:
        """Run the trainer subprocess against the generated config. Returns workdir."""

    def find_checkpoint(self, workdir: str | Path) -> Path:
        """Locate the canonical 'best' checkpoint in the workdir."""

    def supports_dynamic_input(self, variant: str) -> bool:
        """True iff this variant tolerates per-batch input-size variation.

        DEIMv2 HGNetv2 = False; DEIMv2 DINOv3 = True; OpenGroundingDINO DETR = True.
        ``round_loop`` coerces ``train_policy=multiscale`` → ``fixed`` when False.
        """

    def memory_tier_default_batch(
        self,
        variant: str,
        input_hw: tuple[int, int],
        total_vram_gb: float,
    ) -> int:
        """Per-GPU default batch from the trainer's memory table. Multi-GPU multiplies."""

    def supports_webdataset_input(self) -> bool:
        """True iff the trainer can consume pre-rendered webdataset shards."""

    def build_predictor(
        self,
        workdir: str | Path,
        *,
        device: str = "cpu",
        checkpoint: str | Path | None = None,
    ) -> DetectorPredictor:
        """Instantiate the predictor for a trained checkpoint in workdir.

        ``checkpoint`` pins a SPECIFIC weights file instead of letting
        ``find_checkpoint`` autoselect (best_stg2 > best_stg1 > last). The
        workdir is still required: it supplies the generated config, and with
        it the preprocessing contract, class order and input geometry the
        checkpoint was trained under. Passing a checkpoint from a different
        run is therefore a caller error that this interface cannot detect.

        The motivating case is per-epoch selection. A run that stages a
        checkpoint per epoch has no way to score epoch 4 rather than the
        autoselected one, so any comparison across epochs had to bypass
        ``run_kwcoco_eval`` entirely -- and thereby lose tiled inference, the
        protocol fingerprint and the distractor sidecar.
        """
