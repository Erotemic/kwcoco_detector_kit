"""SAM2 segmenter: inference wrapper and optional fine-tuning trainer.

SAM2 fills the second stage of the det→seg pipeline:
  detector boxes → SAM2 mask prompt → polygon → kwcoco annotation

Two public classes live here:

``SAM2Segmenter``
    Lazy-loaded inference wrapper.  Accepts box prompts, returns boolean masks.
    Used by :func:`~kwcoco_detector_kit.data.postprocess.detector_records_to_anns`.

``SAM2Trainer``
    Standalone trainer for SAM2 fine-tuning.  Prepares SAM2-format training
    data from kwcoco splits, generates a Hydra training config, and subprocesses
    SAM2's ``training/train.py``.  **Not** registered in the
    ``DetectorTrainer`` plugin registry because SAM2 is a segmenter, not a
    detector.

Environment variables
---------------------
``KCD_SAM2_REPO_DPATH``
    Path to a local SAM2 repository clone.  If unset, ``SAM2Segmenter`` falls
    back to loading a HuggingFace pretrained checkpoint via
    ``SAM2ImagePredictor.from_pretrained()``.  Fine-tuning always requires the
    repo.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

_ENV_REPO = "KCD_SAM2_REPO_DPATH"
_CONFIGS_SUBDIR = "sam2/configs/kcd_training"


# ---------------------------------------------------------------------------
# Repo resolution helpers
# ---------------------------------------------------------------------------

def _resolve_repo_dpath(segmenter_cfg: dict) -> Optional[Path]:
    env_key = segmenter_cfg.get("repo_envvar", _ENV_REPO)
    repo = segmenter_cfg.get("repo_dpath") or os.environ.get(env_key)
    return Path(repo).expanduser().resolve() if repo else None


def _ensure_repo_on_path(repo_dpath: Path) -> None:
    repo_str = str(repo_dpath)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _resolve_inference_config_name(segmenter_cfg: dict, repo_dpath: Optional[Path]) -> Optional[str]:
    """Derive the Hydra config name for SAM2ImagePredictor construction."""
    config_name = segmenter_cfg.get("hydra_config_name")
    if config_name is not None:
        return str(config_name).replace("\\", "/")
    config_relpath = segmenter_cfg.get("config_relpath")
    if config_relpath is not None:
        config_relpath = str(config_relpath).replace("\\", "/")
        return config_relpath.removeprefix("sam2/")
    config_fpath = segmenter_cfg.get("config_fpath")
    if config_fpath is not None and repo_dpath is not None:
        relpath = str(
            Path(config_fpath).expanduser().resolve().relative_to(repo_dpath)
        ).replace("\\", "/")
        return relpath.removeprefix("sam2/")
    return None


def _dump_hydra_global_yaml(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text("# @package _global_\n\n" + yaml.safe_dump(data, sort_keys=False))


def _deep_update(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# SAM2Segmenter — inference wrapper
# ---------------------------------------------------------------------------

class SAM2Segmenter:
    """Lazy-loaded SAM2 image predictor for box-prompted mask prediction.

    Args:
        segmenter_cfg: Config dict with optional keys:
            ``repo_dpath`` / ``repo_envvar`` — path to SAM2 repo or env var name.
            ``checkpoint_fpath`` — path to a trained checkpoint.
            ``config_relpath`` / ``hydra_config_name`` — Hydra config for the model.
            ``hf_model_id`` — HuggingFace model ID (used when checkpoint+config absent).
            ``device`` — torch device (default ``"cuda:0"``).
            ``mask_threshold`` — logit threshold for mask binarisation (default ``0.0``).
    """

    def __init__(self, segmenter_cfg: dict):
        self.segmenter_cfg = dict(segmenter_cfg)
        self._predictor = None

    def _lazy_init(self) -> None:
        if self._predictor is not None:
            return
        repo_dpath = _resolve_repo_dpath(self.segmenter_cfg)
        if repo_dpath is not None:
            _ensure_repo_on_path(repo_dpath)
        try:
            SAM2ImagePredictor = importlib.import_module(
                "sam2.sam2_image_predictor"
            ).SAM2ImagePredictor
        except Exception as ex:
            raise ImportError(
                "Could not import sam2.  Install it (pip install sam2) or set "
                f"{_ENV_REPO} to a local clone."
            ) from ex

        device = str(self.segmenter_cfg.get("device", "cuda:0"))
        checkpoint_fpath = self.segmenter_cfg.get("checkpoint_fpath")
        config_name = _resolve_inference_config_name(self.segmenter_cfg, repo_dpath)

        if checkpoint_fpath and config_name:
            build_sam2 = importlib.import_module("sam2.build_sam").build_sam2
            self._predictor = SAM2ImagePredictor(
                build_sam2(str(config_name), str(checkpoint_fpath), device=device),
                mask_threshold=float(self.segmenter_cfg.get("mask_threshold", 0.0)),
            )
        else:
            hf_model_id = self.segmenter_cfg.get("hf_model_id")
            if hf_model_id is None:
                raise KeyError(
                    "SAM2Segmenter needs either (checkpoint_fpath + config) or hf_model_id"
                )
            self._predictor = SAM2ImagePredictor.from_pretrained(
                hf_model_id, device=device
            )

    def predict_masks_for_boxes(self, image, boxes_xyxy: list) -> list[dict]:
        """Predict one mask per box prompt.

        Args:
            image: HWC uint8 numpy array.
            boxes_xyxy: List of ``[x1, y1, x2, y2]`` boxes in pixel coordinates.

        Returns:
            List of dicts with keys ``"mask"`` (2-D bool array) and ``"score"``
            (float), one per input box.
        """
        self._lazy_init()
        self._predictor.set_image(image)
        results = []
        for box in boxes_xyxy:
            masks, scores, _ = self._predictor.predict(
                box=box,
                multimask_output=False,
                return_logits=False,
                normalize_coords=True,
            )
            best = int(scores.argmax()) if len(scores) else 0
            results.append({
                "mask": masks[best],
                "score": float(scores[best]) if len(scores) else 0.0,
            })
        return results


# ---------------------------------------------------------------------------
# SAM2Trainer — fine-tuning
# ---------------------------------------------------------------------------

class SAM2Trainer:
    """Trainer for SAM2 fine-tuning from kwcoco training splits.

    Not registered in the ``DetectorTrainer`` plugin registry — SAM2 is a
    segmenter, not a detector.  Use :class:`SAM2TrainConfig` for the CLI.
    """

    name: str = "sam2"

    VARIANTS: dict = {
        "sam2.1_hiera_base_plus": {
            "hf_model_id": "facebook/sam2.1-hiera-base-plus",
            "config_relpath": "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",
            "training_template_relpath": "training/configs/sam2.1_hiera_b+_image_finetuning.yaml",
        },
        "sam2.1_hiera_large": {
            "hf_model_id": "facebook/sam2.1-hiera-large",
            "config_relpath": "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            "training_template_relpath": "training/configs/sam2.1_hiera_l_image_finetuning.yaml",
        },
        "sam2.1_hiera_small": {
            "hf_model_id": "facebook/sam2.1-hiera-small",
            "config_relpath": "sam2/configs/sam2.1/sam2.1_hiera_s.yaml",
            "training_template_relpath": "training/configs/sam2.1_hiera_s_image_finetuning.yaml",
        },
    }

    def __init__(self, variant: str = "sam2.1_hiera_base_plus"):
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown SAM2 variant {variant!r}; choose from {list(self.VARIANTS)}")
        self.variant = variant
        self.variant_cfg = dict(self.VARIANTS[variant])

    def _resolve_repo(self, segmenter_cfg: dict) -> Path:
        repo_dpath = _resolve_repo_dpath(segmenter_cfg)
        if repo_dpath is None:
            raise FileNotFoundError(
                f"SAM2 fine-tuning requires a local repo clone.  "
                f"Set {_ENV_REPO} to its path."
            )
        if not repo_dpath.exists():
            raise FileNotFoundError(repo_dpath)
        return repo_dpath

    def generate_training_config(
        self,
        *,
        prepared,
        workdir: str | Path,
        init_checkpoint_fpath: str | Path,
        segmenter_cfg: Optional[dict] = None,
        resolution: int = 1024,
        train_batch_size: int = 1,
        num_train_workers: int = 4,
        num_epochs: int = 20,
        base_lr: float = 5e-6,
        vision_lr: float = 3e-6,
        max_num_objects: int = 8,
        multiplier: int = 1,
        checkpoint_save_freq: int = 1,
        num_gpus: int = 1,
        num_nodes: int = 1,
        use_cluster: bool = False,
        config_overrides: Optional[dict] = None,
    ) -> dict:
        """Write Hydra training config and return metadata dict.

        ``prepared`` is the return value of
        :func:`~kwcoco_detector_kit.data.sam2_export.export_sam2_training_splits`
        (or one of its split sub-dicts — must have ``image_dpath``,
        ``gt_dpath``, ``file_list_fpath`` keys under ``prepared["train"]``).

        Returns a metadata dict that should be passed directly to
        :meth:`launch`.
        """
        seg_cfg = dict(self.variant_cfg)
        if segmenter_cfg:
            seg_cfg.update(segmenter_cfg)

        repo_dpath = self._resolve_repo(seg_cfg)
        workdir = Path(workdir).expanduser().resolve()
        generated_dpath = workdir / "generated_configs"
        generated_dpath.mkdir(parents=True, exist_ok=True)

        template_relpath = seg_cfg.get(
            "training_template_relpath",
            self.variant_cfg.get("training_template_relpath"),
        )
        if template_relpath is None:
            raise FileNotFoundError(
                f"No training_template_relpath for SAM2 variant {self.variant!r}"
            )
        template_fpath = (repo_dpath / template_relpath).resolve()
        if not template_fpath.exists():
            raise FileNotFoundError(
                f"SAM2 training template not found: {template_fpath}"
            )
        template = yaml.safe_load(template_fpath.read_text())

        # Extract prepared split paths
        train_split = prepared.get("train", prepared)  # allow passing train-only dict
        train_image_dpath = str(train_split["image_dpath"])
        train_gt_dpath = str(train_split["gt_dpath"])
        train_file_list_fpath = str(train_split["file_list_fpath"])

        # Patch scratch / training hyperparams
        template.setdefault("scratch", {})
        template["scratch"].update({
            "resolution": resolution,
            "train_batch_size": train_batch_size,
            "num_train_workers": num_train_workers,
            "num_frames": 1,
            "max_num_objects": max_num_objects,
            "base_lr": base_lr,
            "vision_lr": vision_lr,
            "num_epochs": num_epochs,
        })

        # Patch dataset paths
        template.setdefault("dataset", {})
        template["dataset"].update({
            "img_folder": train_image_dpath,
            "gt_folder": train_gt_dpath,
            "file_list_txt": train_file_list_fpath,
            "multiplier": multiplier,
        })

        # Patch video_dataset block inside nested trainer.data config
        try:
            video_ds = (
                template["trainer"]["data"]["train"]["datasets"][0]
                ["dataset"]["datasets"][0]["video_dataset"]
            )
            video_ds["_target_"] = "training.dataset.vos_raw_dataset.SA1BRawDataset"
            video_ds.update({
                "img_folder": "${dataset.img_folder}",
                "gt_folder": "${dataset.gt_folder}",
                "file_list_txt": "${dataset.file_list_txt}",
                "num_frames": 1,
                "mask_area_frac_thresh": video_ds.get("mask_area_frac_thresh", 1.1),
                "uncertain_iou": video_ds.get("uncertain_iou", -1),
            })
            for stale_key in ["is_palette", "single_object_mode", "truncate_video", "frames_sampling_mult"]:
                video_ds.pop(stale_key, None)

            sampler = (
                template["trainer"]["data"]["train"]["datasets"][0]
                ["dataset"]["datasets"][0]["sampler"]
            )
            sampler.update({"num_frames": 1, "max_num_objects": max_num_objects})
        except (KeyError, IndexError, TypeError):
            pass  # template may differ; leave unmodified

        # Patch model prompt probabilities
        try:
            model_cfg = template["trainer"]["model"]
            model_cfg.update({
                "prob_to_use_pt_input_for_train": 1.0,
                "prob_to_use_box_input_for_train": 1.0,
                "prob_to_sample_from_gt_for_train": 0.5,
                "num_frames_to_correct_for_train": 1,
                "rand_frames_to_correct_for_train": False,
                "num_init_cond_frames_for_train": 1,
                "rand_init_cond_frames_for_train": False,
                "add_all_frames_to_correct_as_cond": True,
                "num_frames_to_correct_for_eval": 1,
                "num_init_cond_frames_for_eval": 1,
            })
        except (KeyError, TypeError):
            pass

        # Patch checkpoint
        try:
            ckpt = template["trainer"]["checkpoint"]
            ckpt["save_dir"] = "${launcher.experiment_log_dir}/checkpoints"
            ckpt["save_freq"] = checkpoint_save_freq
            ckpt["model_weight_initializer"]["state_dict"]["checkpoint_path"] = str(
                Path(init_checkpoint_fpath).expanduser().resolve()
            )
        except (KeyError, TypeError):
            pass

        # Patch launcher
        template.setdefault("launcher", {})
        template["launcher"].update({
            "num_nodes": num_nodes,
            "gpus_per_node": num_gpus,
            "experiment_log_dir": str(workdir),
        })
        template.setdefault("submitit", {})
        template["submitit"]["use_cluster"] = use_cluster

        if config_overrides:
            template = _deep_update(template, config_overrides)

        # Write config into repo (Hydra requires it to be inside the search path)
        logical_name = f"kcd_{self.variant.replace('.', '_')}_{workdir.name}"
        hydra_config_name = f"configs/kcd_training/{logical_name}.yaml"
        repo_config_fpath = repo_dpath / "sam2" / hydra_config_name
        _dump_hydra_global_yaml(template, repo_config_fpath)

        # Also save a standalone copy in the workdir for reproducibility
        workdir_config_fpath = generated_dpath / "train_sam2.yaml"
        _dump_hydra_global_yaml(template, workdir_config_fpath)

        expected_ckpt = workdir / "checkpoints" / "checkpoint.pt"
        metadata = {
            "repo_config_fpath": str(repo_config_fpath),
            "hydra_config_name": hydra_config_name,
            "workdir_config_fpath": str(workdir_config_fpath),
            "expected_checkpoint_fpath": str(expected_ckpt),
            "repo_dpath": str(repo_dpath),
            "num_gpus": num_gpus,
        }
        (generated_dpath / "train_sam2_metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )
        return metadata

    def launch(self, config_metadata: dict) -> Path:
        """Subprocess SAM2's ``training/train.py`` with the generated config.

        Args:
            config_metadata: Dict returned by :meth:`generate_training_config`.

        Returns:
            Path to the expected trained checkpoint.
        """
        repo_dpath = Path(config_metadata["repo_dpath"])
        train_script = repo_dpath / "training" / "train.py"
        if not train_script.exists():
            raise FileNotFoundError(
                f"SAM2 training script not found: {train_script}"
            )
        num_gpus = int(config_metadata.get("num_gpus", 1))
        command = [
            sys.executable,
            str(train_script),
            "-c",
            config_metadata["hydra_config_name"],
            "--num-gpus",
            str(num_gpus),
        ]
        subprocess.run(command, cwd=str(repo_dpath), check=True)
        return Path(config_metadata["expected_checkpoint_fpath"])

    def find_checkpoint(self, workdir: str | Path) -> Optional[Path]:
        """Return the best available checkpoint in the SAM2 workdir."""
        workdir = Path(workdir).expanduser()
        ckpt_dir = workdir / "checkpoints"
        if ckpt_dir.is_dir():
            candidates = sorted(ckpt_dir.glob("*.pt")) + sorted(ckpt_dir.glob("*.pth"))
            if candidates:
                return candidates[-1]
        # Fallback: workdir root
        root_cands = sorted(workdir.glob("*.pt")) + sorted(workdir.glob("*.pth"))
        return root_cands[-1] if root_cands else None

    def build_segmenter(
        self,
        workdir: str | Path,
        *,
        device: str = "cuda:0",
        checkpoint_fpath: Optional[str | Path] = None,
        segmenter_cfg: Optional[dict] = None,
    ) -> SAM2Segmenter:
        """Build an inference SAM2Segmenter from a trained workdir.

        Args:
            workdir: Trainer workdir (or a directory containing a checkpoint).
            device: Torch device.
            checkpoint_fpath: Explicit checkpoint path; auto-detected if None.
            segmenter_cfg: Extra segmenter config overrides.

        Returns:
            Ready-to-use :class:`SAM2Segmenter` instance.
        """
        if checkpoint_fpath is None:
            checkpoint_fpath = self.find_checkpoint(workdir)
        if checkpoint_fpath is None:
            raise FileNotFoundError(f"No SAM2 checkpoint found in {workdir}")

        cfg = dict(self.variant_cfg)
        cfg["checkpoint_fpath"] = str(checkpoint_fpath)
        cfg["device"] = device
        if segmenter_cfg:
            cfg.update(segmenter_cfg)
        return SAM2Segmenter(cfg)


# ---------------------------------------------------------------------------
# kwconf CLI
# ---------------------------------------------------------------------------

def _build_train_cli():
    import kwconf

    class SAM2TrainConfig(kwconf.Config):
        """Fine-tune a SAM2 segmenter from kwcoco splits."""

        train_kwcoco = kwconf.Value(None, required=True, help="training kwcoco path")
        vali_kwcoco = kwconf.Value(None, required=True, help="validation kwcoco path")
        workdir = kwconf.Value(None, required=True, help="output directory for checkpoints and configs")
        variant = kwconf.Value("sam2.1_hiera_base_plus",
                             help=f"SAM2 variant; one of {list(SAM2Trainer.VARIANTS)}")
        init_checkpoint = kwconf.Value(None, required=True,
                                     help="initial SAM2 checkpoint (.pt) to fine-tune from")
        category_names = kwconf.Value(None, help="comma-separated category names to train on (default=all)")
        resolution = kwconf.Value(1024, parser=int)
        train_batch_size = kwconf.Value(1, parser=int)
        num_epochs = kwconf.Value(20, parser=int)
        base_lr = kwconf.Value(5e-6, parser=float)
        vision_lr = kwconf.Value(3e-6, parser=float)
        max_num_objects = kwconf.Value(8, parser=int)
        num_gpus = kwconf.Value(1, parser=int)
        num_train_workers = kwconf.Value(4, parser=int)

        @classmethod
        def main(cls, argv=1, **kwargs):
            from kwcoco_detector_kit.data.sam2_export import export_sam2_training_splits

            config = cls.cli(argv=argv, data=kwargs, strict=True)
            workdir = Path(config.workdir).expanduser().resolve()
            workdir.mkdir(parents=True, exist_ok=True)

            cat_names = None
            if config.category_names:
                cat_names = [s.strip() for s in str(config.category_names).split(",")]

            prepared = export_sam2_training_splits(
                train_kwcoco=config.train_kwcoco,
                vali_kwcoco=config.vali_kwcoco,
                output_dpath=workdir / "prepared_data" / "sam2",
                category_names=cat_names,
            )

            trainer = SAM2Trainer(variant=str(config.variant))
            metadata = trainer.generate_training_config(
                prepared=prepared,
                workdir=workdir,
                init_checkpoint_fpath=config.init_checkpoint,
                resolution=int(config.resolution),
                train_batch_size=int(config.train_batch_size),
                num_epochs=int(config.num_epochs),
                base_lr=float(config.base_lr),
                vision_lr=float(config.vision_lr),
                max_num_objects=int(config.max_num_objects),
                num_gpus=int(config.num_gpus),
                num_train_workers=int(config.num_train_workers),
            )
            ckpt_fpath = trainer.launch(metadata)
            print(f"SAM2 fine-tuning complete. Checkpoint: {ckpt_fpath}")
            return 0

    return SAM2TrainConfig


__cli__ = _build_train_cli()
