"""
Round-based hard-negative-mining driver.

Layout per round::

  $KCD_ROOT/rounds/round<N>/
    train_round.kwcoco.zip       merged pos + (rN-0 random | rN-N hard) negs
    hard_negs.kwcoco.zip         this round's mined hards (input to round N+1)
    kcd_root/runs/<cid>/         per-candidate trainer workdir for this round

Failure #14: the orchestration layer coerces ``train_policy=multiscale ->
fixed`` when the trainer plugin reports ``supports_dynamic_input(variant)
== False``. This is the second line of defense; the trainer plugin's
config generator also enforces the constraint at YAML build time.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import scriptconfig as scfg

from kwcoco_detector_kit.trainers._registry import get_trainer


class RoundLoopConfig(scfg.DataConfig):
    """Run a round-based train -> mine -> train loop."""

    pos_tiles_kwcoco = scfg.Value(None, required=True, help="positive-tile kwcoco bundle")
    neg_tiles_kwcoco = scfg.Value(None, required=True, help="negative-tile kwcoco bundle")
    vali_kwcoco = scfg.Value(None, required=True, help="validation kwcoco bundle")
    test_kwcoco = scfg.Value(None, required=True, help="test kwcoco bundle")
    kcd_root = scfg.Value(None, help="$KCD_ROOT — workspace for rounds/")

    trainer = scfg.Value("mock_tiny")
    variant = scfg.Value("mock_tiny")
    input_hw = scfg.Value([256, 256])
    train_policy = scfg.Value("fixed")
    category_name = scfg.Value("widget")
    num_classes = scfg.Value(1)

    num_rounds = scfg.Value(3)
    round0_neg_over_pos = scfg.Value(3.0)
    mine_score_thresh = scfg.Value(0.30)
    max_hard_per_round = scfg.Value(5000)
    # Mining budget passed to data.mine (see MineConfig for semantics).
    # 0 = score every negative tile (legacy behavior); ~30000-100000 is
    # the recommended range for million-tile pools.
    mine_max_candidates = scfg.Value(0)
    mine_candidate_strategy = scfg.Value("stratified_by_image")
    mine_candidate_seed = scfg.Value(0)

    num_epochs = scfg.Value(2)
    batch_size = scfg.Value(2)
    val_batch_size = scfg.Value(2)
    lr = scfg.Value(1e-2)
    backbone_lr = scfg.Value(1e-2)
    use_amp = scfg.Value(False)
    scale_tier = scfg.Value("S")
    num_gpus = scfg.Value(1)
    init_checkpoint = scfg.Value(
        None,
        help=(
            "Optional path to a pretrained detector checkpoint (e.g. "
            "deimv2_<variant>_coco.pth) used as the round-0 init. Rounds "
            "1+ automatically resume from the prior round's best_stg2.pth "
            "(or best_stg1.pth fallback) and ignore this value."
        ),
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


def _coerce_policy_for_variant(trainer, variant: str, requested_policy: str) -> str:
    """Failure #14: coerce multiscale -> fixed if the variant doesn't support it."""
    if requested_policy == "fixed":
        return requested_policy
    try:
        ok = bool(trainer.supports_dynamic_input(variant))
    except Exception:
        ok = True
    if not ok:
        warnings.warn(
            f"round_loop: variant {variant!r} reports "
            f"supports_dynamic_input=False; coercing train_policy "
            f"{requested_policy!r} -> 'fixed' (failure #14)."
        )
        return "fixed"
    return requested_policy


def _merge_round(*, pos_kwcoco: Path, neg_kwcoco: Path, dst: Path,
                 neg_over_pos: float, round_index: int, category_name: str,
                 seed: int = 0):
    """In-process call into data.merge.run."""
    from kwcoco_detector_kit.data.merge import MergeConfig, run as merge_run

    cfg = MergeConfig.cli(
        argv=False,
        data={
            "pos_kwcoco": str(pos_kwcoco),
            "neg_kwcoco": str(neg_kwcoco),
            "dst": str(dst),
            "category_name": category_name,
            "neg_over_pos": float(neg_over_pos),
            "seed": int(seed),
            "round_index": int(round_index),
        },
    )
    merge_run(cfg)


def _mine_round(*, neg_kwcoco: Path, workdir: Path, dst: Path,
                trainer_name: str, score_thresh: float, max_hard: int,
                max_candidates: int = 0,
                candidate_strategy: str = "stratified_by_image",
                candidate_seed: int = 0):
    """In-process call into data.mine.run."""
    from kwcoco_detector_kit.data.mine import MineConfig, run as mine_run

    cfg = MineConfig.cli(
        argv=False,
        data={
            "neg_kwcoco": str(neg_kwcoco),
            "workdir": str(workdir),
            "dst": str(dst),
            "trainer": trainer_name,
            "score_thresh": float(score_thresh),
            "max_hard_per_round": int(max_hard),
            "max_candidates": int(max_candidates),
            "candidate_strategy": str(candidate_strategy),
            "candidate_seed": int(candidate_seed),
        },
    )
    mine_run(cfg)


def _train_round(trainer, *, train_kwcoco: Path, vali_kwcoco: Path, workdir: Path,
                 variant: str, input_hw, train_policy: str, num_classes: int,
                 batch_size: int, val_batch_size: int, num_epochs: int,
                 lr: float, backbone_lr: float, use_amp: bool, scale_tier: str,
                 num_gpus: int, category_name: str,
                 init_checkpoint=None):
    cfg_fpath = trainer.generate_config(
        train_kwcoco_fpath=str(train_kwcoco),
        vali_kwcoco_fpath=str(vali_kwcoco),
        workdir=workdir,
        variant=variant,
        input_hw=tuple(input_hw),
        train_policy=train_policy,
        num_classes=int(num_classes),
        batch_size=int(batch_size),
        val_batch_size=int(val_batch_size),
        num_epochs=int(num_epochs),
        lr=float(lr),
        backbone_lr=float(backbone_lr),
        use_amp=bool(use_amp),
        init_checkpoint=str(init_checkpoint) if init_checkpoint else None,
        channels="r|g|b",
        scale_tier=str(scale_tier),
        num_gpus=int(num_gpus),
        data_format="kwcoco",
        extra={"category_name": category_name,
               "init_checkpoint": str(init_checkpoint) if init_checkpoint else ""},
    )
    trainer.launch(
        cfg_fpath,
        init_checkpoint=str(init_checkpoint) if init_checkpoint else None,
        num_gpus=int(num_gpus),
    )


def run(config):
    kcd_root = Path(
        config.kcd_root or os.environ.get("KCD_ROOT")
        or (Path.home() / "data" / "kcd")
    )
    kcd_root.mkdir(parents=True, exist_ok=True)
    rounds_root = kcd_root / "rounds"
    rounds_root.mkdir(parents=True, exist_ok=True)

    trainer_name = str(config.trainer)
    trainer = get_trainer(trainer_name)
    variant = str(config.variant)
    requested_policy = str(config.train_policy)
    effective_policy = _coerce_policy_for_variant(trainer, variant, requested_policy)

    pos_fpath = Path(str(config.pos_tiles_kwcoco))
    neg_fpath_initial = Path(str(config.neg_tiles_kwcoco))

    final_workdir = None
    neg_for_round = neg_fpath_initial
    prior_round_workdir = None
    for round_index in range(int(config.num_rounds)):
        round_dpath = rounds_root / f"round{round_index}"
        round_dpath.mkdir(parents=True, exist_ok=True)
        train_kwcoco = round_dpath / "train_round.kwcoco.zip"
        workdir = round_dpath / "runs" / f"{variant}_{int(config.input_hw[0])}x{int(config.input_hw[1])}"
        workdir.mkdir(parents=True, exist_ok=True)

        # Merge pos + (random or hard) negatives for this round
        n_over_p = (
            float(config.round0_neg_over_pos) if round_index == 0
            else 0.0   # round N>0: keep ALL mined hard negs
        )
        _merge_round(
            pos_kwcoco=pos_fpath, neg_kwcoco=neg_for_round, dst=train_kwcoco,
            neg_over_pos=n_over_p, round_index=round_index,
            category_name=str(config.category_name),
        )

        # Pick the init checkpoint for THIS round.
        #   round 0           -> --init_checkpoint (COCO-pretrained .pth)
        #   round 1, 2, ...   -> prior round's best_stg2.pth (fall back to
        #                        best_stg1.pth, then last.pth)
        if round_index == 0:
            this_init_ckpt = (
                str(config.init_checkpoint) if config.init_checkpoint else None
            )
        else:
            this_init_ckpt = None
            for cand in ("best_stg2.pth", "best_stg1.pth", "last.pth"):
                p = prior_round_workdir / cand
                if p.exists():
                    this_init_ckpt = str(p)
                    break
            if this_init_ckpt is None:
                raise FileNotFoundError(
                    f"round_loop: round {round_index} expected a checkpoint "
                    f"under {prior_round_workdir} but none of best_stg2.pth/"
                    f"best_stg1.pth/last.pth exist. Did the prior round "
                    f"train successfully?"
                )

        # Train this round
        os.environ["KCD_ROUND"] = str(round_index)
        _train_round(
            trainer, train_kwcoco=train_kwcoco, vali_kwcoco=str(config.vali_kwcoco),
            workdir=workdir, variant=variant, input_hw=config.input_hw,
            train_policy=effective_policy, num_classes=int(config.num_classes),
            batch_size=int(config.batch_size), val_batch_size=int(config.val_batch_size),
            num_epochs=int(config.num_epochs), lr=float(config.lr),
            backbone_lr=float(config.backbone_lr), use_amp=bool(config.use_amp),
            scale_tier=str(config.scale_tier), num_gpus=int(config.num_gpus),
            category_name=str(config.category_name),
            init_checkpoint=this_init_ckpt,
        )
        prior_round_workdir = workdir

        # Mine hard negatives unless this is the last round
        if round_index + 1 < int(config.num_rounds):
            hard_kwcoco = round_dpath / "hard_negs.kwcoco.zip"
            _mine_round(
                neg_kwcoco=neg_fpath_initial, workdir=workdir, dst=hard_kwcoco,
                trainer_name=trainer_name,
                score_thresh=float(config.mine_score_thresh),
                max_hard=int(config.max_hard_per_round),
                max_candidates=int(config.mine_max_candidates),
                candidate_strategy=str(config.mine_candidate_strategy),
                candidate_seed=int(config.mine_candidate_seed),
            )
            neg_for_round = hard_kwcoco

        final_workdir = workdir

    print(f"\nround_loop complete; final workdir: {final_workdir}")
    return final_workdir


__cli__ = RoundLoopConfig
