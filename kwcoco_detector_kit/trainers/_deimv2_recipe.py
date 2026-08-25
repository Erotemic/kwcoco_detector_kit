"""Read DEIMv2's training recipe from the selected upstream config.

Why this exists
---------------
The kit used to reconstruct DEIMv2's schedule from constants written into
``trainers/deimv2.py``. Those constants had drifted from the vendored upstream
configs, and nobody noticed because nothing compared them:

- ``_UPSTREAM_AUG_POLICY_EPOCHS = (4, 78, 148)`` over 150 epochs was the
  HGNETV2-N recipe -- the sea-lion project's variant -- applied universally,
  with its true total of 160 mis-transcribed as 150. DINOv3-X is
  ``[4, 29, 50]`` over 58. So every fish run scaled a schedule belonging to a
  different backbone family, from a total that was wrong even for that family.
- ``weight_decay`` was hardcoded to ``1e-4`` for all DINOv3 variants, but
  upstream uses ``1.25e-4`` for L and X.
- ``mixup_epochs`` / ``copyblend_epochs`` were never emitted at all, so a
  14-epoch run silently inherited upstream's ``[4, 29]`` / ``[4, 50]`` and
  never reached the clean final stage the recipe is built around.
- ``matcher_change_epoch`` was likewise never emitted, inheriting 45 and being
  unreachable on every short schedule.

Duplicating upstream's numbers is what caused all four. A per-variant table in
the kit would fix today's values and preserve the failure mode, so instead we
read the merged upstream config and scale it. If upstream changes, or a new
variant is added, the recipe follows automatically.

Scaling
-------
Every landmark is expressed as a fraction of ``total_epochs`` and re-applied to
the target schedule, so the shape of the recipe is preserved rather than
approximated. For DINOv3-X (58 epochs) at 14 epochs:

    flat_epoch      29 -> 7      policy    [4, 29, 50] -> [1, 7, 12]
    no_aug_epoch     8 -> 2      mixup          [4, 29] -> [1, 7]
    stop_epoch      50 -> 12     copyblend      [4, 50] -> [1, 12]
    matcher_change  45 -> 11

Note ``stop_epoch == aug_policy_epochs[-1]`` upstream, deliberately: the epoch
that ends heavy augmentation is also the epoch that enters the final EMA stage
(``det_solver.py:83-86``). That coincidence is the recipe working as designed.
An earlier kit change treated *any* boundary landing on ``stop_epoch`` as a bug
and nudged it away; only the boundary that TURNS augmentation ON must be kept
clear of it, which is the gen002 failure mode.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import yaml

INCLUDE_KEY = "__include__"


def load_upstream_config(fpath: str) -> Dict[str, Any]:
    """Include-aware YAML load, mirroring DEIMv2's ``load_config`` semantics.

    Reimplemented kit-side for two reasons. It keeps config generation free of
    any DEIMv2 import (the module contract in ``trainers/deimv2.py``), and it
    avoids upstream's ``load_config(file_path, cfg=dict())`` mutable default
    argument, which accumulates state across calls -- resolving two variants in
    one process would otherwise blend them.
    """
    return _load(fpath, {})


def _load(fpath: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    ext = os.path.splitext(fpath)[1]
    if ext not in (".yml", ".yaml"):
        raise ValueError(f"expected a yaml config, got {fpath!r}")
    with open(fpath) as fh:
        file_cfg = yaml.load(fh, Loader=yaml.Loader)
    if file_cfg is None:
        return cfg
    for base in list(file_cfg.get(INCLUDE_KEY, [])):
        if base.startswith("~"):
            base = os.path.expanduser(base)
        if not os.path.isabs(base):
            base = os.path.join(os.path.dirname(fpath), base)
        _merge(cfg, _load(base, {}))
    return _merge(cfg, file_cfg)


def _merge(dct: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in other.items():
        if k in dct and isinstance(dct[k], dict) and isinstance(v, dict):
            _merge(dct[k], v)
        else:
            dct[k] = v
    return dct


@dataclass(frozen=True)
class DEIMv2Recipe:
    """The schedule DEIMv2 actually ships for a given variant."""
    total_epochs: int
    flat_epoch: int
    no_aug_epoch: int
    aug_policy_epochs: Tuple[int, int, int]
    mixup_epochs: Tuple[int, int]
    copyblend_epochs: Tuple[int, int]
    stop_epoch: int
    matcher_change_epoch: int
    weight_decay: float

    def validate(self) -> "DEIMv2Recipe":
        e0, e1, e2 = self.aug_policy_epochs
        if not (0 <= e0 <= e1 <= e2):
            raise ValueError(f"aug policy not ordered: {self.aug_policy_epochs}")
        if e2 > self.total_epochs:
            raise ValueError(f"aug policy ends past the schedule: {e2} > {self.total_epochs}")
        if self.flat_epoch > self.total_epochs:
            raise ValueError(f"flat_epoch {self.flat_epoch} > total {self.total_epochs}")
        return self


def _repair_flat_epoch(flat_epoch: int, total: int, e1: int) -> int:
    """Work around an upstream typo in deimv2_hgnetv2_n_coco.yml.

    That config sets ``flat_epoch: 7800`` against ``epoches: 160``. Under
    FlatCosineLR a flat_epoch beyond the schedule means the LR never anneals at
    all, which is plainly not the intent -- the neighbouring comment reads
    "4 + epoch // 2".

    Every correctly-configured variant sets ``flat_epoch`` equal to the MIDDLE
    augmentation boundary:

        dinov3_x  flat 29  policy [4, 29, 50]
        dinov3_s  flat 64  policy [4, 64, 120]
        dinov3_m  flat 49  policy [4, 49, 90]
        dinov3_l  flat 34  policy [4, 34, 60]

    So hgnetv2_n's value should be 78, and 7800 is a transcription slip. Fall
    back to ``e1`` rather than propagating it, because faithfully carrying it
    would silently give every HGNetv2 run -- the sea-lion project's variant --
    a constant learning rate for its entire schedule.
    """
    if 0 < flat_epoch <= total:
        return flat_epoch
    return e1


def extract_recipe(upstream_cfg_fpath: str) -> DEIMv2Recipe:
    """Pull the recipe out of a merged upstream config."""
    cfg = load_upstream_config(upstream_cfg_fpath)
    train_ds = cfg["train_dataloader"]["dataset"]
    policy = train_ds["transforms"]["policy"]["epoch"]
    collate = cfg["train_dataloader"]["collate_fn"]
    matcher = cfg["DEIMCriterion"]["matcher"]
    _total = int(cfg["epoches"])
    _e1 = int(policy[1])
    return DEIMv2Recipe(
        total_epochs=_total,
        flat_epoch=_repair_flat_epoch(int(cfg["flat_epoch"]), _total, _e1),
        no_aug_epoch=int(cfg["no_aug_epoch"]),
        aug_policy_epochs=tuple(int(v) for v in policy),
        mixup_epochs=tuple(int(v) for v in collate["mixup_epochs"]),
        copyblend_epochs=tuple(int(v) for v in collate["copyblend_epochs"]),
        stop_epoch=int(collate["stop_epoch"]),
        matcher_change_epoch=int(matcher["matcher_change_epoch"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    ).validate()


def scale_recipe(recipe: DEIMv2Recipe, num_epochs: int) -> DEIMv2Recipe:
    """Rescale a recipe's landmarks to ``num_epochs``, preserving its shape.

    Clamps keep the invariants that make the schedule meaningful even on very
    short smoke-test runs, where rounding would otherwise collapse stages onto
    each other:

      * ``0 <= e0 <= e1 <= e2 <= num_epochs``
      * at least one epoch of heavy augmentation, when the schedule has room
      * ``stop_epoch == e2`` (upstream's deliberate coupling)
      * ``matcher_change_epoch < num_epochs`` so it is actually reachable
    """
    num_epochs = max(1, int(num_epochs))
    total = max(1, int(recipe.total_epochs))
    if num_epochs == total:
        return recipe
    r = num_epochs / float(total)

    def s(v: int) -> int:
        return int(round(v * r))

    e0, e1, e2 = (s(v) for v in recipe.aug_policy_epochs)
    # The final NoAug stage must run, so e2 cannot reach the end.
    e2 = min(max(e2, 1), max(1, num_epochs - 1))
    # Augmentation must start before it ends, and after at least one warmup
    # epoch -- an aug boundary at epoch 0 has nothing to warm up from.
    e0 = min(max(e0, 1), e2)
    e1 = min(max(e1, e0), e2)

    flat = min(max(s(recipe.flat_epoch), 1), max(1, num_epochs - 1))
    no_aug = min(max(s(recipe.no_aug_epoch), 1), max(1, num_epochs - 1))
    matcher = min(max(s(recipe.matcher_change_epoch), 0), max(0, num_epochs - 1))

    return DEIMv2Recipe(
        total_epochs=num_epochs,
        flat_epoch=flat,
        no_aug_epoch=no_aug,
        aug_policy_epochs=(e0, e1, e2),
        mixup_epochs=(e0, e1),
        copyblend_epochs=(e0, e2),
        stop_epoch=e2,
        matcher_change_epoch=matcher,
        weight_decay=recipe.weight_decay,
    ).validate()
