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

#: The single upstream (flat_epoch, epoches) pair known to be a transcription
#: slip -- deimv2_hgnetv2_n_coco.yml:65,68. Anything else out of range is a NEW
#: upstream problem and must fail validation loudly rather than be repaired.
_KNOWN_BAD_FLAT_EPOCH = (7800, 160)


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

    Deliberately narrow: this fires ONLY on the exact known-bad pair, so a
    future upstream mistake surfaces as a loud validation failure instead of
    being silently absorbed by a general "clamp anything out of range" rule.
    """
    if (flat_epoch, total) == _KNOWN_BAD_FLAT_EPOCH:
        return e1
    return flat_epoch


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
    """Rescale a recipe to ``num_epochs``, scaling EVERY field independently.

    An earlier version reconstructed ``mixup_epochs``, ``copyblend_epochs`` and
    ``stop_epoch`` from the three policy boundaries, on the assumption that
    upstream always couples them. It does for DINOv3, and for hgnetv2 l/m/s/x.
    It does not for the rest, and the assumption silently corrupted them:

      hgnetv2_n      copyblend is (4, 78), NOT (4, e2=148)
      atto/femto/pico  mixup and copyblend are (40000, 15000) -- start AFTER
                       end, i.e. deliberately DISABLED -- and stop_epoch is
                       468 while e2 is 400

    Rebuilding those from the policy would have re-enabled augmentations
    upstream turned off and moved a stage boundary on four of twelve variants.
    So each field is scaled from its own value and nothing is inferred.

    Clamping is applied only where a value must be in range to be meaningful:

      * the policy triple, which drives a four-stage state machine and must
        satisfy ``0 < e0 <= e1 <= e2 <= num_epochs - 1`` so the final NoAug
        stage actually runs;
      * ``stop_epoch``, ``flat_epoch``, ``no_aug_epoch`` and
        ``matcher_change_epoch``, which must fall inside the schedule to fire.

    ``mixup_epochs`` and ``copyblend_epochs`` are deliberately NOT clamped:
    clamping them into range is precisely what would enable a disabled
    augmentation. Scaling alone preserves the sentinel, since 40000 * (14/500)
    is still 1120 -- far outside a 14-epoch run.

    The guarantee is one-directional, by design. A DISABLED window can never
    become enabled. An ENABLED window may collapse to empty (start == end) on a
    schedule with no room for it -- a 2-epoch smoke test -- which turns the
    augmentation off rather than corrupting it, and is the safe failure.
    """
    num_epochs = max(1, int(num_epochs))
    total = max(1, int(recipe.total_epochs))
    if num_epochs == total:
        return recipe
    r = num_epochs / float(total)

    def s(v: int) -> int:
        return int(round(v * r))

    def s_pair(pair):
        """Scale an augmentation window, preserving a disabled sentinel.

        For an ENABLED window the start is floored at 1, matching the rule
        applied to the policy's own e0: upstream never starts augmenting in
        epoch 0, and rounding a small start (4/132 -> 0.35) down to 0 would fire
        mixup during the NoAug warmup epoch the policy is simultaneously
        declaring. A DISABLED window (start >= end) is left exactly as scaled,
        because flooring its start is what would drag it back into range.
        """
        lo, hi = s(int(pair[0])), s(int(pair[1]))
        if int(pair[0]) < int(pair[1]):
            lo = max(1, lo)
        return (lo, hi)

    e0, e1, e2 = (s(v) for v in recipe.aug_policy_epochs)
    e2 = min(max(e2, 1), max(1, num_epochs - 1))
    e0 = min(max(e0, 1), e2)
    e1 = min(max(e1, e0), e2)

    stop = min(max(s(recipe.stop_epoch), 1), num_epochs)
    flat = min(max(s(recipe.flat_epoch), 1), max(1, num_epochs - 1))
    no_aug = min(max(s(recipe.no_aug_epoch), 1), max(1, num_epochs - 1))
    matcher = min(max(s(recipe.matcher_change_epoch), 0), max(0, num_epochs - 1))

    return DEIMv2Recipe(
        total_epochs=num_epochs,
        flat_epoch=flat,
        no_aug_epoch=no_aug,
        aug_policy_epochs=(e0, e1, e2),
        mixup_epochs=s_pair(recipe.mixup_epochs),
        copyblend_epochs=s_pair(recipe.copyblend_epochs),
        stop_epoch=stop,
        matcher_change_epoch=matcher,
        weight_decay=recipe.weight_decay,
    ).validate()


#: Upstream's own "never run this" encoding, used verbatim by
#: deimv2_hgnetv2_{atto,femto,pico}_coco.yml. Reused rather than invented so a
#: reader who greps for it lands on the same convention.
DISABLED_WINDOW = (40000, 15000)


def disable_compositing(recipe: DEIMv2Recipe) -> DEIMv2Recipe:
    """Turn off mixup and copyblend, keeping the rest of the schedule.

    Mixup and copyblend paste content from one image into another. On COCO,
    where every image is an independent scene, that manufactures useful
    novelty. On a corpus of TILES cut from video frames it manufactures
    something else: a crop of a reef with a second reef's fish composited into
    it, at a scale and lighting the sensor never produced. The model is asked
    to detect objects in scenes that cannot occur, and the tiling has already
    supplied the crop diversity these ops exist to synthesise.

    Uses upstream's disabled sentinel rather than deleting the keys, so the
    generated config still states the decision explicitly and
    ``augmentation_is_disabled`` reports it.
    """
    from dataclasses import replace
    return replace(recipe,
                   mixup_epochs=DISABLED_WINDOW,
                   copyblend_epochs=DISABLED_WINDOW)


def retarget_tail(recipe: DEIMv2Recipe, tail_epochs: int) -> DEIMv2Recipe:
    """Re-place the landmarks so the run ends with ``tail_epochs`` clean epochs.

    Proportional scaling preserves upstream's SHAPE, which is the right default
    but not always the right schedule. At 58 epochs DINOv3-X ends with 8 epochs
    past ``stop_epoch`` -- the stage-2 phase where augmentation is off, EMA is
    restarted from the stage-1 best, and the model consolidates. Scaled to 28
    epochs that tail becomes 4, and to 14 epochs it becomes 2. The phase that
    does the consolidating shrinks exactly as the schedule gets shorter, which
    is backwards: a shorter run has LESS opportunity to consolidate, not more.

    This fixes the tail at an absolute length and fits the primary phase into
    what remains, keeping every landmark at upstream's own RATIO within that
    phase rather than at a hand-chosen epoch:

        stop_epoch  = num_epochs - tail_epochs
        no_aug      = tail_epochs
        e0, e1      = upstream's e0/e2 and e1/e2 ratios, applied to stop_epoch
        flat_epoch  = e1            (upstream's convention -- see _repair_flat_epoch)
        matcher     = upstream's matcher/stop ratio, applied to stop_epoch

    For DINOv3-X at 28 epochs with an 8-epoch tail this yields policy
    [2, 12, 20], flat 12, stop 20, matcher 18, no_aug 8 -- roughly 60k primary
    updates at batch 32 x 3000/epoch, which is where gen006 actually peaked,
    followed by 24k updates of the consolidation phase.
    """
    total = max(1, int(recipe.total_epochs))
    tail = int(tail_epochs)
    if tail < 1:
        raise ValueError(f"tail_epochs must be >= 1, got {tail}")
    if tail >= total:
        raise ValueError(
            f"tail_epochs {tail} leaves no primary phase in {total} epochs")
    stop = total - tail
    e0_u, e1_u, e2_u = recipe.aug_policy_epochs
    if e2_u <= 0:
        raise ValueError(f"cannot retarget a degenerate policy {recipe.aug_policy_epochs}")
    e0 = min(max(int(round(stop * e0_u / e2_u)), 1), stop)
    e1 = min(max(int(round(stop * e1_u / e2_u)), e0), stop)
    matcher = min(max(int(round(stop * recipe.matcher_change_epoch / e2_u)), 0),
                  total - 1)
    from dataclasses import replace
    return replace(
        recipe,
        aug_policy_epochs=(e0, e1, stop),
        flat_epoch=min(max(e1, 1), max(1, total - 1)),
        no_aug_epoch=min(max(tail, 1), max(1, total - 1)),
        stop_epoch=stop,
        matcher_change_epoch=matcher,
    ).validate()


def augmentation_is_disabled(window) -> bool:
    """Upstream encodes 'never run this' as a start at/after the end.

    atto/femto/pico use ``(40000, 15000)``. Any scaling of that pair keeps the
    relationship, which is what makes it safe to carry through unclamped.
    """
    return int(window[0]) >= int(window[1])
