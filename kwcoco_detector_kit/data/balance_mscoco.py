"""Re-balance an MSCOCO json by duplicating image entries.

Class balance for the JPEG (non-WDS) training path is fixed at
on-disk composition time. The kit's WebDataset path lets a run
re-weight bucket frequencies at sample-pick time via
:class:`WeightedChunkMix`; the JPEG path has no equivalent —
stock DEIMv2 :class:`CocoDetection` is map-style and uses
torch's default uniform-random shuffle, with no sampler
injection point in the trainer.

This module provides the JPEG-path equivalent: given an
MSCOCO json and a target class distribution, emit a new
MSCOCO json with image entries duplicated to hit the target
proportions. The assets on disk are unchanged — the new
MSCOCO just references them more (or less) often.

Why duplication instead of a runtime sampler:

* Zero changes to the DEIMv2 dataloader (kept out of the
  cross-variant submodule).
* Reproducible — the resampled MSCOCO is a literal file you
  can diff between recipes; no need to capture sampler state.
* Reversible — keep the unbalanced MSCOCO alongside; ablate
  by swapping ``train_mscoco_fpath`` in the submit script.
* Matches how the WDS path bakes buckets at write time and
  re-weights at read time: same conceptual split.

Bucket assignment per image:

* If the image has no annotations after scheme collapse, its
  bucket is ``"<empty>"``.
* Otherwise the bucket is the image's *dominant* target class
  (highest count; ties broken by ``category_names`` order).

The output preserves the input's category list and id space;
only ``images`` and ``annotations`` are regenerated (with new
ids, since duplication would otherwise produce non-unique ids).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import scriptconfig as scfg


EMPTY_BUCKET = "<empty>"


class BalanceMSCOCOConfig(scfg.DataConfig):
    """Resample an MSCOCO json to hit a target class distribution."""

    src = scfg.Value(None, position=1, help="input MSCOCO json")
    dst = scfg.Value(None, position=2, help="output MSCOCO json")

    target_distribution = scfg.Value(
        None,
        type=str,
        help=(
            "JSON object mapping bucket name -> target fraction. Bucket "
            'names are category names from the input MSCOCO, plus the '
            f'literal "{EMPTY_BUCKET}" for images with no annotations. '
            'Example for pup_vs_nonpup: '
            '\'{"<empty>": 0.5, "pup": 0.25, "nonpup": 0.25}\'. '
            "Fractions are normalized to 1.0."
        ),
    )

    target_size = scfg.Value(
        None,
        type=int,
        help=(
            "Total number of image entries in the output MSCOCO. "
            "When set explicitly, overrides max_oversample. Defaults: "
            "if max_oversample is also unset, output size matches "
            "len(src.images) (legacy behavior, may heavily oversample "
            "the rarest bucket); otherwise computed from max_oversample."
        ),
    )

    max_oversample = scfg.Value(
        None, type=int,
        help=(
            "Cap on per-sample repetition for the rarest bucket. "
            "When set (and target_size is not), target_size becomes "
            "max_oversample × min(len(bucket_b) / target_fraction_b "
            "for each b in target_distribution). The rarest bucket "
            "is repeated at most max_oversample times; more-common "
            "buckets are subsampled to match the target distribution. "
            "For typical detection class-balance use, max_oversample=1 "
            "is recommended: each positive tile seen once per epoch, "
            "negatives subsampled, epochs faster, augmentation provides "
            "stochastic diversity across repeats in later epochs. "
            "max_oversample=3 is a reasonable middle ground when the "
            "rarest bucket is very small and the model needs more "
            "exposure to per-epoch."
        ),
    )

    seed = scfg.Value(0, type=int, help="RNG seed for sampling")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


def _bucket_of_image(
    image_id: int,
    anns_by_image: Dict[int, List[dict]],
    cat_name_by_id: Dict[int, str],
    category_order: List[str],
) -> str:
    """Return the dominant-class bucket name for one image, or EMPTY_BUCKET.

    Ties are broken by `category_order` position (earlier wins).
    """
    anns = anns_by_image.get(image_id, [])
    if not anns:
        return EMPTY_BUCKET
    counts: Dict[str, int] = {}
    for a in anns:
        name = cat_name_by_id.get(a["category_id"])
        if name is None:
            # Unknown category — drop. Same semantics as the WDS
            # adapter's scheme.unmapped_policy='drop'.
            continue
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return EMPTY_BUCKET
    # Tie-break by category_order position; entries not in order are
    # appended after the known ones.
    rank = {n: i for i, n in enumerate(category_order)}
    def _key(name: str):
        return (-counts[name], rank.get(name, len(category_order)))
    return min(counts, key=_key)


def _bucket_image_ids(
    mscoco: dict,
) -> Dict[str, List[int]]:
    """Group image_ids by bucket name."""
    cat_name_by_id = {c["id"]: c["name"] for c in mscoco.get("categories", [])}
    category_order = [c["name"] for c in mscoco.get("categories", [])]
    anns_by_image: Dict[int, List[dict]] = {}
    for a in mscoco.get("annotations", []):
        anns_by_image.setdefault(a["image_id"], []).append(a)

    buckets: Dict[str, List[int]] = {}
    for img in mscoco.get("images", []):
        b = _bucket_of_image(
            img["id"], anns_by_image, cat_name_by_id, category_order,
        )
        buckets.setdefault(b, []).append(img["id"])
    return buckets


def _normalize_distribution(target: Dict[str, float]) -> Dict[str, float]:
    total = sum(float(v) for v in target.values())
    if total <= 0:
        raise ValueError(
            "target_distribution must have at least one positive weight"
        )
    return {k: float(v) / total for k, v in target.items()}


def _compute_per_bucket_counts(
    target_distribution: Dict[str, float],
    target_size: int,
) -> Dict[str, int]:
    """Largest-remainder rounding so per-bucket counts sum to target_size."""
    raw = {k: target_size * v for k, v in target_distribution.items()}
    floors = {k: int(v) for k, v in raw.items()}
    assigned = sum(floors.values())
    leftover = target_size - assigned
    # Distribute leftover to buckets with the largest fractional part.
    fracs = sorted(
        ((raw[k] - floors[k], k) for k in raw),
        key=lambda t: (-t[0], t[1]),
    )
    for _frac, k in fracs[:leftover]:
        floors[k] += 1
    return floors


def _sample_with_cap(
    pool: List[int], k: int, max_repeats: Optional[int], rng,
) -> List[int]:
    """Draw k items from pool with each item picked at most
    max_repeats times.

    * max_repeats=None: pure with-replacement (legacy behavior).
    * max_repeats=1: without-replacement, k must be <= len(pool).
    * max_repeats=K: each item picked 0..K times, k must be <= K*len(pool).
    """
    import numpy as np
    if not pool:
        raise ValueError("pool is empty")
    arr = np.asarray(pool)
    n = len(arr)
    if max_repeats is None:
        idxs = rng.randint(0, n, size=k)
        return arr[idxs].tolist()
    if k > max_repeats * n:
        raise ValueError(
            f"cannot pick {k} items from pool of {n} with "
            f"max_repeats={max_repeats}; the cap allows only "
            f"{max_repeats * n}. Lower max_oversample or pick a "
            f"target_distribution that doesn't ask for so many."
        )
    # Build a virtual extended pool of size max_repeats * n by repeating
    # the index list, then sample k *without* replacement. Each original
    # index appears at most max_repeats times in the result.
    extended = np.tile(np.arange(n), max_repeats)
    chosen = rng.choice(len(extended), size=k, replace=False)
    return arr[extended[chosen]].tolist()


def run(config: BalanceMSCOCOConfig):
    """Entry point — read src, resample, write dst."""
    import numpy as np

    src_fpath = Path(str(config.src)).expanduser().resolve()
    dst_fpath = Path(str(config.dst)).expanduser().resolve()
    if not src_fpath.exists():
        raise FileNotFoundError(src_fpath)

    raw_target = config.target_distribution
    if raw_target is None:
        raise ValueError("target_distribution is required")
    if isinstance(raw_target, str):
        try:
            target_distribution = json.loads(raw_target)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"target_distribution must be valid JSON: {e}. "
                f"Got: {raw_target!r}"
            )
    else:
        target_distribution = dict(raw_target)
    if not isinstance(target_distribution, dict):
        raise TypeError(
            f"target_distribution must be a JSON object, got "
            f"{type(target_distribution).__name__}"
        )

    mscoco = json.loads(src_fpath.read_text())
    buckets = _bucket_image_ids(mscoco)

    # Validate target keys vs available buckets.
    missing = [k for k in target_distribution if k not in buckets
               or not buckets[k]]
    if missing:
        raise ValueError(
            f"target_distribution references buckets that don't "
            f"exist or are empty in {src_fpath}: {missing}. "
            f"Available non-empty buckets: "
            f"{sorted(k for k, v in buckets.items() if v)}."
        )

    target_distribution = _normalize_distribution(target_distribution)

    # Choose target_size. Three modes:
    #
    #   1. target_size set explicitly: use it (overrides max_oversample).
    #   2. max_oversample set: target_size = K × min(len_b / f_b).
    #      The rarest bucket is repeated at most K times; common
    #      buckets are subsampled to match the target ratio.
    #   3. Neither set: legacy default = len(src.images). May
    #      heavily oversample the rarest bucket — kept for backwards
    #      compatibility but typically NOT what you want for class
    #      balancing (see max_oversample help).
    # effective_max_repeats: enforced by the sampler. None means
    # unlimited (legacy with-replacement). When target_size is set
    # explicitly we leave the cap off because the caller is taking
    # responsibility for the size and the cap could make the size
    # infeasible (e.g. target_size > len(pool) + max_oversample).
    effective_max_repeats: Optional[int] = None
    if config.target_size is not None and int(config.target_size) > 0:
        target_size = int(config.target_size)
    elif config.max_oversample is not None:
        K = int(config.max_oversample)
        if K <= 0:
            raise ValueError(
                f"max_oversample must be > 0; got {K}"
            )
        # Per-bucket "natural fit": the largest target_size at which
        # this bucket is NOT oversampled (every entry contributes at
        # most once). natural_b = len(bucket_b) / target_fraction_b.
        # The overall natural fit is the min across buckets — the
        # rarest one is the binding constraint.
        natural = min(
            len(buckets[b]) / target_distribution[b]
            for b in target_distribution
            if target_distribution[b] > 0
        )
        target_size = max(1, int(K * natural))
        effective_max_repeats = K
    else:
        target_size = len(mscoco.get("images", []))

    if target_size <= 0:
        raise ValueError(
            f"target_size must be positive; got {target_size}"
        )

    per_bucket = _compute_per_bucket_counts(target_distribution, target_size)

    rng = np.random.RandomState(int(config.seed))

    # Build the resampled image list. Re-id everything so duplicates
    # don't collide. Keep the per-image-id ann list so we can copy
    # annotations alongside duplicated images.
    src_images = {img["id"]: img for img in mscoco.get("images", [])}
    anns_by_image: Dict[int, List[dict]] = {}
    for a in mscoco.get("annotations", []):
        anns_by_image.setdefault(a["image_id"], []).append(a)

    new_images: List[dict] = []
    new_annotations: List[dict] = []
    next_img_id = 1
    next_ann_id = 1
    bucket_picks: Dict[str, int] = {}
    for bucket, n in per_bucket.items():
        if n <= 0:
            continue
        picks = _sample_with_cap(
            buckets[bucket], n, effective_max_repeats, rng,
        )
        bucket_picks[bucket] = len(picks)
        for src_img_id in picks:
            new_img = dict(src_images[src_img_id])
            old_img_id = new_img["id"]
            new_img["id"] = next_img_id
            new_img["balance_src_image_id"] = old_img_id
            new_img["balance_bucket"] = bucket
            new_images.append(new_img)
            for a in anns_by_image.get(old_img_id, []):
                new_ann = dict(a)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = next_img_id
                new_annotations.append(new_ann)
                next_ann_id += 1
            next_img_id += 1

    out = {
        "categories": list(mscoco.get("categories", [])),
        "images": new_images,
        "annotations": new_annotations,
        "info": {
            **dict(mscoco.get("info", {})),
            "balance_mscoco": {
                "source": str(src_fpath),
                "target_distribution": target_distribution,
                "target_size": int(target_size),
                "max_oversample": (
                    int(config.max_oversample)
                    if config.max_oversample is not None else None
                ),
                "per_bucket_counts": {k: int(v) for k, v in per_bucket.items()},
                "actual_bucket_picks": {k: int(v) for k, v in bucket_picks.items()},
                "seed": int(config.seed),
            },
        },
    }

    dst_fpath.parent.mkdir(parents=True, exist_ok=True)
    dst_fpath.write_text(json.dumps(out))
    print(
        f"balance_mscoco: wrote {len(new_images)} images / "
        f"{len(new_annotations)} annotations to {dst_fpath} "
        f"(target={target_distribution}, actual={bucket_picks})"
    )
    return dst_fpath


if __name__ == "__main__":
    BalanceMSCOCOConfig.main()
