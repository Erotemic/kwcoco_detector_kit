"""Sequence- and track-aware sampling weights for video-derived tile corpora.

Why this exists
---------------
A tile corpus cut from video is not a set of independent samples. The fish
train split is 439 sequences -> 251,143 frames -> 495,514 tiles, and adjacent
frames of one sequence are near-duplicates: the same fish, the same background,
a few pixels apart. Uniform sampling over tiles therefore spends its epoch in
proportion to how long a camera happened to run, not in proportion to how much
distinct signal a sequence carries.

Two failure modes follow, and both were visible in gen006:

  * a long sequence contributes thousands of near-identical tiles, so the model
    can drive training loss down by memorising it while learning nothing
    transferable -- gen006's loss fell from 33.20 to 26.80 while vali AP fell
    from its epoch-4 peak;
  * a short sequence with unusual conditions is seen a handful of times per
    epoch and is effectively drowned out.

This module reweights the draw instead of editing the corpus. Nothing is
deleted: every tile keeps a strictly positive weight, so a rare sequence is
sampled MORE and a dominant one LESS, but the full corpus stays reachable
across epochs. That is the difference between rebalancing and discarding, and
it matters because the "redundant" neighbouring frames are still real
augmentation-like variation -- just badly over-weighted.

How it plugs in
---------------
Nothing new is needed on the training side. The kit already has the whole
path: :mod:`kwcoco_detector_kit.data.balanced_sampler` writes a per-index
weight sidecar, ``sampler_from_weights_file`` builds a rank-aware
``DistributedWeightedRandomSampler``, and the patched DEIMv2 solver
(``tpl/DEIMv2/engine/solver/_solver.py``) swaps it in when the generated config
carries ``kcd_sample_weights_fpath``. This module only computes a DIFFERENT
weight vector for that same sidecar -- keyed on sequence and track rather than
on class -- and is deliberately independent of ``BalancedSampleForest``, whose
``index_weights()`` contract is not yet shipped by the submodule.

The weights
-----------
For tile ``i`` in sequence ``s(i)`` carrying tracks ``T_i``::

    w_seq(i)   = count_seq[s(i)] ** -seq_alpha
    w_track(i) = mean over t in T_i of count_track[t] ** -track_alpha
    w(i)       = w_seq(i) * w_track(i)          (then normalised to sum 1)

``seq_alpha`` interpolates continuously rather than switching modes:

    0.0   proportional -- exactly today's uniform-over-tiles behaviour
    0.5   square-root damping; a 100x longer sequence gets 10x the mass
    1.0   flat -- every sequence contributes equally regardless of length

Partial flattening is usually right. ``seq_alpha=1.0`` is not obviously the
goal: a sequence that is genuinely ten times longer often does carry more
distinct content than one that is not, and full flattening throws that away
along with the redundancy. Pick the value from the measured distribution
(``measure`` below), not from taste.

``track_alpha`` does the same within a sequence, so a single fish tracked
across 900 frames stops outvoting nine fish seen in 100 frames each.

Empty (negative) tiles have no tracks. They are scaled by ``empty_weight``
RELATIVE to the mean non-empty track term, so ``empty_weight=1.0`` means "an
empty tile is as likely as an average annotated one" and the knob does not
silently change meaning when ``track_alpha`` moves.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Sequence

__all__ = [
    "EMPTY_TRACK",
    "summarize_groups",
    "flatten_weights",
    "combine_weights",
    "cap_oversample",
    "load_tile_index",
    "compute_sequence_weights",
]

#: Sentinel group for tiles carrying no annotation.
EMPTY_TRACK = "<empty>"


# ---------------------------------------------------------------------------
# Pure numeric core -- no kwcoco, no numpy, so it is testable anywhere
# ---------------------------------------------------------------------------


def summarize_groups(groups: Sequence[Hashable], *, top_k: int = 15) -> Dict[str, Any]:
    """Describe how unevenly ``groups`` divides the corpus.

    ``groups[i]`` is the group (sequence, track, source frame ...) that index
    ``i`` belongs to. Returned fields:

    ``gini``
        0 = every group contributes equally, 1 = one group is everything.
    ``effective_count``
        ``1 / sum(share^2)`` -- the inverse Simpson index, i.e. the number of
        EQUALLY-SIZED groups that would produce the same concentration. Read
        it against ``n_groups``: 439 sequences with an effective count of 60
        means the corpus behaves like 60 sequences, not 439. This is the
        single most useful number here, because it says what the redundancy
        actually costs in units the reader already understands.
    ``imbalance_ratio``
        largest group / smallest group.
    """
    counts: Dict[Hashable, int] = {}
    for g in groups:
        counts[g] = counts.get(g, 0) + 1
    n = len(groups)
    if n == 0:
        raise ValueError("no items to summarise")
    sizes = sorted(counts.values())
    k = len(sizes)
    shares = [c / n for c in sizes]

    # Gini over group sizes, via the sorted-cumulative form.
    cum = 0.0
    for rank, size in enumerate(sizes, start=1):
        cum += rank * size
    total = float(sum(sizes))
    gini = (2.0 * cum) / (k * total) - (k + 1.0) / k if k > 1 else 0.0

    eff = 1.0 / sum(s * s for s in shares) if shares else 0.0
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    return {
        "n_items": n,
        "n_groups": k,
        "mean_per_group": n / k,
        "median_per_group": sizes[k // 2],
        "min_per_group": sizes[0],
        "max_per_group": sizes[-1],
        "imbalance_ratio": sizes[-1] / sizes[0] if sizes[0] else float("inf"),
        "gini": gini,
        "effective_count": eff,
        "effective_frac": eff / k if k else 0.0,
        "top": [{"group": str(g), "count": c, "share": c / n} for g, c in ordered[:top_k]],
        "bottom": [{"group": str(g), "count": c, "share": c / n}
                   for g, c in ordered[-top_k:]],
        "share_of_top_10pct": (
            sum(sizes[-max(1, k // 10):]) / total if k else 0.0),
    }


def flatten_weights(groups: Sequence[Hashable], alpha: float) -> List[float]:
    """Per-index weight ``count[group]**-alpha``.

    ``alpha=0`` returns all-ones (proportional / unchanged); ``alpha=1``
    equalises the total mass of every group. Not normalised -- the caller
    combines several of these before normalising once.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    counts: Dict[Hashable, int] = {}
    for g in groups:
        counts[g] = counts.get(g, 0) + 1
    if alpha == 0:
        return [1.0] * len(groups)
    return [float(counts[g]) ** (-alpha) for g in groups]


def combine_weights(*vectors: Sequence[float]) -> List[float]:
    """Elementwise product, normalised to sum to 1."""
    if not vectors:
        raise ValueError("no weight vectors")
    n = len(vectors[0])
    for v in vectors:
        if len(v) != n:
            raise ValueError(
                f"weight vectors disagree on length: {[len(x) for x in vectors]}")
    out = [1.0] * n
    for v in vectors:
        for i, x in enumerate(v):
            out[i] *= float(x)
    total = sum(out)
    if not (total > 0) or math.isnan(total):
        raise ValueError("combined weights do not sum to a positive number")
    if any(w < 0 or math.isnan(w) for w in out):
        raise ValueError("weights must be finite and non-negative")
    return [w / total for w in out]


def cap_oversample(weights: Sequence[float], max_oversample: float) -> List[float]:
    """Bound how often any one index may be drawn per nominal epoch.

    Mirrors ``balanced_sampler.compute_index_weights``' cap: no index may
    exceed ``max_oversample / N`` of the mass. Capping lowers the total, so
    renormalising can push other indices back over the cap; the loop repeats
    until stable, bounded so that floating-point non-convergence cannot hang
    the process.

    This matters more here than for class balance. With ``seq_alpha=1`` a
    439-sequence corpus gives a one-tile sequence the same mass as a
    12,000-tile one, i.e. 12,000x oversampling of a single image -- straight
    back to memorisation, just of a different tile.
    """
    k = float(max_oversample)
    if k <= 0:
        raise ValueError(f"max_oversample must be > 0, got {k}")
    w = [float(x) for x in weights]
    n = len(w)
    if n == 0:
        raise ValueError("no weights")
    cap = k / n
    for _ in range(512):
        over = [i for i, x in enumerate(w) if x > cap]
        if not over:
            break
        for i in over:
            w[i] = cap
        total = sum(w)
        if total <= 0:
            raise ValueError(f"all weights capped to zero at max_oversample={k}")
        w = [x / total for x in w]
    return w


def oversample_profile(weights: Sequence[float], epoch_length: int) -> Dict[str, float]:
    """Expected draws per index in one epoch -- the sanity check on a weighting.

    ``weights[i] * epoch_length`` is how many times index ``i`` is expected to
    appear. A max far above 1 means some tile is being shown repeatedly within
    a single epoch, which is the exact thing this module exists to prevent.
    """
    n = len(weights)
    if n == 0:
        raise ValueError("no weights")
    draws = [w * float(epoch_length) for w in weights]
    hi = max(draws)
    return {
        "epoch_length": float(epoch_length),
        "n_indices": float(n),
        "expected_draws_max": hi,
        "expected_draws_mean": sum(draws) / n,
        "expected_draws_min": min(draws),
        "frac_unseen_per_epoch": sum(
            math.exp(-d) for d in draws) / n,  # P(never drawn), Poisson approx
    }


# ---------------------------------------------------------------------------
# Loading -- the three-way join
# ---------------------------------------------------------------------------


def load_tile_index(
    tiles_kwcoco_fpath,
    source_kwcoco_fpath=None,
    *,
    sequence_field: str = "sequence",
) -> Dict[str, List[Any]]:
    """Per-tile sequence and track keys, in tile-image order.

    Order is the contract. ``coco_export.export_mscoco`` walks
    ``src_dset.images()`` in order and keeps every image, so position ``i``
    here is position ``i`` in the exported MSCOCO that DEIMv2 indexes, and the
    sampler's weights are positional. A reordering upstream would silently
    misalign every weight, which is why ``compute_sequence_weights`` re-checks
    the length against the mscoco it is actually paired with.

    Sequence identity comes from the SOURCE bundle: the tiler stamps
    ``tile_source_gid`` on each tile but does not copy ``video_id``, so the
    join is tile -> source frame -> sequence. Without ``source_kwcoco_fpath``
    the source frame itself is used as the group, which still removes
    per-frame tile redundancy but not per-sequence redundancy.
    """
    tiles = json.loads(Path(tiles_kwcoco_fpath).read_text())

    gid_to_sequence: Dict[Any, Any] = {}
    if source_kwcoco_fpath is not None:
        src = json.loads(Path(source_kwcoco_fpath).read_text())
        for img in src.get("images", []):
            key = img.get(sequence_field, img.get("video_id"))
            gid_to_sequence[img["id"]] = key if key is not None else img["id"]

    tracks_by_gid: Dict[Any, List[Any]] = {}
    for ann in tiles.get("annotations", []):
        gid = ann.get("image_id")
        # A track_id is required to merge an instance across frames. Without
        # one, the annotation is its OWN track rather than being lumped in
        # with every other untracked annotation -- lumping would make one
        # giant pseudo-track and suppress exactly the tiles it appears in.
        tid = ann.get("track_id", ("ann", ann.get("id")))
        tracks_by_gid.setdefault(gid, []).append(tid)

    sequences: List[Any] = []
    tracks: List[List[Any]] = []
    source_frames: List[Any] = []
    roles: List[str] = []
    for img in tiles.get("images", []):
        gid = img["id"]
        src_gid = img.get("tile_source_gid", gid)
        source_frames.append(src_gid)
        sequences.append(gid_to_sequence.get(src_gid, src_gid))
        tracks.append(tracks_by_gid.get(gid, []))
        roles.append(img.get("tile_role", "unknown"))
    return {"sequences": sequences, "tracks": tracks,
            "source_frames": source_frames, "roles": roles}


def compute_sequence_weights(
    index: Dict[str, List[Any]],
    *,
    seq_alpha: float = 0.5,
    track_alpha: float = 0.5,
    frame_alpha: float = 0.0,
    empty_weight: float = 1.0,
    max_oversample: Optional[float] = 32.0,
) -> List[float]:
    """Combine the sequence, frame and track terms into one weight vector.

    ``frame_alpha`` damps per-source-frame tile count (a 4K frame yields more
    tiles than a 720p one); it is off by default because the tile grid is
    already fixed per frame and the effect is small next to sequence length.
    """
    sequences = index["sequences"]
    tracks = index["tracks"]
    n = len(sequences)
    if n == 0:
        raise ValueError("empty tile index")

    w_seq = flatten_weights(sequences, seq_alpha)
    w_frame = flatten_weights(index["source_frames"], frame_alpha)

    # Track counts are global: track ids in this corpus already carry the
    # sequence name ("CDFW-LakeCam-April-Tules1_0"), so they do not collide
    # across sequences. Pairing with the sequence anyway costs nothing and
    # makes the function correct for a corpus whose ids are only locally
    # unique.
    track_counts: Dict[Any, int] = {}
    for seq, tids in zip(sequences, tracks):
        for tid in tids:
            key = (seq, tid)
            track_counts[key] = track_counts.get(key, 0) + 1

    raw: List[Optional[float]] = []
    for seq, tids in zip(sequences, tracks):
        if not tids:
            raw.append(None)
            continue
        if track_alpha == 0:
            raw.append(1.0)
            continue
        raw.append(sum(float(track_counts[(seq, t)]) ** (-track_alpha)
                       for t in tids) / len(tids))

    present = [x for x in raw if x is not None]
    # Anchor the empty tiles to the MEAN annotated tile so that empty_weight
    # keeps one meaning as track_alpha varies.
    mean_present = (sum(present) / len(present)) if present else 1.0
    w_track = [float(empty_weight) * mean_present if x is None else x for x in raw]

    weights = combine_weights(w_seq, w_frame, w_track)
    if max_oversample is not None:
        weights = cap_oversample(weights, max_oversample)
    return weights


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _report(index: Dict[str, List[Any]], top_k: int = 10) -> Dict[str, Any]:
    seqs, tracks = index["sequences"], index["tracks"]
    flat = [(s, t) for s, ts in zip(seqs, tracks) for t in ts]
    out = {
        "sequence": summarize_groups(seqs, top_k=top_k),
        "source_frame": summarize_groups(index["source_frames"], top_k=top_k),
        "n_empty_tiles": sum(1 for t in tracks if not t),
    }
    if flat:
        out["track"] = summarize_groups(flat, top_k=top_k)
    return out


def _print_report(rep: Dict[str, Any]) -> None:
    for key in ("sequence", "source_frame", "track"):
        s = rep.get(key)
        if not s:
            continue
        print(f"=== {key.upper()} ===")
        print(f"  {s['n_groups']} groups over {s['n_items']} items")
        print(f"  per-group  min {s['min_per_group']}  median {s['median_per_group']}"
              f"  mean {s['mean_per_group']:.0f}  max {s['max_per_group']}")
        print(f"  imbalance {s['imbalance_ratio']:.0f}x   gini {s['gini']:.3f}")
        print(f"  effective count {s['effective_count']:.0f} of {s['n_groups']}"
              f" ({s['effective_frac']:.1%})")
        print(f"  top 10% of groups hold {s['share_of_top_10pct']:.1%}")
        for r in s["top"][:5]:
            print(f"     {r['count']:8d}  {r['share']:7.3%}  {r['group']}")
        print()


def main(argv=None) -> int:
    """``measure`` reports the imbalance; ``weights`` writes the sidecar."""
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mode", choices=["measure", "weights"])
    p.add_argument("--tiles_kwcoco", required=True,
                   help="the TILED bundle (carries tile_source_gid + track_id)")
    p.add_argument("--source_kwcoco", default=None,
                   help="the SOURCE bundle; supplies sequence identity. Without "
                        "it, tiles group by source frame only.")
    p.add_argument("--dst", default=None, help="weights sidecar (weights mode)")
    p.add_argument("--report_json", default=None)
    p.add_argument("--seq_alpha", type=float, default=0.5)
    p.add_argument("--track_alpha", type=float, default=0.5)
    p.add_argument("--frame_alpha", type=float, default=0.0)
    p.add_argument("--empty_weight", type=float, default=1.0)
    p.add_argument("--max_oversample", type=float, default=8.0)
    p.add_argument("--epoch_length", type=int, default=0,
                   help="if set, report expected draws per index per epoch")
    p.add_argument("--mscoco", default=None, help=(
        "the exported MSCOCO the sampler will actually index. When given, the "
        "weight count is checked against it -- a mismatch means the weights "
        "are positionally misaligned with the dataset, which is silent skew."))
    args = p.parse_args(argv)

    index = load_tile_index(args.tiles_kwcoco, args.source_kwcoco)
    rep = _report(index)
    _print_report(rep)

    if args.mode == "measure":
        if args.report_json:
            Path(args.report_json).write_text(json.dumps(rep, indent=2) + "\n")
            print(f"wrote {args.report_json}")
        return 0

    if not args.dst:
        p.error("weights mode needs --dst")
    weights = compute_sequence_weights(
        index,
        seq_alpha=args.seq_alpha, track_alpha=args.track_alpha,
        frame_alpha=args.frame_alpha, empty_weight=args.empty_weight,
        max_oversample=(args.max_oversample if args.max_oversample > 0 else None),
    )
    if args.mscoco:
        n_mscoco = len(json.loads(Path(args.mscoco).read_text()).get("images", []))
        if n_mscoco != len(weights):
            raise SystemExit(
                f"weight/dataset mismatch: {len(weights)} weights from "
                f"{args.tiles_kwcoco} but {n_mscoco} images in {args.mscoco}. "
                "The sampler indexes positionally, so this would train on a "
                "silently shuffled weighting.")
    meta = {
        "generator": "kwcoco_detector_kit.data.sequence_balance",
        "seq_alpha": args.seq_alpha, "track_alpha": args.track_alpha,
        "frame_alpha": args.frame_alpha, "empty_weight": args.empty_weight,
        "max_oversample": args.max_oversample,
        "tiles_kwcoco": str(args.tiles_kwcoco),
        "source_kwcoco": str(args.source_kwcoco),
        "report": rep,
    }
    if args.epoch_length:
        prof = oversample_profile(weights, args.epoch_length)
        meta["oversample_profile"] = prof
        print(f"epoch_length {args.epoch_length}: max expected draws/index "
              f"{prof['expected_draws_max']:.2f}, "
              f"{prof['frac_unseen_per_epoch']:.1%} unseen per epoch")
    # Same sidecar schema balanced_sampler.write_balance_weights emits, so
    # load_balance_weights and sampler_from_weights_file consume it unchanged.
    Path(args.dst).write_text(json.dumps({"weights": weights, "meta": meta}))
    print(f"wrote {args.dst}: {len(weights)} weights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
