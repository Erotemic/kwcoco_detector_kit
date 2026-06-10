"""Frozen selection probe: a small stratified subset of whole val frames.

The probe is the **same** ``true_tiled`` protocol bound to a smaller
dataset — real sliding-window inference with cross-tile NMS over ~50
whole validation frames — so proxy error is pure dataset subsampling,
never a procedural divergence (spec: "the probe is the same protocol on
a smaller dataset").

Construction (deterministic given the source dataset + params):

1. Per-frame per-class instance counts from the source kwcoco.
2. Rarity weights ``w_c = 1 / corpus_count_c`` — rare-positive frames
   score high; frame score = ``sum_c count_fc * w_c``.
3. Greedy coverage: ascending-frequency classes each claim their
   top-count frames until covered by ``min_frames_per_class``.
4. Remaining positive slots fill by seeded weighted sampling without
   replacement (probability ∝ frame score).
5. ``empty_frac`` of the budget comes from empty frames (seeded sample).

The probe is **frozen per scheme**: the manifest (sorted image names +
params + per-class support) hashes to ``probe_id``; a rebuild is a new
``probe_id`` = a new fingerprint = a new comparison space, never silently
compared across. If a manifest already exists for the same source +
params, it is reused as-is.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from kwcoco_detector_kit.eval.protocols import canonical_json, short_hash

__all__ = ["ProbeResult", "build_probe"]


@dataclass
class ProbeResult:
    probe_kwcoco_fpath: Path
    manifest_fpath: Path
    probe_id: str
    manifest: Dict[str, Any]


def _frame_stats(dset) -> Dict[int, Dict[str, int]]:
    """gid -> {category_name: instance count} (bbox annotations only)."""
    cat_name = {c["id"]: c["name"] for c in dset.dataset.get("categories", [])}
    counts: Dict[int, Dict[str, int]] = {gid: {} for gid in dset.images()}
    for ann in dset.dataset.get("annotations", []):
        name = cat_name.get(ann.get("category_id"))
        if name is None:
            continue
        gid = ann["image_id"]
        counts.setdefault(gid, {})
        counts[gid][name] = counts[gid].get(name, 0) + 1
    return counts


def build_probe(
    source_kwcoco: Union[str, Path],
    out_dpath: Union[str, Path],
    *,
    frames: int = 50,
    seed: int = 0,
    empty_frac: float = 0.1,
    min_frames_per_class: int = 3,
    source_id: Optional[str] = None,
    log=print,
) -> ProbeResult:
    import kwcoco

    out_dpath = Path(out_dpath)
    out_dpath.mkdir(parents=True, exist_ok=True)
    manifest_fpath = out_dpath / "probe_manifest.json"
    probe_fpath = out_dpath / "probe.kwcoco.zip"

    params = {
        "frames": int(frames),
        "seed": int(seed),
        "empty_frac": float(empty_frac),
        "min_frames_per_class": int(min_frames_per_class),
        "source_id": source_id,
    }

    # frozen-per-scheme: same source + params -> reuse without rebuilding
    if manifest_fpath.exists() and probe_fpath.exists():
        manifest = json.loads(manifest_fpath.read_text())
        if manifest.get("params") == params:
            log(f"[probe] reusing frozen probe {manifest['probe_id']} "
                f"({len(manifest['image_names'])} frames)")
            return ProbeResult(
                probe_kwcoco_fpath=probe_fpath,
                manifest_fpath=manifest_fpath,
                probe_id=manifest["probe_id"],
                manifest=manifest,
            )

    dset = kwcoco.CocoDataset.coerce(str(source_kwcoco))
    counts = _frame_stats(dset)
    corpus: Dict[str, int] = {}
    for per in counts.values():
        for name, n in per.items():
            corpus[name] = corpus.get(name, 0) + n

    weights = {c: 1.0 / n for c, n in corpus.items() if n > 0}
    frame_score = {
        gid: sum(n * weights.get(c, 0.0) for c, n in per.items())
        for gid, per in counts.items()
    }
    positives = [g for g, s in frame_score.items() if s > 0]
    empties = sorted(g for g, s in frame_score.items() if s <= 0)

    n_total = min(int(frames), len(counts))
    n_empty = min(int(round(n_total * float(empty_frac))), len(empties))
    n_pos = min(n_total - n_empty, len(positives))

    rng = random.Random(int(seed))
    chosen: List[int] = []

    # greedy coverage for rare classes first
    for cls in sorted(corpus, key=lambda c: corpus[c]):
        have = sum(1 for g in chosen if counts[g].get(cls, 0) > 0)
        if have >= min_frames_per_class:
            continue
        candidates = sorted(
            (g for g in positives if g not in chosen and counts[g].get(cls, 0) > 0),
            key=lambda g: (-counts[g].get(cls, 0), g),
        )
        for g in candidates[: max(0, min_frames_per_class - have)]:
            if len(chosen) >= n_pos:
                break
            chosen.append(g)
        if len(chosen) >= n_pos:
            break

    # weighted sampling without replacement for the rest
    pool = [g for g in sorted(positives) if g not in chosen]
    while len(chosen) < n_pos and pool:
        total = sum(frame_score[g] for g in pool)
        r = rng.random() * total
        acc = 0.0
        pick = pool[-1]
        for g in pool:
            acc += frame_score[g]
            if acc >= r:
                pick = g
                break
        chosen.append(pick)
        pool.remove(pick)

    chosen.extend(rng.sample(empties, n_empty))

    sub = dset.subset(sorted(chosen))
    # rewrite to absolute paths so the probe bundle is usable anywhere the
    # source imagery is mounted (see [[feedback-kwcoco-bakes-absolute-paths]])
    for img in sub.dataset["images"]:
        try:
            img["file_name"] = str(dset.get_image_fpath(img["id"]))
        except Exception:
            pass
    sub.fpath = str(probe_fpath)
    sub.dump()

    image_names = sorted(
        str(img.get("name") or img.get("file_name")) for img in sub.dataset["images"]
    )
    class_support = {
        c: sum(counts[g].get(c, 0) for g in chosen) for c in sorted(corpus)
    }
    manifest = {
        "params": params,
        "image_names": image_names,
        "class_support": class_support,
        "n_images": len(image_names),
    }
    manifest["probe_id"] = short_hash(canonical_json(manifest))
    manifest_fpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    log(
        f"[probe] built probe {manifest['probe_id']}: {len(image_names)} frames "
        f"({n_pos} positive / {n_empty} empty); class support {class_support}"
    )
    return ProbeResult(
        probe_kwcoco_fpath=probe_fpath,
        manifest_fpath=manifest_fpath,
        probe_id=manifest["probe_id"],
        manifest=manifest,
    )
