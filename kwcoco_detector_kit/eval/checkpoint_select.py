"""
Checkpoint shortlist sweep — for trainers that emit one checkpoint per
epoch (DEIMv2, OpenGroundingDINO), evaluate each checkpoint against the
validation kwcoco and pick the highest-AP one.

Ported in shape from the v9 OpenGroundingDINO + SAM2 script's "Sweep
checkpoint shortlist on validation" block.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Candidate:
    candidate_id: str
    ckpt_fpath: Path
    vali_ap: Optional[float]


def shortlist_checkpoints(
    workdir: Path,
    *,
    pattern: str = "checkpoint*.pth",
) -> List[Path]:
    """Return per-epoch checkpoints under workdir, sorted by epoch index."""
    return sorted(Path(workdir).glob(pattern))


def select_best(rows: List[Candidate]) -> Optional[Candidate]:
    """Return the row with the highest vali_ap, ignoring rows with None."""
    have_ap = [r for r in rows if r.vali_ap is not None]
    if not have_ap:
        return None
    return max(have_ap, key=lambda r: float(r.vali_ap))


def write_summary_tsv(rows: List[Candidate], out_fpath: Path) -> Path:
    out_fpath = Path(out_fpath)
    out_fpath.parent.mkdir(parents=True, exist_ok=True)
    with out_fpath.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["candidate_id", "ckpt_fpath", "vali_ap"])
        for r in rows:
            w.writerow([
                r.candidate_id,
                str(r.ckpt_fpath),
                "" if r.vali_ap is None else f"{r.vali_ap:.6f}",
            ])
    return out_fpath
