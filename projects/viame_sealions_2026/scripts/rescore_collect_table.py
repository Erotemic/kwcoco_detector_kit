#!/usr/bin/env python3
"""
Collect WHOLE-IMAGE vs TILED rescore results across every run that has a
``tiled_compare/`` dir (produced by rescore_tiled_compare.py / rescore_all.sh)
and print one comparison table.

For each run it reports the class-agnostic detection AP (the selection
criterion, [[feedback-detection-ap-is-selection-criterion]]) whole-image vs
tiled. When a scheme carries a distractor (lifestage_6cls -> northern_fur_seal)
the NFS-EXCLUDED sidecar (``detect_metrics.<distractor>.json``) is the true
selection number; for schemes that drop NFS entirely (pup_vs_nonpup,
single_sealion) the plain metrics already exclude it.

Usage (host, no container needed — pure json reads)::

    python3 projects/viame_sealions_2026/scripts/rescore_collect_table.py \\
        --runs_dpath /data/users/jon.crall/kcd_sealion/runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _nocls_ap(metrics_fpath: Path):
    try:
        d = json.loads(metrics_fpath.read_text())
    except Exception:
        return None
    for _k, v in (d.items() if isinstance(d, dict) else []):
        if not isinstance(v, dict):
            continue
        if "nocls_ap" in v:
            return float(v["nocls_ap"])
        nm = v.get("nocls_measures")
        if isinstance(nm, dict) and nm.get("ap") is not None:
            return float(nm["ap"])
    return None


def _perclass_ap(metrics_fpath: Path):
    try:
        d = json.loads(metrics_fpath.read_text())
    except Exception:
        return {}
    out = {}
    for _k, v in (d.items() if isinstance(d, dict) else []):
        if not isinstance(v, dict):
            continue
        ovr = v.get("ovr_measures") or v.get("perclass_measures") or {}
        if isinstance(ovr, dict):
            for cls, m in ovr.items():
                if isinstance(m, dict) and m.get("ap") is not None:
                    out[cls] = float(m["ap"])
        if out:
            break
    return out


# scheme -> distractor used to find the NFS-excluded sidecar
_SCHEME_DISTRACTOR = {
    "lifestage_6cls": "northern_fur_seal",
}


def _scheme_of(run_name: str) -> str:
    for s in ("pup_vs_nonpup", "single_sealion", "lifestage_6cls"):
        if run_name.startswith(s):
            return s
    return "?"


def _selection_ap(eval_dir: Path, scheme: str):
    """class-agnostic AP, NFS-excluded (the selection criterion)."""
    main = eval_dir / "detect_metrics.json"
    distractor = _SCHEME_DISTRACTOR.get(scheme)
    if distractor:
        sidecar = eval_dir / f"detect_metrics.{distractor}.json"
        if sidecar.exists():
            return _nocls_ap(sidecar)
    return _nocls_ap(main)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_dpath", type=Path, required=True)
    p.add_argument("--tsv", type=Path, default=None,
                   help="also write the table as TSV here")
    args = p.parse_args()

    rows = []
    for tc in sorted(args.runs_dpath.glob("*/tiled_compare")):
        run_name = tc.parent.name
        scheme = _scheme_of(run_name)
        w_dir = tc / "eval" / "wholeimage" / "eval"
        t_dir = tc / "eval" / "tiled" / "eval"
        w_ap = _selection_ap(w_dir, scheme) if w_dir.exists() else None
        t_ap = _selection_ap(t_dir, scheme) if t_dir.exists() else None
        # backbone token from the run name (hgnetv2_n / dinov3_s / dinov3_x)
        backbone = "?"
        for b in ("hgnetv2_n", "dinov3_s", "dinov3_m", "dinov3_l", "dinov3_x"):
            if b in run_name:
                backbone = b
                break
        pc = _perclass_ap(t_dir / "detect_metrics.json") if t_dir.exists() else {}
        rows.append(dict(run=run_name, scheme=scheme, backbone=backbone,
                         whole=w_ap, tiled=t_ap, perclass=pc))

    # ---- print grouped by scheme -----------------------------------------
    def f(x):
        return f"{x:.4f}" if isinstance(x, float) else "  -   "

    hdr = f"{'run':<62} {'backbone':<10} {'whole':>7} {'tiled':>7} {'x':>5}"
    for scheme in ("pup_vs_nonpup", "single_sealion", "lifestage_6cls", "?"):
        srows = [r for r in rows if r["scheme"] == scheme]
        if not srows:
            continue
        print(f"\n### {scheme}  (class-agnostic detection AP, NFS-excluded)")
        print(hdr)
        print("-" * len(hdr))
        for r in sorted(srows, key=lambda r: (r["backbone"], r["run"])):
            mult = ""
            if isinstance(r["whole"], float) and isinstance(r["tiled"], float) and r["whole"] > 0:
                mult = f"{r['tiled'] / r['whole']:.1f}x"
            print(f"{r['run']:<62} {r['backbone']:<10} {f(r['whole']):>7} {f(r['tiled']):>7} {mult:>5}")
            if r["perclass"]:
                pcs = "  ".join(f"{c}={v:.3f}" for c, v in sorted(r["perclass"].items()))
                print(f"    tiled per-class: {pcs}")

    if args.tsv:
        lines = ["run\tscheme\tbackbone\twhole_ap\ttiled_ap"]
        for r in rows:
            lines.append(
                f"{r['run']}\t{r['scheme']}\t{r['backbone']}\t"
                f"{r['whole'] if r['whole'] is not None else ''}\t"
                f"{r['tiled'] if r['tiled'] is not None else ''}")
        args.tsv.write_text("\n".join(lines) + "\n")
        print(f"\nTSV -> {args.tsv}")

    done = sum(1 for r in rows if isinstance(r["tiled"], float))
    print(f"\n{done}/{len(rows)} runs have a tiled score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
