#!/usr/bin/env python3
"""
Select the best trained checkpoint per sea-lion scheme and (re-)export it to a
provenance-complete ONNX package.

Why this exists
---------------
The on-disk exports under ``$KCD_TRAINING_ROOT/runs/*/runs/*/export`` were
produced by an older path that fell back to a placeholder ``category_names``
("widget") and never ran ``package-build``. This script:

  1. Aggregates every run's ``manifest.tsv`` and picks, per scheme, the
     checkpoint with the highest ``test_ap`` (class-agnostic detection AP is
     the selection criterion; desktop-latency eligibility is intentionally
     ignored — slow sea-lion detectors are fine).
  2. Resolves ``category_names`` from the cleanest available source and records
     where it came from. When the names cannot be read from a clean data-driven
     source (policy.json), they are imputed from the scheme's ``target_order``
     in ``class_schemes.yaml`` and flagged ``imputed`` so downstream systems
     distrust them.
  3. Runs ``export-onnx`` (writes <name>.onnx + .modelspec.json + .labels.txt
     with full provenance) followed by ``package-build`` (bundles checkpoint,
     config, datasets, onnx, modelspec into a provenance manifest).

It is DRY-RUN by default: it prints the selection plan (which works anywhere,
reading only the tiny manifest.tsv + yaml). Pass ``--run`` to actually export
— that step needs the DEIMv2 torch environment and is meant to run inside the
training docker image on the GPU box.

Usage
-----
    # plan only (safe anywhere)
    python dev/export_best_sealion_models.py

    # actually export+package (on the GPU box, in the DEIMv2 image)
    python dev/export_best_sealion_models.py --run

    # restrict to one scheme, force re-export
    python dev/export_best_sealion_models.py --schemes pup_vs_nonpup --run --force
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import kwconf

KNOWN_SCHEMES = ["single_sealion", "pup_vs_nonpup", "lifestage_6cls", "full_8cls"]


class ExportBestConfig(kwconf.Config):
    kcd_root = kwconf.Value(
        "/data/users/jon.crall/kcd_sealion",
        help="training root containing runs/ (env KCD_TRAINING_ROOT overrides)",
    )
    schemes_yaml = kwconf.Value(
        None,
        help="class_schemes.yaml; defaults to projects/viame_sealions_2026/docs/class_schemes.yaml",
    )
    schemes = kwconf.Value("all", help="comma-separated subset of schemes, or 'all'")
    run = kwconf.Value(False, isflag=True, help="actually export+package (default: dry-run plan only)")
    force = kwconf.Value(False, isflag=True, help="re-export even if .onnx already exists")
    min_ap = kwconf.Value(0.0, parser=float, help="warn (and skip on --run) selections below this test_ap")
    score_thresh = kwconf.Value(0.30, parser=float, help="score threshold written into the export")
    opset = kwconf.Value(18, parser=int, help="ONNX opset")
    archive = kwconf.Value(False, isflag=True, help="emit a package .zip instead of a package/ directory")


def _kit_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_target_orders(schemes_yaml: Path) -> dict:
    import yaml
    data = yaml.safe_load(schemes_yaml.read_text()) or {}
    schemes = data.get("schemes", data)
    out = {}
    for name, spec in schemes.items():
        if isinstance(spec, dict) and spec.get("target_order"):
            out[name] = [str(c) for c in spec["target_order"]]
    return out


def _scheme_of(run_name: str) -> str | None:
    for s in sorted(KNOWN_SCHEMES, key=len, reverse=True):
        if run_name.startswith(s):
            return s
    return None


def _aggregate_manifests(runs_root: Path):
    """Return (rows_by_scheme, runs_without_manifest)."""
    rows_by_scheme: dict[str, list[dict]] = {s: [] for s in KNOWN_SCHEMES}
    no_manifest: list[str] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        scheme = _scheme_of(run_dir.name)
        if scheme is None:
            continue
        tsv = run_dir / "manifest.tsv"
        if not tsv.is_file():
            no_manifest.append(run_dir.name)
            continue
        with open(tsv) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                try:
                    ap = float(r.get("test_ap", ""))
                except (TypeError, ValueError):
                    ap = None
                cand = r.get("candidate_id", "")
                # workdir is mount-independent: <run>/runs/<candidate_id>
                workdir = run_dir / "runs" / cand
                rows_by_scheme[scheme].append({
                    "test_ap": ap,
                    "run": run_dir.name,
                    "run_dir": run_dir,
                    "candidate_id": cand,
                    "variant": r.get("variant", ""),
                    "workdir": workdir,
                    "export_h": r.get("export_input_h", ""),
                    "export_w": r.get("export_input_w", ""),
                })
    return rows_by_scheme, no_manifest


def _resolve_category_names(workdir: Path, scheme: str, target_orders: dict):
    """Return (names, source, imputed)."""
    policy = {}
    pj = workdir / "policy.json"
    if pj.is_file():
        try:
            policy = json.loads(pj.read_text())
        except Exception:
            policy = {}
    names = policy.get("category_names") or policy.get("label_list")
    if names:
        return list(names), "policy.json", False
    to = target_orders.get(scheme)
    if to:
        # Spec-derived via scheme name parsed from the run directory — NOT read
        # from the model's actual training data. Flag as imputed.
        return list(to), f"imputed:class_schemes.yaml:{scheme}", True
    return [], "unknown", True


def _kwcoco_for(run_dir: Path, name: str) -> Path | None:
    p = run_dir / "scheme_applied" / name
    return p if p.is_file() else None


def _cli(*args: str) -> list[str]:
    return [sys.executable, "-m", "kwcoco_detector_kit.cli", *args]


def main(argv=None):
    import os
    config = ExportBestConfig.cli(argv=argv)
    kcd_root = Path(os.environ.get("KCD_TRAINING_ROOT", config.kcd_root)).expanduser()
    runs_root = kcd_root / "runs"
    if not runs_root.is_dir():
        print(f"ERROR: {runs_root} not found (set --kcd_root or KCD_TRAINING_ROOT)", file=sys.stderr)
        return 1

    schemes_yaml = Path(config.schemes_yaml) if config.schemes_yaml else (
        _kit_root() / "projects" / "viame_sealions_2026" / "docs" / "class_schemes.yaml")
    target_orders = _load_target_orders(schemes_yaml)

    requested = KNOWN_SCHEMES if config.schemes == "all" else [
        s.strip() for s in str(config.schemes).split(",") if s.strip()]

    rows_by_scheme, no_manifest = _aggregate_manifests(runs_root)

    print(f"# kcd_root      : {kcd_root}")
    print(f"# class_schemes : {schemes_yaml}")
    print(f"# mode          : {'RUN (export+package)' if config.run else 'DRY-RUN (plan only)'}")
    print()

    plan = []
    for scheme in requested:
        rows = [r for r in rows_by_scheme.get(scheme, []) if r["test_ap"] is not None]
        if not rows:
            print(f"[{scheme}] NO manifest rows with test_ap — cannot select. SKIP.")
            continue
        best = max(rows, key=lambda r: r["test_ap"])
        names, src, imputed = _resolve_category_names(best["workdir"], scheme, target_orders)
        ok = best["workdir"].is_dir() and (best["workdir"] / "policy.json").is_file()
        flag = "" if best["test_ap"] >= config.min_ap else "  ** BELOW min_ap **"
        warn = "" if ok else "  ** workdir/policy.json MISSING **"
        print(f"[{scheme}] best test_ap={best['test_ap']:.4f}{flag}{warn}")
        print(f"    run        : {best['run']}")
        print(f"    candidate  : {best['candidate_id']}  ({best['export_h']}x{best['export_w']})")
        print(f"    workdir    : {best['workdir']}")
        print(f"    categories : {names}")
        print(f"    names_src  : {src}{'  [IMPUTED]' if imputed else ''}")
        plan.append((scheme, best, names, src, imputed, ok))

    if no_manifest:
        print()
        print("# Runs WITHOUT manifest.tsv (not ranked — a better un-manifested model may exist):")
        for n in no_manifest:
            print(f"#   {n}")

    if not config.run:
        print()
        print("# DRY-RUN: re-run with --run on the GPU box (DEIMv2 image) to export + package.")
        return 0

    # --- execute ---------------------------------------------------------
    print()
    failures = []
    for scheme, best, names, src, imputed, ok in plan:
        if not ok:
            print(f"[{scheme}] SKIP — workdir/policy.json missing")
            failures.append(scheme)
            continue
        if best["test_ap"] < config.min_ap:
            print(f"[{scheme}] SKIP — test_ap {best['test_ap']:.4f} < min_ap {config.min_ap}")
            continue
        workdir = best["workdir"]
        variant = best["variant"] or "deimv2"

        export_cmd = _cli(
            "export-onnx", str(workdir),
            "--category-names", ",".join(names),
            "--category-names-source", src,
            "--score-thresh", str(config.score_thresh),
            "--opset", str(config.opset),
        )
        if imputed:
            export_cmd.append("--category-names-imputed")
        if config.force:
            export_cmd.append("--force")

        out_pkg = workdir / ("package.zip" if config.archive else "package")
        pkg_cmd = _cli(
            "package-build", "--workdir", str(workdir),
            "--trainer", "deimv2",
            "--variant", variant,
            "--category-names", ",".join(names),
            "--experiment-slug", best["candidate_id"],
            "--dataset-slug", scheme,
            "--out", str(out_pkg),
        )
        train = _kwcoco_for(best["run_dir"], "train.kwcoco.zip")
        vali = _kwcoco_for(best["run_dir"], "vali.kwcoco.zip")
        test = _kwcoco_for(best["run_dir"], "test.kwcoco.zip")
        if train:
            pkg_cmd += ["--train-kwcoco", str(train)]
        if vali:
            pkg_cmd += ["--vali-kwcoco", str(vali)]
        if test:
            pkg_cmd += ["--test-kwcoco", str(test)]

        print(f"\n=== [{scheme}] exporting {best['candidate_id']} (AP={best['test_ap']:.4f}) ===")
        for cmd in (export_cmd, pkg_cmd):
            print("  $ " + " ".join(cmd))
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                print(f"  !! command failed (exit {proc.returncode})")
                failures.append(scheme)
                break
        else:
            print(f"  -> package: {out_pkg}")

    print()
    if failures:
        print(f"# DONE with failures in: {sorted(set(failures))}")
        return 1
    print("# DONE — all selected schemes exported + packaged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
