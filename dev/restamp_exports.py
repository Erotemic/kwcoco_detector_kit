#!/usr/bin/env python3
"""
Restamp correct category_names + provenance into EVERY exported model's
modelspec on this machine, so stale placeholder ('widget') exports can never
clobber good ones over rsync again.

Why this (and not re-export)
----------------------------
The stale exports have the right ONNX *graph* (a trained model outputs N class
logits — the graph does not encode class NAMES); only the .modelspec.json
sidecar carried the placeholder ('widget') category_names. So the fix is a
metadata rewrite, NOT a re-trace: fast, torch-free, and deterministic. We never
touch the (large) .onnx, so we don't create needless rsync churn.

Idempotent + convergent
-----------------------
For each model we resolve the authoritative category_names (policy.json when
present, else the scheme's target_order from class_schemes.yaml, flagged
imputed). If the modelspec already matches, we skip it untouched. So:
  * re-running does nothing (no churn),
  * once every machine has been restamped, rsync in any direction only ever
    propagates correct metadata — there is no stale copy left to clobber with.

Run this on EVERY machine that holds a copy of the runs tree (namek, aiq-gpu,
arisia, ...). It only rewrites the small JSON/labels files.

Usage
-----
    python dev/restamp_exports.py --dry-run     # preview
    python dev/restamp_exports.py               # apply
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import kwconf

_DEV = Path(__file__).resolve().parent


def _load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, _DEV / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the exporter's scheme detection + category-name resolution verbatim.
_ebs = _load_sibling("export_best_sealion_models")


class RestampConfig(kwconf.Config):
    kcd_root = kwconf.Value(
        "/data/users/jon.crall/kcd_sealion",
        help="training root containing runs/ (env KCD_TRAINING_ROOT overrides)",
    )
    schemes_yaml = kwconf.Value(None, help="class_schemes.yaml (default: project copy)")
    dry_run = kwconf.Value(False, isflag=True, help="print the plan; write nothing")


def _deterministic_provenance() -> dict:
    """kit + DEIMv2 SHAs only — identical across machines on the same commit, so
    restamped modelspecs are byte-stable and don't churn over rsync."""
    try:
        from kwcoco_detector_kit._provenance import provenance_dict
        p = provenance_dict()
    except Exception:
        return {}
    out = {}
    for src_k, dst_k in (("kit_sha", "kit_sha"), ("deimv2_sha", "deimv2_sha")):
        v = p.get(src_k)
        if v and v != "<unknown>":
            out[dst_k] = v
    out["restamped_by"] = "dev/restamp_exports.py"
    return out


def _apply_names(spec: dict, names: list, source: str, imputed: bool, prov: dict) -> None:
    """Mutate a loaded modelspec dict in place with correct names + provenance,
    preserving everything else (input/preprocess/postprocess/generated_at)."""
    meta = spec.setdefault("meta", {})
    meta["category_names"] = list(names)
    meta["category_names_source"] = source
    meta["has_imputed_metadata"] = bool(imputed)
    if prov:
        spec["provenance"] = {**spec.get("provenance", {}), **prov}
    if imputed:
        spec["imputed"] = {"category_names": f"imputed via {source}"}
    elif "imputed" in spec:
        del spec["imputed"]


def main(argv=None) -> int:
    config = RestampConfig.cli(argv=argv)
    kcd_root = Path(os.environ.get("KCD_TRAINING_ROOT", config.kcd_root)).expanduser()
    runs_root = kcd_root / "runs"
    if not runs_root.is_dir():
        print(f"ERROR: {runs_root} not found (set --kcd_root or KCD_TRAINING_ROOT)", file=sys.stderr)
        return 1
    schemes_yaml = Path(config.schemes_yaml) if config.schemes_yaml else (
        _ebs._kit_root() / "projects" / "viame_sealions_2026" / "docs" / "class_schemes.yaml")
    target_orders = _ebs._load_target_orders(schemes_yaml)
    prov = _deterministic_provenance()

    print(f"# kcd_root : {kcd_root}")
    print(f"# mode     : {'DRY-RUN' if config.dry_run else 'APPLY'}")
    print(f"# kit_sha  : {prov.get('kit_sha', '?')}")
    print()

    fixed = ok = unresolved = 0
    for workdir in sorted(runs_root.glob("*/runs/*")):
        if not workdir.is_dir():
            continue
        specs = sorted(workdir.glob("**/*.modelspec.json"))
        if not specs:
            continue
        run_name = workdir.parents[1].name  # <kcd_root>/runs/<RUN>/runs/<CAND>
        scheme = _ebs._scheme_of(run_name)
        if scheme is None:
            unresolved += len(specs)
            print(f"  ? {run_name}/{workdir.name} — unknown scheme; left untouched")
            continue
        names, source, imputed = _ebs._resolve_category_names(workdir, scheme, target_orders)
        if not names:
            unresolved += len(specs)
            print(f"  ? {run_name}/{workdir.name} — cannot resolve category_names; left untouched")
            continue

        for ms in specs:
            try:
                spec = json.loads(ms.read_text())
            except Exception as ex:
                print(f"  ! {ms} — unreadable ({ex}); skipped")
                continue
            existing = spec.get("meta", {}).get("category_names")
            if existing == list(names):
                ok += 1
                continue
            rel = ms.relative_to(kcd_root)
            print(f"  FIX [{scheme}] {existing} -> {names}")
            print(f"        {rel}")
            fixed += 1
            if config.dry_run:
                continue
            _apply_names(spec, names, source, imputed, prov)
            ms.write_text(json.dumps(spec, indent=2))
            # labels.txt next to this modelspec (same stem as the sibling .onnx)
            stem = ms.name[: -len(".modelspec.json")]
            (ms.parent / f"{stem}.labels.txt").write_text("\n".join(names) + "\n")
            # a package/labels.json alongside a package/exports modelspec
            lj = ms.parent.parent / "labels.json"
            if ms.parent.name == "exports" and lj.is_file():
                lj.write_text(json.dumps({"labels": list(names)}, indent=2))

    print()
    verb = "would fix" if config.dry_run else "fixed"
    print(f"# {verb}={fixed}  already-correct={ok}  unresolved={unresolved}")
    if config.dry_run and fixed:
        print("# re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
