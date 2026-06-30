#!/usr/bin/env python3
"""
Collect the best sea-lion detector per scheme into a single zip to hand to an
external reviewer (e.g. Matt) for testing in VIAME.

What it ships, per selected model:
  * the ONNX graph + its provenance-complete .modelspec.json + .labels.txt
  * the small provenance files (package.yaml, training_config/policy.json)
  * a ready-to-run VIAME pipe (detector.pipe, with a [-PACKAGE-] placeholder)
  * MODEL.md — AP, category names (+imputed flag), provenance SHAs, caveats
Plus, at the bundle root: README.md, manifest.json, and run_detector.sh.

It deliberately does NOT ship the 820 MB training checkpoint.pth (not needed for
ONNX inference) and skips any model whose modelspec still has placeholder
("widget") / missing category_names — those are undecodable and must be
re-exported first with dev/export_best_sealion_models.py --run --force.

Selection mirrors export_best_sealion_models.py: highest test_ap per scheme.

Usage
-----
    # preview what would be bundled (no zip written)
    python dev/bundle_models_for_review.py --dry-run

    # write the zip (default: under the data drive, not the kit checkout)
    python dev/bundle_models_for_review.py \
        --out /data/users/jon.crall/kcd_sealion/review_bundles/sealion_detectors.zip
"""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import kwconf

_DEV = Path(__file__).resolve().parent


def _load_sibling(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _DEV / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the exporter's selection logic verbatim so the bundle picks the same
# best-per-scheme checkpoint the exporter does.
_ebs = _load_sibling("export_best_sealion_models")


class BundleConfig(kwconf.Config):
    kcd_root = kwconf.Value(
        "/data/users/jon.crall/kcd_sealion",
        help="training root containing runs/ (env KCD_TRAINING_ROOT overrides)",
    )
    schemes = kwconf.Value("all", help="comma-separated subset of schemes, or 'all'")
    schemes_yaml = kwconf.Value(None, help="class_schemes.yaml (default: project copy)")
    out = kwconf.Value(
        None,
        help="output .zip path (default: <kcd_root>/review_bundles/sealion_detectors_for_review.zip)",
    )
    viame_root = kwconf.Value(
        "~/code/VIAME",
        help="VIAME checkout to read the canonical .pipe template from (falls back to a builtin)",
    )
    min_ap = kwconf.Value(0.0, parser=float, help="skip selections below this test_ap")
    dry_run = kwconf.Value(False, isflag=True, help="print the plan; do not write the zip")


# Fallback pipe template, used when the VIAME checkout isn't available. Kept in
# sync with VIAME configs/pipelines/templates/detector_kwcoco_detector_kit.pipe.
_FALLBACK_PIPE = """\
# kwcoco_detector_kit ONNX detector — replace [-PACKAGE-] with the package dir.
config _pipeline:_edge
  :capacity                                    5
config _scheduler
  :type                                        pythread_per_process

process input
  :: video_input
  :video_filename                              input_list.txt
  :frame_time                                  1
  :exit_on_invalid                             false
  :video_reader:type                           image_list
  block video_reader:image_list
    :image_reader:type                         vxl
    :skip_bad_images                           true
    block image_reader:vxl
      :force_byte                              true
    endblock
  endblock

process detector1
  :: image_object_detector
  :detector:type                               kwcoco_detector_kit
  block detector:kwcoco_detector_kit
    :package                                   [-PACKAGE-]
    :device                                    cpu
    :score_thresh                              0.30
    :nms_thresh                                0.50
  endblock
connect from input.image
        to   detector1.image

process detector_writer
  :: detected_object_output
  :file_name                                   computed_detections.csv
  :writer:type                                 viame_csv
connect from detector1.detected_object_set
        to   detector_writer.detected_object_set
connect from input.file_name
        to   detector_writer.image_file_name
"""

_RUN_HELPER = """\
#!/usr/bin/env bash
# Run one bundled detector over an image list with VIAME.
#
# Usage (from anywhere; paths resolve relative to this script):
#   ./run_detector.sh <model_subdir> <image_list.txt> [output.csv] [device]
# e.g.
#   ./run_detector.sh pup_vs_nonpup my_images.txt dets.csv cuda
#
# Requires a VIAME build with the kwcoco_detector_kit detector (branch
# kwcoco-detector-kit-main) and `viame` on PATH (source setup_viame.sh).
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:?model subdir, e.g. pup_vs_nonpup}"
LIST="${2:?image list file (one image path per line)}"
OUT="${3:-computed_detections.csv}"
DEVICE="${4:-cpu}"
PKG="$HERE/models/$MODEL"
[ -d "$PKG" ] || { echo "no such model: $PKG" >&2; exit 1; }
TMP="$(mktemp --suffix=.pipe)"
sed "s|\\[-PACKAGE-\\]|$PKG|g" "$PKG/detector.pipe" > "$TMP"
viame "$TMP" \\
    -s input:video_filename="$LIST" \\
    -s detector1:detector:kwcoco_detector_kit:device="$DEVICE" \\
    -s detector_writer:file_name="$OUT"
echo "wrote $OUT"
"""


def _read_pipe_template(viame_root: Path) -> str:
    tmpl = (viame_root / "configs" / "pipelines" / "templates"
            / "detector_kwcoco_detector_kit.pipe")
    if tmpl.is_file():
        return tmpl.read_text()
    return _FALLBACK_PIPE


def _modelspec_names(modelspec: Path) -> tuple[list[str], dict]:
    spec = json.loads(modelspec.read_text())
    meta = spec.get("meta", {})
    return list(meta.get("category_names", [])), spec


def _is_shippable(names: list[str]) -> bool:
    return bool(names) and names != ["widget"]


def _collect(config: BundleConfig):
    """Return (selected_models, skipped) where each selected model is a dict of
    everything needed to write its bundle entry."""
    import os
    kcd_root = Path(os.environ.get("KCD_TRAINING_ROOT", config.kcd_root)).expanduser()
    runs_root = kcd_root / "runs"
    schemes_yaml = Path(config.schemes_yaml) if config.schemes_yaml else (
        _ebs._kit_root() / "projects" / "viame_sealions_2026" / "docs" / "class_schemes.yaml")
    target_orders = _ebs._load_target_orders(schemes_yaml)

    requested = (_ebs.KNOWN_SCHEMES if config.schemes == "all"
                 else [s.strip() for s in str(config.schemes).split(",") if s.strip()])
    rows_by_scheme, _ = _ebs._aggregate_manifests(runs_root)

    selected, skipped = [], []
    for scheme in requested:
        rows = [r for r in rows_by_scheme.get(scheme, []) if r["test_ap"] is not None]
        if not rows:
            skipped.append((scheme, "no manifest rows with test_ap"))
            continue
        best = max(rows, key=lambda r: r["test_ap"])
        if best["test_ap"] < config.min_ap:
            skipped.append((scheme, f"best test_ap {best['test_ap']:.4f} < min_ap {config.min_ap}"))
            continue
        workdir = best["workdir"]
        export_dir = workdir / "export"
        onnxes = sorted(export_dir.glob("*.onnx")) if export_dir.is_dir() else []
        if not onnxes:
            skipped.append((scheme, f"no exported .onnx under {export_dir} "
                                    "(run export_best_sealion_models.py --run --force)"))
            continue
        onnx = onnxes[0]
        modelspec = onnx.with_suffix(".modelspec.json")
        if not modelspec.is_file():
            skipped.append((scheme, f"missing modelspec {modelspec.name}"))
            continue
        names, spec = _modelspec_names(modelspec)
        if not _is_shippable(names):
            skipped.append((scheme, f"stale/placeholder category_names={names!r} "
                                    "— re-export with --force before bundling"))
            continue
        _, src, imputed = _ebs._resolve_category_names(workdir, scheme, target_orders)
        selected.append({
            "scheme": scheme,
            "candidate_id": best["candidate_id"],
            "run": best["run"],
            "test_ap": best["test_ap"],
            "category_names": names,
            "names_source": spec.get("meta", {}).get("category_names_source", src),
            "imputed": bool(spec.get("meta", {}).get("has_imputed_metadata", imputed)),
            "provenance": spec.get("provenance", {}),
            "onnx": onnx,
            "modelspec": modelspec,
            "labels": onnx.with_suffix(".labels.txt"),
            "bench": onnx.with_suffix(".bench.json"),
            "package_yaml": workdir / "package" / "package.yaml",
            "policy_json": workdir / "package" / "training_config" / "policy.json",
        })
    return selected, skipped


def _model_md(m: dict) -> str:
    prov = m["provenance"]
    return f"""# {m['scheme']} — {m['candidate_id']}

| field | value |
| ----- | ----- |
| scheme | `{m['scheme']}` |
| run | `{m['run']}` |
| candidate | `{m['candidate_id']}` |
| test_ap (class-agnostic, NFS-excluded) | **{m['test_ap']:.4f}** |
| category_names | `{m['category_names']}` |
| category_names_source | `{m['names_source']}` |
| labels imputed? | **{'YES — treat class names with caution' if m['imputed'] else 'no'}** |
| kit_git_sha | `{prov.get('kit_git_sha', '?')}` |
| deimv2_git_sha | `{prov.get('deimv2_sha', '?')}` |
| source_checkpoint | `{prov.get('source_checkpoint', {}).get('name', '?')}` |

Caveats
-------
* Detections are tiled-resolution. AP above was measured on tiles; whole-frame
  inference downscales sea-lions below the small-object floor. Run on crops /
  tiles sized near the model's eval resolution for representative results.
* `category_names` order is the model's class-index order; index 0 = first name.
* This is a self-describing package — `.modelspec.json` carries input size,
  preprocessing, score threshold, and class names; the VIAME side needs no
  extra config beyond the package path.
"""


def _readme(selected: list[dict], skipped: list) -> str:
    rows = "\n".join(
        f"| `{m['scheme']}` | {m['test_ap']:.4f} | `{m['category_names']}` | "
        f"{'imputed' if m['imputed'] else 'clean'} | `{m['candidate_id']}` |"
        for m in selected)
    skip_lines = "\n".join(f"* `{s}` — {why}" for s, why in skipped) or "* (none)"
    return f"""# Sea-lion detectors for review

{len(selected)} ONNX detector(s) exported from `kwcoco_detector_kit`, one per
classification scheme (highest test AP per scheme).

| scheme | test_ap | category_names | labels | candidate |
| ------ | ------- | -------------- | ------ | --------- |
{rows}

Each `models/<scheme>/` contains a self-describing ONNX package
(`.onnx` + `.modelspec.json` + `.labels.txt`), a `detector.pipe`, a `MODEL.md`
with provenance, and a `provenance/` folder (package.yaml + policy.json).

## Running in VIAME

These need a VIAME build that includes the `kwcoco_detector_kit` detector
(branch **kwcoco-detector-kit-main**). Inference is ONNX-only — no PyTorch.

```bash
# from the extracted bundle root, after sourcing setup_viame.sh:
./run_detector.sh <scheme> <image_list.txt> [output.csv] [cpu|cuda]
# e.g.
./run_detector.sh pup_vs_nonpup my_images.txt dets.csv cuda
```

`run_detector.sh` fills the package path into `models/<scheme>/detector.pipe`
(the `[-PACKAGE-]` placeholder) and runs `viame`. Or wire `detector.pipe` into
your own VIAME workflow and set `:package` to the model directory yourself.

## Important caveats

* **Tiled resolution.** AP was measured on tiles; whole-frame inference shrinks
  sea-lions below the detector's small-object floor. Test on tiles/crops near
  the model's eval resolution.
* **Imputed labels.** Models marked `imputed` had their `category_names`
  inferred from the scheme spec (not a clean data-driven source). The order is
  believed correct but is flagged in `MODEL.md` / `.modelspec.json`
  (`meta.has_imputed_metadata`). Treat class *names* with appropriate caution.
* **AP is class-agnostic detection AP with NFS excluded** — the project's
  selection criterion; per-class AP is diagnostic only.

## Skipped schemes
{skip_lines}
"""


def main(argv=None) -> int:
    config = BundleConfig.cli(argv=argv)
    selected, skipped = _collect(config)

    print("# Bundle plan")
    for m in selected:
        flag = "  [IMPUTED labels]" if m["imputed"] else ""
        print(f"  + {m['scheme']:16s} ap={m['test_ap']:.4f}  {m['candidate_id']}{flag}")
        print(f'      categories: {m["category_names"]}')
        print(f"      onnx: {m['onnx']}  ({m['onnx'].stat().st_size // (1024*1024)} MB)")
    for s, why in skipped:
        print(f"  - {s:16s} SKIP — {why}")

    if not selected:
        print("\nNo shippable models. Re-export with "
              "dev/export_best_sealion_models.py --run --force, then retry.", file=sys.stderr)
        return 1

    out = Path(config.out).expanduser() if config.out else (
        Path(config.kcd_root).expanduser() / "review_bundles"
        / "sealion_detectors_for_review.zip")

    if config.dry_run:
        total_mb = sum(m["onnx"].stat().st_size for m in selected) // (1024 * 1024)
        print(f"\n[dry-run] would write {len(selected)} model(s) (~{total_mb} MB of ONNX) -> {out}")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    viame_root = Path(config.viame_root).expanduser()
    pipe_template = _read_pipe_template(viame_root)
    stem = out.stem  # bundle root dir name inside the zip

    manifest = {"models": [], "bundle": stem}
    print(f"\nWriting {out} ...")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for m in selected:
            base = f"{stem}/models/{m['scheme']}"
            # ONNX is already near-incompressible float weights → store, don't deflate.
            zf.write(m["onnx"], f"{base}/{m['onnx'].name}", compress_type=zipfile.ZIP_STORED)
            zf.write(m["modelspec"], f"{base}/{m['modelspec'].name}")
            for opt in ("labels", "bench"):
                if m[opt].is_file():
                    zf.write(m[opt], f"{base}/{m[opt].name}")
            for prov_key, arc in (("package_yaml", "provenance/package.yaml"),
                                  ("policy_json", "provenance/policy.json")):
                if m[prov_key].is_file():
                    zf.write(m[prov_key], f"{base}/{arc}")
            # Keep the [-PACKAGE-] placeholder so run_detector.sh can fill the
            # absolute path at run time (and manual users see what to replace).
            zf.writestr(f"{base}/detector.pipe", pipe_template)
            zf.writestr(f"{base}/MODEL.md", _model_md(m))
            manifest["models"].append({
                "scheme": m["scheme"], "candidate_id": m["candidate_id"],
                "test_ap": m["test_ap"], "category_names": m["category_names"],
                "category_names_source": m["names_source"], "imputed": m["imputed"],
                "onnx": f"models/{m['scheme']}/{m['onnx'].name}",
                "provenance": m["provenance"],
            })
            print(f"  + {m['scheme']}")
        zf.writestr(f"{stem}/README.md", _readme(selected, skipped))
        zf.writestr(f"{stem}/manifest.json", json.dumps(manifest, indent=2))
        info = zipfile.ZipInfo(f"{stem}/run_detector.sh")
        info.external_attr = 0o755 << 16  # make the helper executable
        zf.writestr(info, _RUN_HELPER)

    size_mb = out.stat().st_size // (1024 * 1024)
    print(f"\nDONE — {len(selected)} model(s), {size_mb} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
