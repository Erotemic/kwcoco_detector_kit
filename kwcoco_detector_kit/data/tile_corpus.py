"""
Build a multi-pass tile *corpus* from a single source kwcoco.

The ``tile`` command emits one tiling mode per call (``full_only`` /
``quadrant`` / ``multiscale``). Many projects want a TRAINING bundle that
combines several passes so the model sees objects across the full range of
apparent scales it will encounter at inference — e.g. a downsized whole-frame
view + overlapping full-res crops + fixed-size tiles cut from several source
scales. ``tile-corpus`` runs an ordered list of passes (each a full ``tile``
invocation) and unions the results into one bundle.

This is deliberately GENERIC: the pass list is supplied by the caller via a
spec file, so poop / sealion / any detection project compose their own corpus
with the same code rather than duplicating shell orchestration.

Spec file (YAML or JSON), e.g.::

    # shared keys apply to every pass unless the pass overrides them
    shared:
      category_names: poop
    passes:
      - {name: full,       mode: full_only, full_dim: 1280}
      - {name: quad2,      mode: quadrant,  tile_grid: 2, tile_overlap: 0.20,
                           tile_output_dim: 640, keep_full: false}
      - {name: multiscale, mode: multiscale, tile_size: 640,
                           source_scales: "1.0,0.66,0.40,0.25", stride_frac: 0.5,
                           keep_negative: true}

Each pass key is any ``TileConfig`` field. Per-pass tiled bundles are written
under ``<dst-stem>_passes/`` and then unioned (with absolute image paths so the
final bundle resolves its assets regardless of where it lives).

CLI::

    kwcoco-detector-kit tile-corpus <src> <dst> --spec corpus_spec.yaml
"""
from __future__ import annotations

from pathlib import Path

import kwconf


class TileCorpusConfig(kwconf.Config):
    """Compose several `tile` passes into one unioned training bundle."""

    src = kwconf.Value(None, position=1, required=True,
                     help="source kwcoco (raw, full-resolution images)")
    dst = kwconf.Value(None, position=2, required=True,
                     help="output unioned tiled kwcoco bundle")
    spec = kwconf.Value(None, required=True,
                      help="YAML/JSON file with optional `shared:` dict and a "
                           "`passes:` list; each pass is a dict of TileConfig fields")
    passes_dpath = kwconf.Value(None,
                              help="where per-pass tiled bundles are written "
                                   "(default <dst-stem>_passes/ next to dst)")
    force = kwconf.Value(False, isflag=True,
                       help="rebuild a pass even if its bundle already exists")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        return run(config)


def _load_spec(spec_fpath: Path) -> dict:
    import json
    text = Path(spec_fpath).expanduser().read_text()
    try:
        import yaml
        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if not isinstance(data, dict) or "passes" not in data:
        raise ValueError(f"spec {spec_fpath} must be a mapping with a 'passes' list")
    return data


def run(config):
    import kwcoco
    from kwcoco_detector_kit.data import tile as _tile

    src_fpath = Path(str(config.src)).expanduser().resolve()
    dst_fpath = Path(str(config.dst)).expanduser().resolve()
    if not src_fpath.exists():
        raise FileNotFoundError(src_fpath)
    dst_fpath.parent.mkdir(parents=True, exist_ok=True)

    spec = _load_spec(Path(str(config.spec)))
    shared = dict(spec.get("shared", {}) or {})
    passes = list(spec.get("passes", []) or [])
    if not passes:
        raise ValueError("spec has no passes")

    passes_dpath = Path(str(config.passes_dpath)).expanduser() if config.passes_dpath \
        else dst_fpath.parent / (dst_fpath.stem.replace(".kwcoco", "") + "_passes")
    passes_dpath.mkdir(parents=True, exist_ok=True)

    pass_fpaths = []
    for i, raw_pass in enumerate(passes):
        p = dict(shared)
        p.update(raw_pass or {})
        name = str(p.pop("name", p.get("mode", f"pass{i}")))
        pass_dst = passes_dpath / f"{i:02d}_{name}.kwcoco.zip"
        pass_fpaths.append(pass_dst)
        if pass_dst.exists() and not bool(config.force):
            print(f"[tile-corpus] pass {i} '{name}': reuse {pass_dst}")
            continue
        # Build a TileConfig for this pass (src/dst injected; rest from spec).
        pass_cfg = _tile.TileConfig(**{**p, "src": str(src_fpath), "dst": str(pass_dst)})
        print(f"[tile-corpus] pass {i} '{name}': mode={pass_cfg['mode']} -> {pass_dst}")
        _tile.run(pass_cfg)

    # Union all passes. Reroot each to absolute so the merged bundle resolves
    # its assets no matter where dst lives (passes keep their own asset dirs).
    print(f"[tile-corpus] unioning {len(pass_fpaths)} passes -> {dst_fpath}")
    dsets = []
    for fp in pass_fpaths:
        d = kwcoco.CocoDataset.coerce(str(fp))
        d.reroot(absolute=True)
        dsets.append(d)
    merged = kwcoco.CocoDataset.union(*dsets)
    merged.fpath = str(dst_fpath)
    merged.dump()
    print(f"[tile-corpus] wrote {dst_fpath}: {merged.n_images} images, "
          f"{merged.n_annots} annots from {len(pass_fpaths)} passes")
    return dst_fpath


__cli__ = TileCorpusConfig

if __name__ == "__main__":
    __cli__.main()
