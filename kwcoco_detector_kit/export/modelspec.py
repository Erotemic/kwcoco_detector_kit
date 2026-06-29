"""
Modelspec sidecar — a small JSON file written next to every exported
``.onnx`` describing the deploy contract:

- input_hw: model's fixed input H/W (for fixed-shape ONNX exports)
- input_channels: 3 for RGB; ``len(channels.split('|'))`` for multispectral
- input_dtype: 'float32' (preprocessed image; not the raw uint8 buffer)
- input_layout: 'NCHW' | 'NHWC'
- preprocess: {scale: 1/255, normalize_mean: [...], normalize_std: [...]}
- postprocess: {score_thresh, nms_iou_thresh, topk}
- modelId: canonical cross-device ID
- meta: variant, category_names, candidate_kind, generated_at, kit_version
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence


DEFAULT_NORMALIZE_MEAN = (0.0, 0.0, 0.0)
DEFAULT_NORMALIZE_STD = (1.0, 1.0, 1.0)


def write_labels_txt(onnx_fpath: Path, category_names: Sequence[str]) -> Path:
    """Write ``<onnx_fpath>.labels.txt`` next to ``onnx_fpath``.

    One category name per line, 0-indexed.  No ``__background__`` prefix —
    DEIMv2 outputs labels 0…N-1 for N real classes.
    """
    labels_fpath = Path(onnx_fpath).with_suffix(".labels.txt")
    labels_fpath.write_text("\n".join(category_names) + "\n")
    return labels_fpath


def write_modelspec(
    onnx_fpath: Path,
    *,
    input_hw: tuple,
    input_channels: int = 3,
    input_dtype: str = "float32",
    input_layout: str = "NCHW",
    preprocess_scale: float = 1.0 / 255.0,
    normalize_mean: Iterable[float] = DEFAULT_NORMALIZE_MEAN,
    normalize_std: Iterable[float] = DEFAULT_NORMALIZE_STD,
    postprocess_score_thresh: float = 0.30,
    postprocess_nms_iou_thresh: float = 0.50,
    postprocess_topk: int = 100,
    model_id: Optional[str] = None,
    variant: str = "",
    category_names: Sequence[str] = (),
    candidate_kind: str = "",
    category_names_source: Optional[str] = None,
    source_checkpoint: Optional[dict] = None,
    provenance: Optional[dict] = None,
    imputed: Optional[dict] = None,
    extra_meta: Optional[dict] = None,
) -> Path:
    """Write ``<onnx_fpath>.modelspec.json`` next to ``onnx_fpath``.

    Provenance / trust args
    -----------------------
    category_names_source
        Free-text note on where ``category_names`` came from, e.g.
        ``"train_kwcoco"`` (authoritative), ``"policy.json"``, ``"cli"``, or
        ``"imputed:class_schemes.yaml:pup_vs_nonpup"``. Recorded under
        ``meta.category_names_source``.
    source_checkpoint
        Pre-computed fingerprint of the ``.pth`` that produced this ONNX,
        e.g. ``{"name", "sha256", "size_bytes"}``. Merged into the
        ``provenance`` block so the sidecar is self-describing.
    provenance
        Kit + submodule SHAs (typically ``_provenance.provenance_dict()``).
    imputed
        Mapping of ``field -> reason`` for any metadata that was NOT derived
        from a clean data-driven path but inferred from a secondary artifact.
        When non-empty, a top-level ``"imputed"`` block is written and
        ``meta.has_imputed_metadata`` is set so downstream systems can treat
        those fields with suspicion.
    """
    from kwcoco_detector_kit import __version__ as _kit_version

    onnx_fpath = Path(onnx_fpath)
    H, W = int(input_hw[0]), int(input_hw[1])
    if model_id is None:
        model_id = f"{variant}-h{H}w{W}" if variant else onnx_fpath.stem

    imputed = {k: v for k, v in (imputed or {}).items() if v}

    meta = {
        "variant": variant,
        "category_names": list(category_names),
        "candidate_kind": candidate_kind,
        "kit_version": _kit_version,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "has_imputed_metadata": bool(imputed),
        **(extra_meta or {}),
    }
    if category_names_source is not None:
        meta["category_names_source"] = str(category_names_source)

    spec = {
        "modelId": model_id,
        "input": {
            "shape_hw": [H, W],
            "channels": int(input_channels),
            "dtype": str(input_dtype),
            "layout": str(input_layout),
        },
        "preprocess": {
            "scale": float(preprocess_scale),
            "normalize_mean": [float(v) for v in normalize_mean],
            "normalize_std": [float(v) for v in normalize_std],
        },
        "postprocess": {
            "score_thresh": float(postprocess_score_thresh),
            "nms_iou_thresh": float(postprocess_nms_iou_thresh),
            "topk": int(postprocess_topk),
        },
        "meta": meta,
    }

    prov: dict = {}
    if provenance:
        prov.update(provenance)
    if source_checkpoint:
        prov["source_checkpoint"] = dict(source_checkpoint)
    if prov:
        spec["provenance"] = prov
    if imputed:
        spec["imputed"] = dict(imputed)

    out_fpath = onnx_fpath.with_suffix(".modelspec.json")
    out_fpath.write_text(json.dumps(spec, indent=2))
    if category_names:
        write_labels_txt(onnx_fpath, list(category_names))
    return out_fpath
