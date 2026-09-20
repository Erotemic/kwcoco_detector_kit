"""Package-aware detector prediction over original KWCoco source images.

The public path supports both whole-image and no-cache tiled inference. Tiled
predictions are always merged back into source-image coordinates before they
are written to the prediction KWCoco dataset.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import kwconf

from kwcoco_detector_kit.export.package import materialize_workdir, open_package

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def _coerce_rgb(image):
    """Normalize kwcoco-loaded image arrays into HWC RGB."""
    import numpy as np

    arr = image
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _sha256_file(fpath):
    h = hashlib.sha256()
    with open(fpath, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_labels(package_root: Path, manifest: dict) -> list[str]:
    artifacts = manifest.get("artifacts", {})
    labels_rel = artifacts.get("labels")
    if labels_rel:
        labels_fpath = package_root / labels_rel
        if labels_fpath.exists():
            data = json.loads(labels_fpath.read_text())
            labels = data.get("labels")
            if labels:
                return [str(lbl) for lbl in labels]
    category_names = manifest.get("category_names")
    if category_names:
        return [str(n) for n in category_names]
    return ["object"]


def _build_kwcoco_from_image_dir(image_dpath: Path):
    import kwcoco

    dset = kwcoco.CocoDataset()
    for fpath in sorted(image_dpath.iterdir()):
        if fpath.is_file() and fpath.suffix.lower() in _IMAGE_EXTS:
            dset.add_image(file_name=str(fpath.resolve()))
    return dset


def _clone_for_predictions(true_dset, dst_path=None):
    """Clone source image records and IDs while dropping source annotations."""
    import kwcoco

    pred = kwcoco.CocoDataset()
    if dst_path is not None:
        pred.fpath = str(dst_path)
    for video in true_dset.dataset.get("videos", []):
        video_data = {k: v for k, v in video.items() if k != "id"}
        pred.add_video(**video_data, id=video["id"])
    for img in true_dset.images().objs:
        new = {k: v for k, v in img.items() if k != "id"}
        try:
            new["file_name"] = str(true_dset.get_image_fpath(img["id"]))
        except Exception:
            pass
        pred.add_image(**new, id=img["id"])
    return pred


def _coerce_src_kwcoco(src: str | Path):
    import kwcoco

    src = Path(src).expanduser()
    if src.is_dir():
        candidates = sorted(src.glob("*.kwcoco.*")) + sorted(src.glob("*.kwcoco"))
        if candidates:
            return kwcoco.CocoDataset.coerce(str(candidates[0]))
        return _build_kwcoco_from_image_dir(src)
    return kwcoco.CocoDataset.coerce(str(src))


def _build_segmenter(manifest: dict, package_root: Path):
    from kwcoco_detector_kit.trainers.sam2 import SAM2Segmenter

    segmenter_cfg = dict(manifest.get("segmenter") or {})
    ckpt = segmenter_cfg.get("checkpoint_fpath")
    if ckpt:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            resolved = (package_root / ckpt_path).resolve()
            if resolved.exists():
                segmenter_cfg["checkpoint_fpath"] = str(resolved)
    return SAM2Segmenter(segmenter_cfg)


def _coerce_hw(value, fallback):
    if value is None:
        value = fallback
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("x", ",").split(",") if p.strip()]
        if len(parts) == 1:
            return int(parts[0]), int(parts[0])
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        raise ValueError(f"cannot parse H,W from {value!r}")
    if isinstance(value, (int, float)):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError(f"window must contain H,W; got {value!r}")
    return int(value[0]), int(value[1])


def _build_package_predictor(
    *, package_root, manifest, materialized, trainer, device, score_thresh, backend
):
    """Select verified ONNX when safe; otherwise build the trainer predictor."""
    requested = str(backend or "auto").lower()
    if requested not in {"auto", "onnx", "torch"}:
        raise ValueError("backend must be one of auto, onnx, torch")

    artifacts = manifest.get("artifacts", {}) or {}
    onnx_rel = artifacts.get("onnx")
    onnx_meta = manifest.get("onnx") or {}
    capabilities = manifest.get("capabilities") or {}
    package_needs_masks = bool(capabilities.get("supports_masks", False))
    parity_ok = bool(onnx_meta.get("parity_passed", False))
    onnx_has_masks = bool(onnx_meta.get("supports_masks", False))
    parity_score_thresh = onnx_meta.get("parity_score_thresh")
    parity_covers_request = bool(
        parity_score_thresh is not None
        and float(score_thresh) + 1e-12 >= float(parity_score_thresh)
    )
    can_use_onnx = bool(
        onnx_rel
        and parity_ok
        and parity_covers_request
        and (not package_needs_masks or onnx_has_masks)
    )

    if requested == "onnx" and not can_use_onnx:
        raise RuntimeError(
            "ONNX was explicitly requested, but the package does not contain a "
            "parity-verified ONNX artifact with all required capabilities"
        )

    if requested in {"auto", "onnx"} and can_use_onnx:
        onnx_fpath = package_root / onnx_rel
        try:
            contract = onnx_meta.get("contract")
            if contract == "rfdetr_raw_seg_v1":
                from kwcoco_detector_kit.predictors.rfdetr_onnx import RFDETROnnxPredictor

                predictor = RFDETROnnxPredictor(
                    onnx_fpath,
                    device=str(device),
                    score_thresh=float(score_thresh),
                    category_names=manifest.get("category_names") or [],
                )
            else:
                from kwcoco_detector_kit.predictors.onnx import OnnxPredictor

                predictor = OnnxPredictor(
                    onnx_fpath, device=str(device), score_thresh=float(score_thresh)
                )
            return predictor, "onnx"
        except Exception as ex:
            if requested == "onnx":
                raise
            print(f"predict: verified ONNX unavailable at runtime ({ex}); using torch")

    predictor = trainer.build_predictor(materialized, device=str(device))
    set_score_thresh = getattr(predictor, "set_score_thresh", None)
    if callable(set_score_thresh):
        set_score_thresh(float(score_thresh))
    return predictor, "torch"


def predict_kwcoco(
    *,
    package: str | Path | None = None,
    model: str | Path | None = None,
    src: str | Path,
    dst: str | Path,
    device: str = "cpu",
    score_thresh: Optional[float] = None,
    nms_thresh: Optional[float] = None,
    workdir: Optional[str | Path] = None,
    backend: str = "auto",
    windowed: Optional[bool] = None,
    window=None,
    overlap: Optional[float] = None,
    batch_size: int = 16,
    source_read_strategy: str = "auto",
    prediction_scale=1.0,
    whole_image_pass: Optional[bool] = None,
    max_dets: Optional[int] = None,
    pipeline: bool = True,
    source_workers: int = 2,
    source_prefetch: int = 2,
    window_prefetch: int = 2,
    postprocess_workers: int = 1,
    postprocess_inflight: int = 2,
) -> Path:
    """Run one self-describing model package over original source KWCoco.

    Tiled mode uses :class:`SourceWindowReader` and :class:`TiledPredictor` to
    realize windows only for the duration of this pass. No tile assets or cache
    are created. Boxes and masks are translated back to original source-image
    coordinates before writing.
    """
    import tempfile

    from kwcoco_detector_kit.data.postprocess import (
        add_prediction_annotations,
        detector_records_to_anns,
        detector_records_to_bbox_anns,
        mask_records_to_anns,
    )
    import kwcoco_detector_kit.trainers  # noqa: F401 - register plugins
    from kwcoco_detector_kit.trainers._registry import get_trainer

    model_path = package if package is not None else model
    if model_path is None:
        raise ValueError("one of package= or model= is required")
    model_path = Path(model_path).expanduser()
    dst = Path(dst).expanduser()
    dst.parent.mkdir(parents=True, exist_ok=True)

    package_sha256 = _sha256_file(model_path) if model_path.is_file() else None
    src_path = Path(src).expanduser()
    source_sha256 = _sha256_file(src_path) if src_path.is_file() else None

    with open_package(model_path) as (package_root, manifest):
        trainer = get_trainer(str(manifest["trainer"]))
        post_manifest = manifest.get("postprocess", {}) or {}
        inference_manifest = manifest.get("inference", {}) or {}
        if score_thresh is None:
            score_thresh = float(post_manifest.get("score_thresh", 0.30))
        if nms_thresh is None:
            nms_thresh = float(
                inference_manifest.get(
                    "nms_iou", post_manifest.get("nms_iou_thresh", 0.50)
                )
            )
        if windowed is None:
            windowed = str(inference_manifest.get("mode", "whole_image")) == "windowed"
        if overlap is None:
            overlap = float(inference_manifest.get("overlap", 0.25))
        if whole_image_pass is None:
            whole_image_pass = bool(inference_manifest.get("whole_image_pass", False))

        labels = _load_labels(package_root, manifest)
        label_mapping = {i: name for i, name in enumerate(labels)}
        post_cfg = {
            "score_thresh": float(score_thresh),
            "nms_thresh": float(nms_thresh),
            "crop_padding": float(post_manifest.get("crop_padding", 10)),
            "polygon_simplify": float(post_manifest.get("polygon_simplify", 1.0)),
            "min_component_area": float(post_manifest.get("min_component_area", 50.0)),
            "keep_largest_component": bool(post_manifest.get("keep_largest_component", True)),
        }

        model_pipeline = str(manifest.get("pipeline", "detector_only"))
        segmenter = (
            _build_segmenter(manifest, package_root)
            if model_pipeline == "detector_segmenter" else None
        )

        tmp_ctx = None
        if workdir is None:
            tmp_ctx = tempfile.TemporaryDirectory()
            materialized = Path(tmp_ctx.name) / "workdir"
        else:
            materialized = Path(workdir).expanduser()
        try:
            materialize_workdir(package_root, manifest, materialized)
            base_predictor, backend_used = _build_package_predictor(
                package_root=package_root,
                manifest=manifest,
                materialized=materialized,
                trainer=trainer,
                device=device,
                score_thresh=float(score_thresh),
                backend=backend,
            )

            tiled = None
            resolved_window = None
            if windowed:
                from kwcoco_detector_kit.predictors.tiled import TiledPredictor

                resolved_window = _coerce_hw(
                    window,
                    inference_manifest.get("window") or getattr(base_predictor, "eval_spatial_size", None),
                )
                if resolved_window is None:
                    raise ValueError("windowed prediction needs a window size or predictor eval_spatial_size")
                tiled = TiledPredictor(
                    base_predictor,
                    window=resolved_window,
                    overlap=float(overlap),
                    nms_thresh=float(nms_thresh),
                    keep_full=bool(whole_image_pass),
                    batch_size=int(batch_size),
                    max_dets=max_dets,
                    pre_nms_score_thresh=float(score_thresh),
                    per_window_nms=True,
                )

            true = _coerce_src_kwcoco(src)
            pred = _clone_for_predictions(true, dst)
            backend_name = f"{manifest['trainer']}:{backend_used}"
            strategy_counts = {}
            total_decode = 0.0
            total_window_read = 0.0
            total_whole_infer = 0.0
            inline_postprocess_seconds = 0.0
            annotation_commit_seconds = 0.0
            committed_images = 0
            last_progress_report = 0.0
            started = time.perf_counter()

            from kwcoco_detector_kit.predictors.pipeline import PredictionPipeline
            from kwcoco_detector_kit.predictors.source_window import SourceWindowReader

            if not pipeline:
                source_workers = 0
                source_prefetch = 0
                window_prefetch = 0
                postprocess_workers = 0

            source_items = (
                (int(gid), true.coco_image(gid))
                for gid in true.images()
            )

            def _prepare_source(item):
                gid, coco_img = item
                try:
                    reader = SourceWindowReader(
                        coco_img,
                        strategy=str(source_read_strategy),
                        prediction_scale=prediction_scale,
                    )
                    # Whole-image prediction and detector+segmenter pipelines
                    # need the full array anyway. Native tiled detectors keep
                    # TIFF/COG sources lazy and only decode JPEG-like sources.
                    reader.prepare(force_full=(tiled is None or segmenter is not None))
                    return reader
                except Exception as ex:
                    raise RuntimeError(
                        f"predict: failed to prepare gid {gid}; refusing partial output"
                    ) from ex

            tiled_post_cfg = dict(post_cfg)
            tiled_post_cfg["nms_thresh"] = 0.0

            def _finalize_source(gid, payload, arr, prediction_space):
                try:
                    if tiled is not None:
                        raw = payload
                        if raw.get("deferred") is not None:
                            records = raw["deferred"]
                        else:
                            records = tiled._merge_and_nms(raw)
                        ann_cfg = tiled_post_cfg
                    else:
                        records = payload
                        ann_cfg = post_cfg

                    if segmenter is not None:
                        anns = detector_records_to_anns(
                            arr, records, segmenter, ann_cfg, label_mapping,
                            prediction_space=prediction_space,
                        )
                    elif records and all("mask" in record for record in records):
                        anns = mask_records_to_anns(
                            records, ann_cfg, label_mapping,
                            prediction_space=prediction_space,
                        )
                    else:
                        if manifest.get("capabilities", {}).get("supports_masks") and records:
                            raise RuntimeError(
                                "model package advertises native masks but selected backend returned box-only records"
                            )
                        anns = detector_records_to_bbox_anns(
                            records, ann_cfg, label_mapping,
                            prediction_space=prediction_space,
                        )
                    return anns
                except Exception as ex:
                    raise RuntimeError(
                        f"predict: failed postprocess gid {gid}; refusing partial output"
                    ) from ex

            def _commit_ready(ready):
                nonlocal annotation_commit_seconds
                nonlocal committed_images
                nonlocal last_progress_report
                for gid, anns in ready:
                    commit_started = time.perf_counter()
                    add_prediction_annotations(pred, gid, anns, backend_name)
                    annotation_commit_seconds += time.perf_counter() - commit_started
                    committed_images += 1
                    now = time.perf_counter()
                    should_report = (
                        committed_images == 1
                        or committed_images == int(true.n_images)
                        or committed_images % 25 == 0
                        or now - last_progress_report >= 10.0
                    )
                    if should_report:
                        wall = max(now - started, 1e-9)
                        image_rate = committed_images / wall
                        window_text = ""
                        if tiled is not None and tiled.n_windows:
                            window_text = f", {tiled.n_windows / wall:.1f} windows/s"
                        print(
                            f"predict: finalized {committed_images}/{true.n_images} "
                            f"({image_rate:.2f} images/s{window_text})",
                            flush=True,
                        )
                        last_progress_report = now

            # CUDA is intentionally owned by this main thread. Source decode,
            # future window realization, and CPU merge/polygonization use
            # separate bounded worker pools so they can overlap without model
            # duplication or large multiprocessing IPC copies.
            with PredictionPipeline(
                source_workers=int(source_workers),
                source_prefetch=int(source_prefetch),
                window_prefetch=int(window_prefetch),
                postprocess_workers=int(postprocess_workers),
                postprocess_inflight=int(postprocess_inflight),
            ) as pred_pipeline:
                prepared_iter = pred_pipeline.iter_prepared_sources(
                    source_items, _prepare_source
                )
                for (gid, _coco_img), reader in prepared_iter:
                    # Bound memory before producing one more source's raw masks
                    # / detections. Completed CPU results are committed in the
                    # same order as source images regardless of worker timing.
                    _commit_ready(
                        pred_pipeline.wait_for_postprocess_capacity(reserve=1)
                    )
                    try:
                        H, W = reader.prediction_hw
                        if tiled is not None:
                            payload = tiled._infer_source(
                                reader, (W, H), pipeline=pred_pipeline
                            )
                            # Segmenter pipelines need pixels after detector
                            # merge. They stay synchronous because the
                            # segmenter may itself own CUDA.
                            arr = reader.read_full() if segmenter is not None else None
                        else:
                            arr = reader.read_full()
                            infer_started = time.perf_counter()
                            payload = base_predictor.predict_image(arr, (W, H))
                            total_whole_infer += time.perf_counter() - infer_started
                    except Exception as ex:
                        raise RuntimeError(
                            f"predict: failed inference gid {gid}; refusing partial output"
                        ) from ex

                    strategy_counts[reader.strategy] = (
                        strategy_counts.get(reader.strategy, 0) + 1
                    )
                    total_decode += reader.t_decode
                    total_window_read += reader.t_window_read

                    can_async_finalize = (
                        pred_pipeline.async_postprocess and segmenter is None
                    )
                    if can_async_finalize:
                        # Detector-only finalization does not need source
                        # pixels. Avoid retaining a decoded whole image in the
                        # bounded postprocess queue for whole-image predictors.
                        finalize_arr = None
                        pred_pipeline.submit_postprocess(
                            gid, _finalize_source, gid, payload, finalize_arr, reader.space
                        )
                        _commit_ready(pred_pipeline.drain_postprocess_ready())
                    else:
                        post_started = time.perf_counter()
                        anns = _finalize_source(gid, payload, arr, reader.space)
                        inline_postprocess_seconds += time.perf_counter() - post_started
                        _commit_ready([(gid, anns)])

                _commit_ready(pred_pipeline.finish_postprocess())
                pipeline_profile = pred_pipeline.profile_dict()
                async_post_work = pipeline_profile["postprocess"]["work_seconds"]
                pipeline_profile["postprocess"]["inline_work_seconds"] = (
                    inline_postprocess_seconds
                )
                pipeline_profile["postprocess"]["total_work_seconds"] = (
                    async_post_work + inline_postprocess_seconds
                )

            elapsed = time.perf_counter() - started
            info = {
                "type": "kwcoco_detector_kit.predict",
                "package_schema": manifest.get("schema"),
                "package_sha256": package_sha256,
                "checkpoint_sha256": (manifest.get("checkpoint") or {}).get("sha256"),
                "onnx_sha256": (manifest.get("onnx") or {}).get("sha256") if backend_used == "onnx" else None,
                "trainer": manifest.get("trainer"),
                "variant": manifest.get("variant"),
                "backend": backend_used,
                "kdk_git_commit": (manifest.get("provenance") or {}).get("kdk_git_commit"),
                "kdk_version": (manifest.get("provenance") or {}).get("kdk_version"),
                "category_names": labels,
                "source_dataset": str(src_path),
                "source_dataset_sha256": source_sha256,
                "resolved_inference": {
                    "windowed": bool(windowed),
                    "window": list(resolved_window) if resolved_window else None,
                    "overlap": float(overlap),
                    "batch_size": int(batch_size),
                    "score_thresh": float(score_thresh),
                    "nms_thresh": float(nms_thresh),
                    "source_read_strategy": str(source_read_strategy),
                    "prediction_scale": prediction_scale,
                    "source_read_strategy_counts": strategy_counts,
                    "whole_image_pass": bool(whole_image_pass),
                    "pipeline": bool(pipeline),
                    "source_workers": int(source_workers),
                    "source_prefetch": int(source_prefetch),
                    "window_prefetch": int(window_prefetch),
                    "postprocess_workers": int(postprocess_workers),
                    "postprocess_inflight": int(postprocess_inflight),
                },
                "pipeline": pipeline_profile,
                "timing": {
                    "prediction_wall_seconds": elapsed,
                    # Stage work can overlap, so these work totals are not
                    # expected to sum to prediction_wall_seconds.
                    "source_decode_seconds": total_decode,
                    "window_read_seconds": total_window_read,
                    "source_prefetch_wait_seconds": pipeline_profile["source_prepare"]["wait_seconds"],
                    "window_prefetch_wait_seconds": pipeline_profile["window_read"]["wait_seconds"],
                    "model_infer_seconds": (
                        getattr(tiled, "t_infer", None)
                        if tiled is not None else total_whole_infer
                    ),
                    "merge_nms_seconds": getattr(tiled, "t_nms", None),
                    "postprocess_work_seconds": pipeline_profile["postprocess"]["total_work_seconds"],
                    "postprocess_wait_seconds": pipeline_profile["postprocess"]["wait_seconds"],
                    "annotation_commit_seconds": annotation_commit_seconds,
                    "source_images": int(true.n_images),
                    "windows": getattr(tiled, "n_windows", None),
                    "source_images_per_second": (true.n_images / elapsed) if elapsed > 0 else None,
                    "windows_per_second": (
                        getattr(tiled, "n_windows", 0) / elapsed
                        if elapsed > 0 and tiled is not None else None
                    ),
                },
            }
            pred.dataset.setdefault("info", []).append(info)
            _serialize_t0 = time.perf_counter()
            pred.dump()
            serialization_seconds = time.perf_counter() - _serialize_t0
            profile = {
                "schema": "kwcoco_detector_kit.predict_profile.v1",
                "prediction": str(dst),
                "package_sha256": package_sha256,
                "source_dataset_sha256": source_sha256,
                "backend": backend_used,
                "resolved_inference": info["resolved_inference"],
                "pipeline": pipeline_profile,
                "timing": {**info["timing"], "output_serialization_seconds": serialization_seconds},
            }
            Path(str(dst) + ".profile.json").write_text(
                json.dumps(profile, indent=2, sort_keys=True) + "\n"
            )
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    return dst


class PredictConfig(kwconf.Config):
    """Run packaged detector inference over original source KWCoco images."""

    model = kwconf.Value(None, help="self-describing model package (preferred spelling)")
    package = kwconf.Value(None, help="alias of --model for compatibility")
    src = kwconf.Value(None, required=True, help="source kwcoco dataset or image directory")
    dst = kwconf.Value(None, required=True, help="prediction kwcoco output path")
    device = kwconf.Value("cpu", help="device, e.g. cpu or cuda:0")
    backend = kwconf.Value("auto", choices=["auto", "onnx", "torch"])
    score_thresh = kwconf.Value(None, parser=float)
    nms_thresh = kwconf.Value(None, parser=float)
    workdir = kwconf.Value(None, help="optional persistent materialized predictor workdir")
    windowed = kwconf.Value(
        None,
        isflag=True,
        help=(
            "override package inference mode: --windowed=true forces tiled "
            "source-space prediction; --windowed=false forces whole-image "
            "prediction; omitted uses the package default"
        ),
    )
    window = kwconf.Value(None, help="window H,W or scalar; package default when omitted")
    overlap = kwconf.Value(None, parser=float)
    batch_size = kwconf.Value(16, parser=int, help="maximum realized windows per model batch")
    source_read_strategy = kwconf.Value(
        "auto", choices=["auto", "decode_once", "delayed_region"]
    )
    prediction_scale = kwconf.Value(
        1.0,
        parser=float,
        help=(
            "linear source-image scale used for detector inference; 1.0 is "
            "native resolution, 0.4 predicts on a 40% delayed-image view. "
            "All output annotations are mapped back to native image space."
        ),
    )
    pipeline = kwconf.Value(
        True,
        isflag=True,
        help=(
            "overlap source/window reads and CPU postprocess with GPU inference; "
            "use --pipeline=false for a fully serial diagnostic run"
        ),
    )
    source_workers = kwconf.Value(
        2, parser=int,
        help="threads preparing upcoming source images (0 disables source prefetch)",
    )
    source_prefetch = kwconf.Value(
        2, parser=int,
        help="future source images queued beyond the current source",
    )
    window_prefetch = kwconf.Value(
        2, parser=int,
        help="future realized window batches queued beyond the current GPU batch",
    )
    postprocess_workers = kwconf.Value(
        1, parser=int,
        help="CPU workers for tiled merge/NMS and mask polygonization",
    )
    postprocess_inflight = kwconf.Value(
        2, parser=int,
        help="bounded number of source results awaiting CPU postprocess",
    )
    whole_image_pass = kwconf.Value(
        None, isflag=True, help="include an additional whole-image pass; package default when omitted"
    )
    max_dets = kwconf.Value(None, parser=int)
    create_labelme = kwconf.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        model = config.model or config.package
        if not model:
            raise ValueError("--model is required (or legacy --package)")
        windowed = config.windowed
        out = predict_kwcoco(
            model=model,
            src=config.src,
            dst=config.dst,
            device=str(config.device),
            backend=str(config.backend),
            score_thresh=config.score_thresh,
            nms_thresh=config.nms_thresh,
            workdir=config.workdir,
            windowed=windowed,
            window=config.window,
            overlap=config.overlap,
            batch_size=int(config.batch_size),
            source_read_strategy=str(config.source_read_strategy),
            prediction_scale=float(config.prediction_scale),
            whole_image_pass=config.whole_image_pass,
            max_dets=config.max_dets,
            pipeline=bool(config.pipeline),
            source_workers=int(config.source_workers),
            source_prefetch=int(config.source_prefetch),
            window_prefetch=int(config.window_prefetch),
            postprocess_workers=int(config.postprocess_workers),
            postprocess_inflight=int(config.postprocess_inflight),
        )
        if config.create_labelme:
            from kwcoco_detector_kit.export.labelme import export_to_labelme

            written = export_to_labelme(out, only_missing=True)
            print(f"wrote {len(written)} LabelMe sidecar(s)")
        print(f"wrote predictions: {out}")
        return 0


__cli__ = PredictConfig
