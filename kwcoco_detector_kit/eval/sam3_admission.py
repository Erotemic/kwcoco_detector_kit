"""Small pretrained SAM3 admission test for a single deployment GPU.

The command intentionally runs *pretrained* text-prompted image inference only.
It answers the first questions before KDK grows a full SAM3 training adapter:

* Does SAM3.1 load and run on the deployment GPU?
* What are model-load and per-image peak VRAM footprints?
* What is eager inference latency on representative native-resolution chips?
* Does zero-shot text-prompted segmentation produce useful KWCoco masks?

Long loops always expose progress per the KDK maintainer invariant.
"""
from __future__ import annotations

import json
import math
import contextlib
import subprocess
import sys
import threading
import time
from pathlib import Path

import kwconf
import ubelt as ub

from kwcoco_detector_kit._provenance import provenance_dict



@contextlib.contextmanager
def _heartbeat(label, *, interval=15.0):
    """Emit periodic status while an opaque third-party call is running."""
    stop = threading.Event()
    start = time.perf_counter()

    def worker():
        while not stop.wait(float(interval)):
            elapsed = time.perf_counter() - start
            print(f"[{label}] still running; elapsed={elapsed:.1f}s", flush=True)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _choose_gids(dset, *, target_name, limit, seed, balanced):
    import numpy as np

    all_gids = list(dset.images())
    if limit is None or int(limit) <= 0 or int(limit) >= len(all_gids):
        return all_gids

    rng = np.random.RandomState(int(seed))
    limit = int(limit)
    if not balanced:
        chosen = rng.choice(all_gids, size=limit, replace=False)
        return [int(g) for g in chosen]

    cats = [c for c in dset.dataset.get("categories", []) if c.get("name") == target_name]
    if not cats:
        chosen = rng.choice(all_gids, size=limit, replace=False)
        return [int(g) for g in chosen]
    target_cids = {c["id"] for c in cats}
    pos_gids = {
        ann["image_id"]
        for ann in dset.dataset.get("annotations", [])
        if ann.get("category_id") in target_cids
    }
    positives = np.array(sorted(pos_gids), dtype=int)
    negatives = np.array(sorted(set(all_gids) - pos_gids), dtype=int)

    want_pos = min(len(positives), int(math.ceil(limit / 2)))
    want_neg = min(len(negatives), limit - want_pos)
    short = limit - want_pos - want_neg
    if short:
        extra_pos = min(short, len(positives) - want_pos)
        want_pos += extra_pos
        short -= extra_pos
    if short:
        want_neg += min(short, len(negatives) - want_neg)

    pos = rng.choice(positives, size=want_pos, replace=False) if want_pos else []
    neg = rng.choice(negatives, size=want_neg, replace=False) if want_neg else []
    chosen = np.array(list(pos) + list(neg), dtype=int)
    rng.shuffle(chosen)
    return [int(g) for g in chosen]


def _make_target_truth_subset(dset, gids, target_name, fpath):
    truth = dset.subset(gids=gids, copy=True)
    abs_paths = {gid: str(dset.get_image_fpath(gid)) for gid in gids}
    target_cids = {
        cat["id"] for cat in truth.dataset.get("categories", [])
        if cat.get("name") == target_name
    }
    truth.dataset["annotations"] = [
        ann for ann in truth.dataset.get("annotations", [])
        if ann.get("category_id") in target_cids
    ]
    truth.dataset["categories"] = [
        cat for cat in truth.dataset.get("categories", [])
        if cat.get("id") in target_cids
    ]
    for img in truth.dataset.get("images", []):
        img["file_name"] = abs_paths[img["id"]]
    truth._build_index()
    truth._update_fpath(str(fpath))
    truth.dump()
    return truth


def _empty_prediction_dataset(truth, fpath):
    pred = truth.copy()
    pred.dataset["annotations"] = []
    pred._build_index()
    pred._update_fpath(str(fpath))
    return pred


def _cuda_snapshot(torch, device):
    dev = torch.device(device)
    if dev.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(dev)
    return {
        "allocated_gb": torch.cuda.memory_allocated(dev) / 2**30,
        "reserved_gb": torch.cuda.memory_reserved(dev) / 2**30,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(dev) / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(dev) / 2**30,
        "free_gb": free / 2**30,
        "total_gb": total / 2**30,
    }


def run(config):
    import kwcoco
    import kwimage
    import numpy as np
    import torch

    from kwcoco_detector_kit.data.postprocess import (
        add_prediction_annotations,
        mask_records_to_anns,
    )
    from kwcoco_detector_kit.predictors.sam3 import SAM3TextPredictor, resolve_sam3_repo

    src = Path(config.src).expanduser().resolve()
    out_dpath = Path(config.out_dpath).expanduser().resolve()
    out_dpath.mkdir(parents=True, exist_ok=True)
    truth_fpath = out_dpath / "truth_subset.kwcoco.zip"
    pred_fpath = out_dpath / "predictions.kwcoco.zip"
    report_fpath = out_dpath / "admission_report.json"

    dset = kwcoco.CocoDataset.coerce(str(src))
    gids = _choose_gids(
        dset,
        target_name=str(config.target_category),
        limit=config.limit,
        seed=int(config.seed),
        balanced=bool(config.balanced),
    )
    truth = _make_target_truth_subset(
        dset, gids, str(config.target_category), truth_fpath
    )
    pred = _empty_prediction_dataset(truth, pred_fpath)

    device = str(config.device)
    dev = torch.device(device)
    gpu_info = None
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
        props = torch.cuda.get_device_properties(dev)
        gpu_info = {
            "name": props.name,
            "total_memory_gb": props.total_memory / 2**30,
            "compute_capability": [props.major, props.minor],
            "torch_cuda_version": torch.version.cuda,
        }
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)

    report = {
        "schema_version": 1,
        "src": str(src),
        "num_images": len(gids),
        "target_category": str(config.target_category),
        "prompt": str(config.prompt),
        "checkpoint_version": str(config.checkpoint_version),
        "checkpoint": str(config.checkpoint) if config.checkpoint else None,
        "device": device,
        "amp_dtype": str(config.amp_dtype),
        "resolution": int(config.resolution),
        "score_thresh": float(config.score_thresh),
        "windowed": bool(config.windowed),
        "overlap": float(config.overlap),
        "whole_image_pass": bool(config.whole_image_pass),
        "gpu": gpu_info,
        "sam3_repo": str(resolve_sam3_repo()) if resolve_sam3_repo() else None,
        "provenance": provenance_dict(),
    }

    print("[sam3-admission] loading pretrained model", flush=True)
    t0 = time.perf_counter()
    with _heartbeat("sam3-admission:model-load"):
        predictor = SAM3TextPredictor(
            prompt=str(config.prompt),
            device=device,
            checkpoint_version=str(config.checkpoint_version),
            checkpoint=config.checkpoint,
            score_thresh=float(config.score_thresh),
            resolution=int(config.resolution),
            amp_dtype=str(config.amp_dtype),
            compile=bool(config.compile),
        )
    if bool(config.windowed):
        from kwcoco_detector_kit.predictors.tiled import TiledPredictor

        predictor = TiledPredictor(
            predictor,
            window=(int(config.resolution), int(config.resolution)),
            overlap=float(config.overlap),
            nms_thresh=float(config.nms_thresh),
            keep_full=bool(config.whole_image_pass),
            batch_size=1,
            max_dets=int(config.max_dets) if config.max_dets else None,
            pre_nms_score_thresh=float(config.score_thresh),
        )
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    report["model_load_seconds"] = time.perf_counter() - t0
    report["after_model_load_cuda"] = _cuda_snapshot(torch, dev)

    post_cfg = {
        "score_thresh": float(config.score_thresh),
        "windowed": bool(config.windowed),
        "overlap": float(config.overlap),
        "whole_image_pass": bool(config.whole_image_pass),
        "nms_thresh": float(config.nms_thresh),
        "polygon_simplify": float(config.polygon_simplify),
        "min_component_area": float(config.min_component_area),
        "keep_largest_component": bool(config.keep_largest_component),
    }

    timings = []
    peaks = []
    counts = []
    prog = ub.ProgIter(gids, desc="sam3 admission images", verbose=3, time_thresh=1.0)
    for index, gid in enumerate(prog):
        fpath = dset.get_image_fpath(gid)
        image = kwimage.imread(fpath, space="rgb")
        h, w = image.shape[:2]
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
        start = time.perf_counter()
        records = predictor.predict_image(image, (w, h))
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        elapsed = time.perf_counter() - start
        snap = _cuda_snapshot(torch, dev)
        timings.append(elapsed)
        peaks.append(snap)
        counts.append(len(records))

        anns = mask_records_to_anns(
            records,
            post_cfg,
            label_mapping={0: str(config.target_category)},
        )
        add_prediction_annotations(pred, gid, anns, "sam3")
        prog.set_postfix(
            dets=len(records),
            sec=f"{elapsed:.2f}",
            peak=(f"{snap.get('peak_reserved_gb', 0):.1f}GB" if snap else "cpu"),
        )

    pred.dump()

    warmup = min(int(config.warmup_images), len(timings))
    measured = np.asarray(timings[warmup:] or timings, dtype=float)
    report["warmup_images"] = warmup
    report["latency_seconds"] = {
        "per_image": timings,
        "mean_after_warmup": float(measured.mean()) if len(measured) else None,
        "median_after_warmup": float(np.median(measured)) if len(measured) else None,
        "p95_after_warmup": float(np.quantile(measured, 0.95)) if len(measured) else None,
    }
    report["detections_per_image"] = {
        "counts": counts,
        "mean": float(np.mean(counts)) if counts else 0.0,
    }
    if peaks:
        report["cuda_peak_over_images"] = {
            key: max((p.get(key, 0.0) for p in peaks), default=0.0)
            for key in ["peak_allocated_gb", "peak_reserved_gb"]
        }
        total_gb = report["gpu"]["total_memory_gb"] if report.get("gpu") else None
        peak_reserved = report["cuda_peak_over_images"]["peak_reserved_gb"]
        if total_gb is not None:
            report["cuda_headroom_gb"] = float(total_gb - peak_reserved)

    report["truth_subset"] = str(truth_fpath)
    report["predictions"] = str(pred_fpath)

    if bool(config.evaluate):
        eval_dpath = out_dpath / "eval"
        eval_fpath = eval_dpath / "detect_metrics.json"
        eval_dpath.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "kwcoco", "eval",
            "--true_dataset", str(truth_fpath),
            "--pred_dataset", str(pred_fpath),
            "--out_dpath", str(eval_dpath),
            "--out_fpath", str(eval_fpath),
            "--draw", "False",
            "--iou_thresh", "0.5",
        ]
        print("[sam3-admission] " + " ".join(cmd), flush=True)
        result = subprocess.run(cmd)
        report["eval_returncode"] = int(result.returncode)
        report["eval_metrics"] = str(eval_fpath) if eval_fpath.exists() else None

    report_fpath.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[sam3-admission] report: {report_fpath}")
    return report_fpath


class SAM3AdmissionConfig(kwconf.Config):
    """Measure pretrained SAM3.1 zero-shot segmentation on representative KWCoco chips."""

    src = kwconf.Value(None, required=True, help="source KWCoco dataset")
    out_dpath = kwconf.Value(None, required=True, help="output directory")
    target_category = kwconf.Value("poop", help="truth/output category name")
    prompt = kwconf.Value("poop", help="SAM3 text prompt")
    device = kwconf.Value("cuda:0")
    checkpoint_version = kwconf.Value("sam3.1", choices=["sam3", "sam3.1"])
    checkpoint = kwconf.Value(None, help="optional local checkpoint; skips HF download")
    resolution = kwconf.Value(1008, parser=int)
    amp_dtype = kwconf.Value("bfloat16", choices=["bfloat16", "float16", "none"])
    compile = kwconf.Value(False, isflag=True)
    score_thresh = kwconf.Value(0.05, parser=float)
    nms_thresh = kwconf.Value(0.5, parser=float)
    polygon_simplify = kwconf.Value(1.0, parser=float)
    min_component_area = kwconf.Value(4.0, parser=float)
    keep_largest_component = kwconf.Value(False, isflag=True)
    limit = kwconf.Value(32, parser=int, help="number of representative images; <=0 means all")
    seed = kwconf.Value(0, parser=int)
    balanced = kwconf.Value(True, isflag=True, help="sample roughly half positive / half target-negative")
    warmup_images = kwconf.Value(2, parser=int)
    windowed = kwconf.Value(False, isflag=True, help="run native-resolution 1008px windows over each source image")
    overlap = kwconf.Value(0.25, parser=float)
    whole_image_pass = kwconf.Value(False, isflag=True, help="also resize/run the entire image in windowed mode")
    max_dets = kwconf.Value(300, parser=int, help="cap merged detections per source image in windowed mode")
    evaluate = kwconf.Value(True, isflag=True)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        return run(config)


__cli__ = SAM3AdmissionConfig
