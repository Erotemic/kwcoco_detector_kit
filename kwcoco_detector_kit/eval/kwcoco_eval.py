"""
kwcoco eval driver — write predictions for every image in a test bundle,
then subprocess ``python -m kwcoco eval`` to compute detection metrics.

Output layout (mirrors prior project so eligibility.py finds it)::

  <kcd_root>/eval/<candidate_id>/
    pred_boxes.kwcoco.zip
    eval/
      detect_metrics.json
      confusion.kwcoco.zip
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence, Tuple

from kwcoco_detector_kit._lineprofile import profile


def _valid_detection_bbox(bbox) -> bool:
    """True iff ``bbox`` is a concrete kwcoco detection box."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    return all(v is not None for v in bbox)


def _iter_prefetched(items, read_fn, workers, ahead=None):
    """Yield ``(item, read_fn(item))`` in order, decoding ahead on threads.

    Keeps up to ``ahead`` reads in flight so slow per-item I/O (JPEG decode
    off HDD) overlaps with whatever the consumer does between yields (GPU
    inference). Order is preserved. ``workers<=0`` falls back to a plain
    sequential read with no threads.
    """
    if workers is None or workers <= 0:
        for it in items:
            yield it, read_fn(it)
        return
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    if ahead is None:
        ahead = 2 * workers
    ahead = max(ahead, 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pending = deque()
        src = iter(items)
        for _ in range(ahead):
            try:
                nxt = next(src)
            except StopIteration:
                break
            pending.append((nxt, ex.submit(read_fn, nxt)))
        while pending:
            item, fut = pending.popleft()
            try:
                nxt = next(src)
                pending.append((nxt, ex.submit(read_fn, nxt)))
            except StopIteration:
                pass
            yield item, fut.result()


def filter_bbox_only_kwcoco(src_fpath, dst_fpath) -> Tuple[Path, int, int]:
    """Write a copy of ``src_fpath`` with non-detection annotations removed.

    Some kwcoco datasets carry image-level/caption-only annotations or other
    task metadata in the annotation table. Those rows are valid for broader
    kwcoco workflows, but ``kwcoco eval``'s detection coercion expects every
    annotation row it sees to have a length-4 ``bbox``. Filtering at the eval
    boundary keeps this toolkit detection-focused without mutating the user's
    source dataset.

    Returns:
        ``(dst_fpath, kept, dropped)``.
    """
    import kwcoco

    src_fpath = Path(src_fpath)
    dst_fpath = Path(dst_fpath)
    if (
        dst_fpath.exists()
        and dst_fpath.stat().st_mtime >= src_fpath.stat().st_mtime
    ):
        dset = kwcoco.CocoDataset.coerce(str(dst_fpath))
        return dst_fpath, len(dset.dataset.get("annotations", [])), 0

    dset = kwcoco.CocoDataset.coerce(str(src_fpath))
    abs_image_fpaths = {}
    for img in dset.dataset.get("images", []):
        try:
            abs_image_fpaths[img["id"]] = str(dset.get_image_fpath(img["id"]))
        except Exception:
            pass
    drop_ids = []
    kept = 0
    for ann in list(dset.anns.values()):
        if _valid_detection_bbox(ann.get("bbox")):
            kept += 1
        else:
            drop_ids.append(ann["id"])
    for aid in drop_ids:
        dset.remove_annotation(aid)
    for img in dset.dataset.get("images", []):
        if img["id"] in abs_image_fpaths:
            img["file_name"] = abs_image_fpaths[img["id"]]

    dst_fpath.parent.mkdir(parents=True, exist_ok=True)
    # remove_annotation() invalidates the imgs/anns indexes (sets them to
    # None in modern kwcoco), and _update_fpath() -> reroot() needs
    # len(self.imgs). Rebuild the index before the save so the reroot path
    # doesn't trip TypeError: object of type 'NoneType' has no len().
    dset._build_index()
    dset._update_fpath(str(dst_fpath))
    dset.dump()
    return dst_fpath, kept, len(drop_ids)


def _distractor_sidecar_fpath(metrics_fpath: Path, distractor_classes) -> Path:
    """Sidecar filename for the NFS-excluded (or other-excluded) eval pass.

    A single excluded class produces ``detect_metrics.<name>.json``;
    multiple classes are joined by ``_`` so the filename remains
    self-describing. This is the file eligibility's model-selection
    path looks for first.
    """
    if not distractor_classes:
        return metrics_fpath
    suffix = "_".join(sorted(c.strip() for c in distractor_classes if c.strip()))
    if not suffix:
        return metrics_fpath
    # detect_metrics.json -> detect_metrics.<suffix>.json
    return metrics_fpath.with_name(f"{metrics_fpath.stem}.{suffix}.json")


def _rerun_eval_dropping_distractors(true_fpath: Path, pred_fpath: Path,
                          distractor_names, out_fpath: Path,
                          score_thresh: float, test_kwcoco: str,
                          candidate_id: str, category_names) -> Path:
    """Run kwcoco eval with distractor classes pruned from both bundles.

    Distractor classes are kept in the trained model's class set (so it
    learns to discriminate them) but excluded from class-agnostic
    detection AP. Reuses the bbox-only filtered kwcocos (cheap; we
    already wrote them). Writes a sibling metrics json; the original
    ``detect_metrics.json`` stays on disk with full per-class numbers
    (including distractor AP) as a diagnostic.
    """
    import json as _json
    import kwcoco
    from kwcoco_detector_kit._provenance import provenance_dict

    distractor_set = {n.strip() for n in distractor_names if n.strip()}
    if not distractor_set:
        return out_fpath

    def _prune(src_fpath, suffix):
        src = kwcoco.CocoDataset.coerce(str(src_fpath))
        cat_ids_to_drop = {c["id"] for c in src.dataset["categories"]
                           if c["name"] in distractor_set}
        if not cat_ids_to_drop:
            return src_fpath
        keep = [a for a in src.dataset["annotations"]
                if a.get("category_id") not in cat_ids_to_drop]
        src.dataset["annotations"] = keep
        src._build_index()
        dst = src_fpath.with_name(src_fpath.stem.replace(".kwcoco", "")
                                  + f".sans_{suffix}.kwcoco.zip")
        src._update_fpath(str(dst))
        src.dump()
        return dst

    suffix = "_".join(sorted(distractor_set))
    true_pruned = _prune(true_fpath, suffix)
    pred_pruned = _prune(pred_fpath, suffix)

    cmd = [
        sys.executable, "-m", "kwcoco", "eval",
        "--true_dataset", str(true_pruned),
        "--pred_dataset", str(pred_pruned),
        "--out_dpath", str(out_fpath.parent),
        "--out_fpath", str(out_fpath),
        "--draw", "False",
        "--iou_thresh", "0.5",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0 and not out_fpath.exists():
        raise subprocess.CalledProcessError(result.returncode, cmd)

    try:
        m = _json.loads(out_fpath.read_text())
        m.setdefault("provenance", provenance_dict())
        m.setdefault("eval_inputs", {
            "test_kwcoco": str(test_kwcoco),
            "score_thresh": float(score_thresh),
            "candidate_id": str(candidate_id),
            "category_names": list(category_names),
            "distractor_classes": sorted(distractor_set),
        })
        out_fpath.write_text(_json.dumps(m, indent=2))
    except Exception as ex:
        print(f"  warn: failed to stamp provenance into {out_fpath}: {ex}")

    return out_fpath


@profile
def run_kwcoco_eval(
    *,
    trainer,
    workdir: Path,
    test_kwcoco: str,
    kcd_root: Path,
    candidate_id: str,
    category_names: Sequence[str],
    score_thresh: float = 0.001,
    force: bool = False,
    distractor_classes=None,
    tiled_eval: bool = False,
    tiled_window=None,
    tiled_overlap: float = 0.25,
    tiled_nms_thresh: float = 0.5,
    tiled_keep_full: bool = True,
    tiled_batch: int = 64,
    tiled_max_dets=None,
    tiled_pre_nms_score_thresh=None,
    tiled_pre_nms_topk=None,
    tiled_per_window_nms: bool = True,
    eval_nms_workers: int = 0,
    read_workers: int = 4,
    device: str = "cpu",
) -> Path:
    """Score every image in `test_kwcoco` with the trained model; eval.

    ``category_names`` must be the same ordered list passed to training so
    the predictor's class indices map back to the correct kwcoco category
    names. ``kwcoco eval`` matches true/pred categories by name, so the
    pred bundle is built with the same names as the trainer used.

    ``score_thresh`` defaults to 0.001 so the COCO AP integral sees the
    full precision-recall curve. Setting this above ~0.01 caps recall
    and artificially deflates AP (the prior 0.30 default cost ~0.07-0.10
    AP on shitspotter pico@416 vs. matching v4's evaluation).
    """
    import kwcoco

    if isinstance(category_names, str):
        raise TypeError(
            "category_names must be a sequence of names, not a single string"
        )
    category_names = list(category_names)
    if not category_names:
        raise ValueError("category_names must contain at least one name")

    workdir = Path(workdir)
    eval_root = Path(kcd_root) / "eval" / candidate_id
    eval_root.mkdir(parents=True, exist_ok=True)
    eval_inner = eval_root / "eval"
    eval_inner.mkdir(parents=True, exist_ok=True)
    metrics_fpath = eval_inner / "detect_metrics.json"

    if metrics_fpath.exists() and not bool(force):
        print(f"  reusing existing eval metrics: {metrics_fpath}")
        return metrics_fpath

    predictor = trainer.build_predictor(workdir, device=str(device))

    if bool(tiled_eval):
        # Windowed inference: slide native-resolution windows over each full
        # image and merge, instead of resizing the whole image to the model
        # input. Closes the train/eval resolution gap for small objects (see
        # eval/tiled_predictor.py). The window defaults to the model's
        # eval_spatial_size (= training tile size).
        from kwcoco_detector_kit.eval.tiled_predictor import TiledPredictor
        window = tiled_window
        if window is not None and not (isinstance(window, (list, tuple)) and len(window) == 2):
            window = (int(window), int(window))
        # Default the per-window score floor to the eval's own score_thresh
        # (lossless: the eval drops < score_thresh anyway, this just applies
        # it earlier to shrink the merge).
        pre_floor = (float(tiled_pre_nms_score_thresh)
                     if tiled_pre_nms_score_thresh is not None
                     else float(score_thresh))
        predictor = TiledPredictor(
            predictor,
            window=window,
            overlap=float(tiled_overlap),
            nms_thresh=float(tiled_nms_thresh),
            keep_full=bool(tiled_keep_full),
            batch_size=int(tiled_batch),
            max_dets=(int(tiled_max_dets) if tiled_max_dets else None),
            pre_nms_score_thresh=pre_floor,
            pre_nms_topk=(int(tiled_pre_nms_topk) if tiled_pre_nms_topk else None),
            per_window_nms=bool(tiled_per_window_nms),
        )
        print(
            f"  eval: TILED inference window={predictor._window} "
            f"overlap={tiled_overlap} nms={tiled_nms_thresh} "
            f"keep_full={tiled_keep_full} batch={tiled_batch} "
            f"max_dets={predictor._max_dets} | pre-merge: "
            f"score>={predictor._pre_nms_score_thresh} "
            f"per_window_nms={predictor._per_window_nms} "
            f"topk={predictor._pre_nms_topk}"
        )

    true = kwcoco.CocoDataset.coerce(str(test_kwcoco))
    pred = kwcoco.CocoDataset()
    pred.fpath = str(eval_root / "pred_boxes.kwcoco.zip")
    # Predictor labels are 0-indexed in the order training used; map each
    # label -> the kwcoco category_id we register here in the same order.
    label_to_cat_id = [pred.add_category(name=name) for name in category_names]
    n_labels = len(label_to_cat_id)
    # Copy image rows but rewrite file_name to the absolute on-disk path so
    # the eval subprocess can reroot regardless of where pred_boxes.kwcoco.zip
    # lives. Without this, kwcoco reroots against the pred bundle's dir and
    # the relative file_name="raw_assets/foo.jpg" points at a nonexistent
    # path under the pred bundle's parent.
    for img in true.images().objs:
        new = {k: v for k, v in img.items() if k != "id"}
        try:
            abs_fpath = true.get_image_fpath(img["id"])
            new["file_name"] = str(abs_fpath)
        except Exception:
            pass
        pred.add_image(**new, id=img["id"])

    dropped_unknown_label = 0
    import ubelt as ub
    _gids = list(true.images())

    # Decode is HDD/JPEG-bound and leaves the GPU idle between inference
    # spikes. Prefetch upcoming images on worker threads (libjpeg/gdal
    # release the GIL during decode) so the next image is ready when the
    # current one finishes scoring — fills the idle gaps without changing
    # scoring order. read_workers<=0 disables (sequential read).
    def _read(_gid):
        try:
            return true.coco_image(_gid).imdelay().finalize()
        except Exception as ex:  # surfaced in the loop, same as before
            return ex

    # ProgIter handles cadence/rate/eta. verbose=3 -> newline-per-update
    # (clearline off) so the stream tees/logs cleanly instead of rewriting
    # one line. Tiled eval can be seconds/image, so a steady heartbeat
    # confirms liveness.
    import time as _time

    def _add_dets(_gid, detections):
        """Add one image's detections to the pred bundle (main thread only —
        kwcoco add_annotation isn't thread-safe)."""
        nonlocal dropped_unknown_label
        for det in detections:
            score = float(det.get("score", 0.0))
            if score < float(score_thresh):
                continue
            label = int(det.get("label", 0))
            if label < 0 or label >= n_labels:
                dropped_unknown_label += 1
                continue
            x1, y1, x2, y2 = det["bbox_xyxy"]
            pred.add_annotation(
                image_id=_gid, category_id=label_to_cat_id[label],
                bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                score=score,
            )

    # Pipeline GPU inference with CPU NMS when asked and possible: profiling
    # showed GPU (_infer_windows) and the NMS merge are comparable serial
    # costs, so overlapping image N's NMS with image N+1's inference ~halves
    # the compute. batched_nms releases the GIL, so consumer threads run truly
    # parallel to the GPU-dispatching main thread. Annotation building stays
    # on the main thread. Falls back to serial when eval_nms_workers<=0 or the
    # predictor isn't a windowed one.
    _pipeline = (
        bool(tiled_eval) and int(eval_nms_workers) > 0
        and hasattr(predictor, "_infer_windows")
        and hasattr(predictor, "_merge_and_nms")
    )
    _t_decode = _t_predict = _t_annot = 0.0
    _loop_t0 = _time.perf_counter()
    _iter = _iter_prefetched(_gids, _read, int(read_workers))
    _prog = ub.ProgIter(_iter, total=len(_gids), desc="eval: scoring", verbose=3)

    if _pipeline:
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        print(f"  eval: pipelined NMS ({eval_nms_workers} consumer threads)")
        _max_inflight = 2 * int(eval_nms_workers) + 2
        pending = deque()  # (gid, future)
        with ThreadPoolExecutor(max_workers=int(eval_nms_workers)) as _pool:
            def _drain(block):
                while pending and (block or pending[0][1].done()):
                    g, fut = pending.popleft()
                    _add_dets(g, fut.result())
            for gid, arr in _prog:
                if isinstance(arr, BaseException):
                    print(f"  eval: failed to read gid {gid}: {arr}")
                    continue
                H, W = arr.shape[:2]
                raw = predictor._infer_windows(arr, (W, H))  # GPU, main thread
                if raw.get("deferred") is not None:
                    _add_dets(gid, raw["deferred"])
                else:
                    pending.append((gid, _pool.submit(predictor._merge_and_nms, raw)))
                _drain(block=False)
                while len(pending) >= _max_inflight:  # backpressure
                    g, fut = pending.popleft()
                    _add_dets(g, fut.result())
            _drain(block=True)
    else:
        _mark = _time.perf_counter()
        for gid, arr in _prog:
            _t_decode += _time.perf_counter() - _mark
            if isinstance(arr, BaseException):
                print(f"  eval: failed to read gid {gid}: {arr}")
                _mark = _time.perf_counter()
                continue
            H, W = arr.shape[:2]
            _tp = _time.perf_counter()
            detections = predictor.predict_image(arr, (W, H))
            _t_predict += _time.perf_counter() - _tp
            _ta = _time.perf_counter()
            _add_dets(gid, detections)
            _t_annot += _time.perf_counter() - _ta
            _mark = _time.perf_counter()

    _loop_total = _time.perf_counter() - _loop_t0
    _n = max(1, len(_gids))
    print(
        f"  eval timing: {_loop_total:.0f}s total over {len(_gids)} imgs "
        f"({1000 * _loop_total / _n:.0f} ms/img) | "
        f"decode_wait {_t_decode:.0f}s, predict {_t_predict:.0f}s, "
        f"annot {_t_annot:.0f}s"
    )
    if hasattr(predictor, "t_infer") and hasattr(predictor, "t_nms"):
        print(
            f"  eval timing (tiled): window_infer {predictor.t_infer:.0f}s, "
            f"nms_merge {predictor.t_nms:.0f}s "
            f"(the rest of 'predict' is window cropping + box offsetting)"
        )
    if dropped_unknown_label:
        print(
            f"  eval: dropped {dropped_unknown_label} detections with "
            f"label outside [0, {n_labels})"
        )
    # The phases after the scoring loop are silent and can be slow — tiled
    # eval writes many detections/image, so the dump + bbox-filter copies
    # are O(n_predictions). Announce each with a timer so "scoring 100%"
    # isn't followed by a long unexplained stall.
    import time as _t2
    _npred = pred.n_annots
    print(f"  eval: writing {_npred} predictions -> {Path(pred.fpath).name} "
          f"(large for tiled eval; this can take a while) ...", flush=True)
    _m = _t2.perf_counter()
    pred.dump()
    print(f"  eval: wrote predictions in {_t2.perf_counter() - _m:.0f}s", flush=True)

    print("  eval: filtering to bbox-only detections (true + pred) ...", flush=True)
    _m = _t2.perf_counter()
    true_filtered, true_kept, true_dropped = filter_bbox_only_kwcoco(
        test_kwcoco, eval_root / "true_bbox_only.kwcoco.zip")
    pred_filtered, pred_kept, pred_dropped = filter_bbox_only_kwcoco(
        pred.fpath, eval_root / "pred_boxes_bbox_only.kwcoco.zip")
    print(f"  eval: bbox filter done in {_t2.perf_counter() - _m:.0f}s", flush=True)
    if true_dropped or pred_dropped:
        print(
            "  eval bbox filter: "
            f"true kept={true_kept} dropped={true_dropped}; "
            f"pred kept={pred_kept} dropped={pred_dropped}"
        )

    print(f"  eval: computing detection metrics over {_npred} predictions "
          f"(kwcoco eval: assign + AP) ...", flush=True)

    cmd = [
        sys.executable, "-m", "kwcoco", "eval",
        "--true_dataset", str(true_filtered),
        "--pred_dataset", str(pred_filtered),
        "--out_dpath", str(eval_inner),
        "--out_fpath", str(metrics_fpath),
        "--draw", "False",
        "--iou_thresh", "0.5",
    ]
    # The confusion sidecar pass inside kwcoco eval can raise a reroot
    # exception in some asset layouts, but the metrics JSON is written
    # before that step. Tolerate a non-zero subprocess exit when the
    # metrics file landed on disk anyway — same recovery pattern as the
    # ONNX export's onnxsim crash handling.
    result = subprocess.run(cmd)
    if result.returncode != 0 and not metrics_fpath.exists():
        raise subprocess.CalledProcessError(result.returncode, cmd)
    if result.returncode != 0:
        print(
            f"  kwcoco eval exited {result.returncode} but {metrics_fpath} "
            "is present — recovering metrics."
        )

    # Stamp provenance + test-bundle fingerprint into the metrics file so
    # the eval is self-describing. Don't disturb the existing schema;
    # additional top-level keys are tolerated by every downstream reader.
    try:
        import json as _json
        from kwcoco_detector_kit._provenance import provenance_dict
        m = _json.loads(metrics_fpath.read_text())
        m.setdefault("provenance", provenance_dict())
        m.setdefault("eval_inputs", {
            "test_kwcoco": str(test_kwcoco),
            "score_thresh": float(score_thresh),
            "candidate_id": str(candidate_id),
            "category_names": list(category_names),
        })
        metrics_fpath.write_text(_json.dumps(m, indent=2))
    except Exception as ex:
        print(f"  warn: failed to stamp provenance into {metrics_fpath}: {ex}")

    print(f"  wrote {metrics_fpath}")

    # Distractor pass: classes the model learned to discriminate but that
    # the mission treats as non-targets are pruned from both GT and pred,
    # the scorer is rerun, and a sidecar metrics file is written.
    # Eligibility's selection key prefers this sidecar when present, so
    # model selection runs on the mission metric automatically.
    if distractor_classes:
        sidecar_fpath = _distractor_sidecar_fpath(metrics_fpath, distractor_classes)
        if sidecar_fpath.exists() and not bool(force):
            print(f"  reusing existing distractor-pruned metrics: {sidecar_fpath}")
        else:
            print(
                f"  running second eval pass with distractor classes pruned: "
                f"{sorted(distractor_classes)}"
            )
            try:
                _rerun_eval_dropping_distractors(
                    true_fpath=true_filtered,
                    pred_fpath=pred_filtered,
                    distractor_names=distractor_classes,
                    out_fpath=sidecar_fpath,
                    score_thresh=score_thresh,
                    test_kwcoco=test_kwcoco,
                    candidate_id=candidate_id,
                    category_names=category_names,
                )
                print(f"  wrote {sidecar_fpath}")
            except Exception as ex:
                print(
                    f"  warn: distractor-pruned eval failed: {ex}. "
                    "Eligibility will fall back to the full metrics."
                )

    return metrics_fpath
