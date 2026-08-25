"""The real scorer: evaluate one staged checkpoint under one binding.

``KitScorer`` is the production implementation of the worker's injected
``scorer(ckpt_fpath, binding) -> measures`` callable. It:

1. builds a ``DEIMv2Predictor`` pinned to the staged checkpoint (the
   predictor prefers the EMA weights inside a full state dict — the
   EMA-always default needs no extra plumbing);
2. applies the binding's regime: ``SlidingWindow`` wraps the predictor in
   the kit's ``TiledPredictor`` (real cross-window NMS — the probe runs
   the *same* procedure as the reported metric); ``Resize`` uses the
   predictor's native whole-image path;
3. predicts every image in the binding's dataset, scores with
   ``kwcoco eval`` (+ the distractor-pruned second pass when the
   protocol's class filter demands it);
4. parses the metrics JSON into the flat measures dict
   (``AP@0.5`` headline honoring the class-filter rule, ``AP@0.5/all``
   diagnostic, ``ap/<class>`` per class).

Outputs land under ``<workdir>/journal/scores/<fingerprint>/<ckpt_stem>/``
so every ScoreRecord's full eval artifacts remain on disk for drill-down.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from kwcoco_detector_kit.eval.protocols import Resize, SlidingWindow
from kwcoco_detector_kit.selection.config import Binding

__all__ = ["KitScorer", "measures_from_detect_metrics"]

_BLOCK_KEY = "area_range=all,iou_thresh=0.5"


def measures_from_detect_metrics(
    metrics_fpath: Path,
    sidecar_fpath: Optional[Path] = None,
) -> Dict[str, float]:
    """Flatten the kit's detect_metrics.json into the measures dict.

    When a distractor-pruned sidecar exists, its class-agnostic AP is the
    headline ``AP@0.5`` (the mission criterion) and the with-distractor
    number is kept as ``AP@0.5/all``. Per-class APs come from the main
    pass; NaNs (zero-support classes) are dropped.
    """
    def _block(fpath: Path) -> Dict:
        data = json.loads(Path(fpath).read_text())
        return data.get(_BLOCK_KEY, {})

    main = _block(metrics_fpath)
    measures: Dict[str, float] = {}

    def _ap_of(block) -> Optional[float]:
        nocls = block.get("nocls_measures") or {}
        ap = nocls.get("ap")
        if ap is None or (isinstance(ap, float) and math.isnan(ap)):
            return None
        return float(ap)

    main_ap = _ap_of(main)
    if sidecar_fpath is not None and Path(sidecar_fpath).exists():
        sidecar_ap = _ap_of(_block(sidecar_fpath))
        if sidecar_ap is not None:
            measures["AP@0.5"] = sidecar_ap
        if main_ap is not None:
            measures["AP@0.5/all"] = main_ap
    elif main_ap is not None:
        measures["AP@0.5"] = main_ap

    for cat, m in (main.get("ovr_measures") or {}).items():
        ap = (m or {}).get("ap")
        if ap is None or (isinstance(ap, float) and math.isnan(ap)):
            continue
        measures[f"ap/{cat}"] = float(ap)
    return measures


class KitScorer:

    def __init__(
        self,
        *,
        workdir: Path,
        train_workdir: Path,
        category_names: Sequence[str],
        distractor_classes: Optional[Sequence[str]] = None,
        device: str = "cuda",
        tiled_batch: int = 64,
        log=print,
    ):
        """
        Args:
            workdir: the run dir holding ``journal/`` and ``staging/``.
            train_workdir: the trained-model dir holding
                ``generated_configs/train.yml`` (predictor construction).
            category_names: ordered list == train-time class indices.
            distractor_classes: the *list* half of the class filter
                (scheme-owned); applied only when the protocol's rule says
                ``exclude_distractors``.
        """
        self.workdir = Path(workdir)
        self.train_workdir = Path(train_workdir)
        self.category_names = list(category_names)
        self.distractor_classes = list(distractor_classes or [])
        self.device = str(device)
        self.tiled_batch = int(tiled_batch)
        self.log = log
        self._predictor_cache: Dict[str, object] = {}

    # ------------------------------------------------------------------

    def _base_predictor(self, ckpt_fpath: Path):
        key = str(ckpt_fpath)
        if key not in self._predictor_cache:
            from kwcoco_detector_kit.trainers.deimv2 import DEIMv2Predictor
            cfg_fpath = self.train_workdir / "generated_configs" / "train.yml"
            if not cfg_fpath.exists():
                raise FileNotFoundError(
                    f"missing {cfg_fpath} -- the scorer needs the generated "
                    "train.yml to build a predictor for staged checkpoints"
                )
            # cache only the most recent checkpoint's predictor (epochs are
            # scored newest-last; older entries just waste GPU memory)
            self._predictor_cache.clear()
            self._predictor_cache[key] = DEIMv2Predictor(
                ckpt_fpath, cfg_fpath, device=self.device)
        return self._predictor_cache[key]

    def _regime_predictor(self, ckpt_fpath: Path, binding: Binding):
        base = self._base_predictor(ckpt_fpath)
        regime = binding.protocol.regime
        if isinstance(regime, SlidingWindow):
            from kwcoco_detector_kit.eval.tiled_predictor import TiledPredictor
            return TiledPredictor(
                base,
                window=tuple(int(v) for v in regime.window),
                overlap=float(regime.overlap),
                nms_thresh=float(regime.nms_iou),
                batch_size=self.tiled_batch,
                pre_nms_score_thresh=float(binding.protocol.score_thresh),
                # Frozen to match run_kwcoco_eval, whose default is False.
                # TiledPredictor's own default is True, so leaving it implicit
                # made selection rerank checkpoints under a DIFFERENT inference
                # procedure than the one that produced the baseline they are
                # compared against. Two numbers from two procedures are not a
                # comparison. False is chosen because it is what the existing
                # baseline evaluator already does.
                per_window_nms=False,
            )
        if isinstance(regime, Resize):
            # DEIMv2Predictor.predict_image resizes the whole image to the
            # model's eval_spatial_size internally == regime.size by
            # construction (both resolve from train_input_hw)
            return base
        raise TypeError(f"unknown regime {type(regime).__name__}")

    # ------------------------------------------------------------------

    def __call__(self, ckpt_fpath: Path, binding: Binding) -> Dict[str, float]:
        import kwcoco
        from kwcoco_detector_kit.eval.kwcoco_eval import (
            _distractor_sidecar_fpath,
            _rerun_eval_dropping_distractors,
            filter_bbox_only_kwcoco,
        )

        ckpt_fpath = Path(ckpt_fpath)
        out_root = (self.workdir / "journal" / "scores"
                    / binding.fingerprint / ckpt_fpath.stem)
        out_root.mkdir(parents=True, exist_ok=True)
        eval_inner = out_root / "eval"
        eval_inner.mkdir(exist_ok=True)
        metrics_fpath = eval_inner / "detect_metrics.json"

        exclude = (binding.protocol.class_filter.exclude_distractors
                   and bool(self.distractor_classes))
        sidecar_fpath = (
            _distractor_sidecar_fpath(metrics_fpath, self.distractor_classes)
            if exclude else None
        )

        if not metrics_fpath.exists():
            predictor = self._regime_predictor(ckpt_fpath, binding)

            true = kwcoco.CocoDataset.coerce(binding.dataset_fpath)
            pred = kwcoco.CocoDataset()
            pred.fpath = str(out_root / "pred_boxes.kwcoco.zip")
            label_to_cat_id = [
                pred.add_category(name=name) for name in self.category_names
            ]
            n_labels = len(label_to_cat_id)
            for img in true.images().objs:
                new = {k: v for k, v in img.items() if k != "id"}
                try:
                    new["file_name"] = str(true.get_image_fpath(img["id"]))
                except Exception:
                    pass
                pred.add_image(**new, id=img["id"])

            gids = list(true.images())
            for n, gid in enumerate(gids):
                arr = true.coco_image(gid).imdelay().finalize()
                H, W = arr.shape[:2]
                dets = predictor.predict_image(arr, (W, H))
                for det in dets:
                    score = float(det.get("score", 0.0))
                    if score < float(binding.protocol.score_thresh):
                        continue
                    label = int(det.get("label", 0))
                    if label < 0 or label >= n_labels:
                        continue
                    x1, y1, x2, y2 = det["bbox_xyxy"]
                    pred.add_annotation(
                        image_id=gid, category_id=label_to_cat_id[label],
                        bbox=[float(x1), float(y1),
                              float(x2 - x1), float(y2 - y1)],
                        score=score,
                    )
                if (n + 1) % 10 == 0 or n + 1 == len(gids):
                    self.log(f"  [{binding.label}/{ckpt_fpath.stem}] "
                             f"predicted {n + 1}/{len(gids)}")
            pred.dump()

            true_filt, _, _ = filter_bbox_only_kwcoco(
                binding.dataset_fpath, out_root / "true_bbox_only.kwcoco.zip")
            pred_filt, _, _ = filter_bbox_only_kwcoco(
                pred.fpath, out_root / "pred_boxes_bbox_only.kwcoco.zip")

            cmd = [
                sys.executable, "-m", "kwcoco", "eval",
                "--true_dataset", str(true_filt),
                "--pred_dataset", str(pred_filt),
                "--out_dpath", str(eval_inner),
                "--out_fpath", str(metrics_fpath),
                "--draw", "False",
                "--iou_thresh", "0.5",
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0 and not metrics_fpath.exists():
                raise subprocess.CalledProcessError(result.returncode, cmd)

            if exclude and not sidecar_fpath.exists():
                _rerun_eval_dropping_distractors(
                    true_fpath=true_filt,
                    pred_fpath=pred_filt,
                    distractor_names=self.distractor_classes,
                    out_fpath=sidecar_fpath,
                    score_thresh=binding.protocol.score_thresh,
                    test_kwcoco=binding.dataset_fpath,
                    candidate_id=ckpt_fpath.stem,
                    category_names=self.category_names,
                )

        return measures_from_detect_metrics(metrics_fpath, sidecar_fpath)
