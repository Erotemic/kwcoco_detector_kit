"""Regression guard on the faster-coco-eval API contract DEIM relies on.

faster-coco-eval 1.7.x regressed ``COCO.loadRes`` / ``loadAnns``: DEIMv2's
in-loop evaluator (``engine/data/dataset/coco_eval.py`` ``CocoEvaluator.update``)
crashes mid-training with::

    TypeError: 'int' object does not support the context manager protocol

inside ``loadAnns(getAnnIds(...))``. The 1.6 series is fine. We pin
``faster_coco_eval>=1.6.7,<1.7`` in pyproject's ``[deimv2]`` extra and DEIM's
requirements.txt.

Two guards here, no torch / no GPU / no DEIMv2 package import:

- ``test_faster_coco_eval_version_is_pinned_below_1_7`` is THE regression
  guard: it asserts the installed version honors the pin. The exact in-loop
  crash only surfaces with DEIM's full data/model pipeline (real predictions +
  convert_to_coco_api), so a cheap unit test can't reproduce it directly; the
  version assertion is what reliably fails a rebuild that floated the dep.

- ``test_faster_coco_eval_loadres_evaluate_contract`` is a smoke of the API
  *shape* DEIM depends on (``loadRes(results) -> evaluate()``). It passes on
  both 1.6.x and 1.7.2 — it documents/exercises the call path but is not the
  version trigger. Keep it as a contract sanity check.
"""
from __future__ import annotations

import pytest


def _minimal_coco_gt():
    """A 1-category, 1-image, 1-annotation COCO dict — the smallest input
    that exercises loadAnns/getAnnIds/loadRes the way DEIM's eval does."""
    return {
        "images": [{"id": 1, "width": 64, "height": 64, "file_name": "x.jpg"}],
        "categories": [{"id": 1, "name": "poop"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [8.0, 8.0, 16.0, 16.0],
                "area": 256.0,
                "iscrowd": 0,
            }
        ],
    }


def test_faster_coco_eval_loadres_evaluate_contract():
    """loadRes + evaluate must run — the call shape DEIM makes per batch.

    Mirrors CocoEvaluator.update: build COCOeval_faster, loadRes a detection
    list, set cocoDt/imgIds, evaluate(). This passes on both 1.6.x and 1.7.2
    (the in-loop regression needs DEIM's full pipeline to trigger), so it is a
    contract sanity check, not the version trigger — see the version test below.
    """
    faster_coco_eval = pytest.importorskip("faster_coco_eval")
    from faster_coco_eval import COCO, COCOeval_faster

    coco_gt = COCO(_minimal_coco_gt())

    # A single detection on image 1 (the format DEIM's `prepare` emits:
    # one dict per detection with xywh bbox + score).
    results = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [8.0, 8.0, 16.0, 16.0],
            "score": 0.9,
        }
    ]

    coco_dt = coco_gt.loadRes(results)  # <-- the regressed call path
    coco_eval = COCOeval_faster(
        coco_gt, iouType="bbox", print_function=print, separate_eval=True
    )
    coco_eval.cocoDt = coco_dt
    coco_eval.params.imgIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # A perfect detection should score a positive AP — confirms the eval
    # actually ran end-to-end, not just that no exception was raised.
    ap = float(coco_eval.stats[0])
    assert ap > 0.0, f"expected positive AP for a perfect detection, got {ap}"


def test_faster_coco_eval_version_is_pinned_below_1_7():
    """Belt-and-suspenders: assert the installed version honors the pin.

    If this fails, a rebuild floated faster-coco-eval past the pin — fix the
    [deimv2] extra / requirements.txt rather than deleting this test.
    """
    faster_coco_eval = pytest.importorskip("faster_coco_eval")
    version = getattr(faster_coco_eval, "__version__", None)
    if version is None:  # pragma: no cover - older builds without __version__
        pytest.skip("faster_coco_eval has no __version__ attribute")

    parts = version.split(".")
    major, minor = int(parts[0]), int(parts[1])
    assert (major, minor) < (1, 7), (
        f"faster_coco_eval=={version} violates the >=1.6.7,<1.7 pin; "
        "1.7.x breaks DEIM's in-loop COCO eval (loadAnns)"
    )
