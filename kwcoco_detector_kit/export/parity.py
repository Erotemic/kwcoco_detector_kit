"""Postprocessed torch <-> ONNX parity guard.

Parity is checked at the detector-record boundary, not merely on raw graph
outputs.  This verifies preprocessing, class layout, box scaling, score order,
and segmentation mask decoding.  A real kwcoco source can provide representative
windows; synthetic input remains available for CPU CI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import kwconf



def _sha256_file(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _representative_images(src, input_hw, num_samples):
    import numpy as np

    H, W = map(int, input_hw)
    if src is None:
        rng = np.random.RandomState(0)
        return [(rng.rand(H, W, 3) * 255).astype(np.uint8)]

    import kwcoco
    from kwcoco_detector_kit.predictors.source_window import _coerce_rgb

    dset = kwcoco.CocoDataset.coerce(str(src))
    images = []
    for gid in list(dset.images())[:max(1, int(num_samples))]:
        arr = _coerce_rgb(dset.coco_image(gid).imdelay().finalize())
        h, w = arr.shape[:2]
        if h > H or w > W:
            crop_h = min(H, h)
            crop_w = min(W, w)
            y0 = max(0, (h - crop_h) // 2)
            x0 = max(0, (w - crop_w) // 2)
            arr = arr[y0:y0 + crop_h, x0:x0 + crop_w]
        images.append(arr)
    if not images:
        raise ValueError(f"parity source has no images: {src}")
    return images


def _compare_records(torch_records, onnx_records, *, rtol, atol, mask_iou_thresh):
    import numpy as np

    result = {
        "count_torch": len(torch_records),
        "count_onnx": len(onnx_records),
        "labels_equal": False,
        "max_abs_diff_scores": float("inf"),
        "max_abs_diff_boxes": float("inf"),
        "min_mask_iou": None,
        "supports_masks": False,
    }
    if len(torch_records) != len(onnx_records):
        return False, result
    if not torch_records:
        result.update({
            "labels_equal": True,
            "max_abs_diff_scores": 0.0,
            "max_abs_diff_boxes": 0.0,
        })
        return True, result

    t_labels = np.asarray([r["label"] for r in torch_records], dtype=np.int64)
    o_labels = np.asarray([r["label"] for r in onnx_records], dtype=np.int64)
    t_scores = np.asarray([r["score"] for r in torch_records], dtype=np.float64)
    o_scores = np.asarray([r["score"] for r in onnx_records], dtype=np.float64)
    t_boxes = np.asarray([r["bbox_xyxy"] for r in torch_records], dtype=np.float64)
    o_boxes = np.asarray([r["bbox_xyxy"] for r in onnx_records], dtype=np.float64)

    labels_equal = bool(np.array_equal(t_labels, o_labels))
    score_ok = bool(np.allclose(t_scores, o_scores, rtol=rtol, atol=atol))
    box_ok = bool(np.allclose(t_boxes, o_boxes, rtol=rtol, atol=atol))
    result.update({
        "labels_equal": labels_equal,
        "max_abs_diff_scores": float(np.max(np.abs(t_scores - o_scores))),
        "max_abs_diff_boxes": float(np.max(np.abs(t_boxes - o_boxes))),
    })

    t_has_masks = all("mask" in r for r in torch_records)
    o_has_masks = all("mask" in r for r in onnx_records)
    mask_ok = t_has_masks == o_has_masks
    if t_has_masks and o_has_masks:
        ious = []
        for t_rec, o_rec in zip(torch_records, onnx_records):
            tm = np.asarray(t_rec["mask"], dtype=bool)
            om = np.asarray(o_rec["mask"], dtype=bool)
            if tm.shape != om.shape:
                ious.append(0.0)
                continue
            union = np.logical_or(tm, om).sum()
            inter = np.logical_and(tm, om).sum()
            ious.append(1.0 if union == 0 else float(inter / union))
        result["min_mask_iou"] = min(ious, default=1.0)
        result["supports_masks"] = True
        mask_ok = result["min_mask_iou"] >= float(mask_iou_thresh)
    elif not t_has_masks and not o_has_masks:
        result["supports_masks"] = False

    return bool(labels_equal and score_ok and box_ok and mask_ok), result


def check_parity(
    *,
    trainer,
    workdir: Path,
    onnx_fpath: Path,
    input_hw: Tuple[int, int],
    src=None,
    num_samples: int = 2,
    device: str = "cpu",
    score_thresh=None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    mask_iou_thresh: float = 0.99,
) -> dict:
    """Compare packaged predictor semantics on representative image windows."""
    workdir = Path(workdir)
    policy = json.loads((workdir / "policy.json").read_text())
    torch_predictor = trainer.build_predictor(workdir, device=str(device))
    policy_floor = float(policy.get("predict_score_floor", 0.0))
    score_floor = policy_floor if score_thresh is None else max(policy_floor, float(score_thresh))
    set_score_thresh = getattr(torch_predictor, "set_score_thresh", None)
    if callable(set_score_thresh):
        set_score_thresh(score_floor)
    category_names = list(policy.get("category_names") or [])

    if trainer.name == "rfdetr":
        from kwcoco_detector_kit.predictors.rfdetr_onnx import RFDETROnnxPredictor
        onnx_predictor = RFDETROnnxPredictor(
            onnx_fpath,
            device=str(device),
            score_thresh=score_floor,
            category_names=category_names,
        )
        contract = "rfdetr_raw_seg_v1"
    else:
        from kwcoco_detector_kit.predictors.onnx import OnnxPredictor
        onnx_predictor = OnnxPredictor(
            onnx_fpath,
            device=str(device),
            score_thresh=score_floor,
        )
        contract = "kit_processed_detection_v1"

    cases = []
    all_ok = True
    supports_masks = False
    images = _representative_images(src, input_hw, num_samples)
    for idx, image in enumerate(images):
        h, w = map(int, image.shape[:2])
        orig_size = (w, h)
        t_records = list(torch_predictor.predict_image(image, orig_size))
        o_records = list(onnx_predictor.predict_image(image, orig_size))
        ok, detail = _compare_records(
            t_records,
            o_records,
            rtol=float(rtol),
            atol=float(atol),
            mask_iou_thresh=float(mask_iou_thresh),
        )
        detail["sample_index"] = idx
        detail["shape_hw"] = [h, w]
        cases.append(detail)
        all_ok &= ok
        supports_masks |= bool(detail.get("supports_masks"))

    return {
        "schema": "kwcoco_detector_kit.onnx_parity.v2",
        "ok": bool(all_ok),
        "trainer": trainer.name,
        "contract": contract,
        "onnx_name": Path(onnx_fpath).name,
        "onnx_sha256": _sha256_file(Path(onnx_fpath)),
        "supports_masks": bool(supports_masks),
        "score_thresh": float(score_floor),
        "source": str(src) if src is not None else "synthetic",
        "num_samples": len(cases),
        "rtol": float(rtol),
        "atol": float(atol),
        "mask_iou_thresh": float(mask_iou_thresh),
        "cases": cases,
    }


class ParityConfig(kwconf.Config):
    """Check torch <-> ONNX prediction parity for a trained workdir."""

    workdir = kwconf.Value(None, position=1, required=True)
    src = kwconf.Value(None, help="optional representative source kwcoco")
    num_samples = kwconf.Value(2, parser=int)
    device = kwconf.Value("cpu", help="inference device, e.g. cpu or cuda:0")
    score_thresh = kwconf.Value(None, parser=float, help="parity score floor; policy floor when omitted")
    rtol = kwconf.Value(1e-3, parser=float)
    atol = kwconf.Value(1e-3, parser=float)
    mask_iou_thresh = kwconf.Value(0.99, parser=float)

    @classmethod
    def main(cls, argv=1, **kwargs):
        import kwcoco_detector_kit.trainers  # noqa: F401
        from kwcoco_detector_kit.trainers._registry import get_trainer

        config = cls.cli(argv=argv, data=kwargs, strict=True)
        workdir = Path(str(config.workdir)).expanduser().resolve()
        policy = json.loads((workdir / "policy.json").read_text())
        variant = str(policy.get("variant") or "")
        trainer_name = "rfdetr" if variant.startswith("seg_") else variant.split("_")[0]
        if not trainer_name:
            trainer_name = "deimv2"
        trainer = get_trainer(trainer_name)
        input_hw = tuple(policy.get("input_hw") or [
            policy.get("export_input_h", 640), policy.get("export_input_w", 640)
        ])
        export_dpath = workdir / "export"
        onnx_files = sorted(export_dpath.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"no .onnx found in {export_dpath}")
        if len(onnx_files) != 1:
            raise RuntimeError(
                "multiple ONNX exports exist; parity refuses to guess which graph "
                "is current: " + ", ".join(p.name for p in onnx_files)
            )
        result = check_parity(
            trainer=trainer,
            workdir=workdir,
            onnx_fpath=onnx_files[0],
            input_hw=input_hw,
            src=config.src,
            num_samples=int(config.num_samples),
            device=str(config.device),
            score_thresh=config.score_thresh,
            rtol=float(config.rtol),
            atol=float(config.atol),
            mask_iou_thresh=float(config.mask_iou_thresh),
        )
        parity_fpath = export_dpath / "parity.json"
        parity_fpath.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"[parity] {'PASS' if result['ok'] else 'FAIL'} -> {parity_fpath}")
        if not result["ok"]:
            raise SystemExit(1)
        return result


def run(config):
    return ParityConfig.main(argv=False, **{k: v for k, v in config.items()})


__cli__ = ParityConfig
