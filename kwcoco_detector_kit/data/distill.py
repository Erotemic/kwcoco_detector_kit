"""Knowledge distillation support — pseudo-label generation and soft-label scaffolding.

Two distillation modes are implemented here:

**Pseudo-label generation** (teacher → unlabeled images → weak-GT kwcoco)
    The teacher runs inference over a pool of unlabeled or negative tiles;
    high-confidence detections become pseudo ground-truth annotations.  The
    resulting kwcoco dataset can then be merged with real GT and fed into any
    student trainer via the existing tile/merge/train pipeline.  This mode
    works with *any* packaged teacher — it is just :func:`predict_kwcoco`
    with a confidence gate.

**Soft-label distillation** (trainer-side)
    Soft-label loss (KL divergence on teacher logits) is trainer-specific.
    This module provides :func:`generate_distill_policy` which writes a
    ``distill_policy.json`` into the student workdir that trainer plugins
    can read from ``policy.json`` to activate a soft-label training mode.
    Currently only DEIMv2 supports this — see ``trainers/deimv2.py``.

Typical round-loop workflow
-----------------------------
1. Train a large *teacher* model (e.g. ``opengroundingdino_swinb``).
2. Run :func:`pseudo_label_kwcoco` over the negative tile pool.
3. Merge pseudo-labels with real GT: ``kwcoco-detector-kit merge``.
4. Train a small *student* model (e.g. ``deimv2_hgnetv2_n``) on the merged set.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import scriptconfig as scfg


# ---------------------------------------------------------------------------
# Pseudo-label generation
# ---------------------------------------------------------------------------

def pseudo_label_kwcoco(
    *,
    teacher_package: str | Path,
    src: str | Path,
    dst: str | Path,
    device: str = "cpu",
    score_thresh: Optional[float] = None,
    nms_thresh: Optional[float] = None,
    min_annotations: int = 1,
    workdir: Optional[str | Path] = None,
) -> Path:
    """Run a teacher model over unlabeled images and write pseudo-label annotations.

    This is a thin wrapper around :func:`~kwcoco_detector_kit.predict.predict_kwcoco`
    that additionally:
    - Drops images with fewer than ``min_annotations`` predictions (all-negative
      images add no signal and inflate the negative class).
    - Tags every annotation with ``role="pseudo_label"`` and
      ``pseudo_label_score`` so downstream tools can filter by provenance.

    Args:
        teacher_package: Packaged teacher model (directory, archive, or
            ``package.yaml``).
        src: Source kwcoco dataset or image directory.
        dst: Destination pseudo-label kwcoco path.
        device: Torch device.
        score_thresh: Minimum score for a prediction to become a pseudo-label.
            Defaults to the teacher package's own threshold (often 0.30).
        nms_thresh: NMS IoU threshold override.
        min_annotations: Images with fewer surviving annotations are dropped
            from the output.  Set to 0 to keep all images.
        workdir: Optional persistent materialized workdir.

    Returns:
        Path to the written pseudo-label kwcoco dataset.
    """
    import kwcoco
    from kwcoco_detector_kit.predict import predict_kwcoco

    raw_dst = Path(dst).expanduser().with_suffix("") \
        .parent / (Path(dst).stem + "_raw" + Path(dst).suffix)

    predict_kwcoco(
        package=teacher_package,
        src=src,
        dst=raw_dst,
        device=device,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        workdir=workdir,
    )

    # Post-process: tag annotations and filter sparse images.
    raw = kwcoco.CocoDataset.coerce(str(raw_dst))
    out = kwcoco.CocoDataset()
    out.fpath = str(Path(dst).expanduser())
    out.fpath and Path(out.fpath).parent.mkdir(parents=True, exist_ok=True)

    # Copy categories
    for cat in raw.cats.values():
        out.add_category(**{k: v for k, v in cat.items() if k != "id"}, id=cat["id"])

    kept_images = 0
    for img in raw.images().objs:
        gid = img["id"]
        anns = raw.annots(gid=gid).objs
        if len(anns) < max(1, min_annotations):
            continue
        new_img = {k: v for k, v in img.items() if k != "id"}
        try:
            new_img["file_name"] = str(raw.get_image_fpath(gid))
        except Exception:
            pass
        out.add_image(**new_img, id=gid)
        for ann in anns:
            new_ann = {k: v for k, v in ann.items() if k not in ("id", "image_id")}
            new_ann["image_id"] = gid
            new_ann["role"] = "pseudo_label"
            new_ann["pseudo_label_score"] = float(ann.get("score", 0.0))
            out.add_annotation(**new_ann)
        kept_images += 1

    out.dump()
    print(
        f"pseudo_label_kwcoco: kept {kept_images}/{raw.n_images} images, "
        f"{out.n_annots} annotations → {out.fpath}"
    )
    return Path(out.fpath)


# ---------------------------------------------------------------------------
# Soft-label distillation policy (trainer-side hook)
# ---------------------------------------------------------------------------

def generate_distill_policy(
    *,
    teacher_package: str | Path,
    student_variant: str,
    workdir: str | Path,
    distill_alpha: float = 0.5,
    temperature: float = 4.0,
) -> Path:
    """Write a distillation policy JSON into a student trainer workdir.

    Trainer plugins that support soft-label distillation (currently DEIMv2)
    read this file from ``workdir/distill_policy.json`` at config-generation
    time and activate a KL-divergence loss against the teacher's logits.

    Args:
        teacher_package: Path to the packaged teacher model.
        student_variant: Student model variant string (e.g. ``"deimv2_hgnetv2_n"``).
        workdir: Student trainer workdir.
        distill_alpha: Weight of the distillation loss relative to the GT loss
            (1.0 = pure distillation, 0.0 = pure GT).
        temperature: Softmax temperature for label smoothing.

    Returns:
        Path to the written ``distill_policy.json``.
    """
    workdir = Path(workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    policy = {
        "distill_mode": "soft_label",
        "teacher_package": str(Path(teacher_package).expanduser().resolve()),
        "student_variant": student_variant,
        "distill_alpha": float(distill_alpha),
        "temperature": float(temperature),
    }
    out_fpath = workdir / "distill_policy.json"
    out_fpath.write_text(json.dumps(policy, indent=2))
    print(f"wrote distillation policy: {out_fpath}")
    return out_fpath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class PseudoLabelConfig(scfg.DataConfig):
    """Generate pseudo-label annotations from a teacher model over unlabeled images.

    Runs the teacher predictor over ``src``, filters by ``score_thresh``,
    and writes a kwcoco dataset with ``role=pseudo_label`` annotations.
    Images with fewer than ``min_annotations`` surviving detections are dropped.
    """

    teacher_package = scfg.Value(None, required=True, position=1,
                                 help="packaged teacher model (directory, archive, or package.yaml)")
    src = scfg.Value(None, required=True, help="source kwcoco dataset or image directory")
    dst = scfg.Value(None, required=True, help="output pseudo-label kwcoco path")
    device = scfg.Value("cpu", help="torch device")
    score_thresh = scfg.Value(None, type=float, help="override teacher score threshold")
    nms_thresh = scfg.Value(None, type=float, help="override NMS threshold")
    min_annotations = scfg.Value(1, type=int,
                                 help="drop images with fewer than this many pseudo-labels")
    workdir = scfg.Value(None, help="optional persistent materialized workdir")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        out = pseudo_label_kwcoco(
            teacher_package=config.teacher_package,
            src=config.src,
            dst=config.dst,
            device=str(config.device),
            score_thresh=config.score_thresh,
            nms_thresh=config.nms_thresh,
            min_annotations=int(config.min_annotations),
            workdir=config.workdir,
        )
        print(f"wrote pseudo-labels: {out}")
        return 0


__cli__ = PseudoLabelConfig
