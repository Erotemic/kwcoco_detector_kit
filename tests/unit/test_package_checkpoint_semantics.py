from __future__ import annotations

import hashlib
import json
import shutil


def test_rfdetr_package_preserves_canonical_checkpoint_name(tmp_path):
    import kwcoco_detector_kit.trainers  # noqa: F401
    from kwcoco_detector_kit.export.package import build_model_package, open_package, materialize_workdir
    from kwcoco_detector_kit.trainers._registry import get_trainer

    workdir = tmp_path / "run"
    (workdir / "generated_configs").mkdir(parents=True)
    best = workdir / "checkpoint_best_total.pth"
    best.write_bytes(b"best-total")
    (workdir / "checkpoint_best_ema.pth").write_bytes(b"best-ema")
    policy = {
        "variant": "seg_2xlarge",
        "input_hw": [768, 768],
        "num_classes": 1,
        "category_names": ["poop"],
        "supports_masks": True,
    }
    (workdir / "policy.json").write_text(json.dumps(policy))
    (workdir / "generated_configs" / "rfdetr_train.json").write_text(json.dumps({
        "trainer": "rfdetr", "variant": "seg_2xlarge", "policy": policy,
    }))
    package = tmp_path / "model.zip"
    build_model_package(
        workdir=workdir, out=package, trainer="rfdetr", category_names=["poop"]
    )
    # The archive must be sufficient after the source training workdir is gone.
    shutil.rmtree(workdir)
    with open_package(package) as (root, manifest):
        assert manifest["schema"] == "kwcoco_detector_kit.package.v2"
        assert manifest["artifacts"]["checkpoint"] == "weights/checkpoint_best_total.pth"
        assert manifest["checkpoint"]["sha256"] == hashlib.sha256(b"best-total").hexdigest()
        assert manifest["capabilities"]["supports_masks"] is True
        assert manifest["category_names"] == ["poop"]
        assert manifest["preprocess"]["resize"] == "bilinear_half_pixel_antialias_false"
        assert manifest["inference"]["mode"] == "windowed"
        assert manifest["inference"]["window"] == [768, 768]
        materialized = materialize_workdir(root, manifest, tmp_path / "materialized")
        assert get_trainer("rfdetr").find_checkpoint(materialized).name == "checkpoint_best_total.pth"
