"""Tests for data.tile — three modes + oversize_factor + invariants."""
from __future__ import annotations

from pathlib import Path

import kwcoco
import numpy as np
import pytest


def _tile_run(src, dst, **kwargs):
    from kwcoco_detector_kit.data.tile import TileConfig, run

    cfg = TileConfig.cli(
        argv=False,
        data={"src": str(src), "dst": str(dst), **kwargs},
    )
    run(cfg)
    return kwcoco.CocoDataset.coerce(str(dst))


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["full_only", "quadrant", "multiscale"])
def test_each_mode_writes_a_kwcoco_bundle(synthetic_kwcoco, tmp_path, mode):
    dst = tmp_path / f"tiled_{mode}.kwcoco.zip"
    dset = _tile_run(
        synthetic_kwcoco, dst, mode=mode, category_names="widget",
        progress=False,
        # multiscale needs at least one positive scale
        tile_size=128, source_scales="1.0", min_source_scale_long_side=32,
    )
    assert dset.n_images > 0, f"{mode}: produced no tiles"
    assert dset.fpath.endswith(".kwcoco.zip")


def test_unknown_mode_raises(synthetic_kwcoco, tmp_path):
    """Unknown mode strings fail loudly rather than silently going to a default."""
    from kwcoco_detector_kit.data.tile import TileConfig, run

    cfg = TileConfig.cli(
        argv=False,
        data={
            "src": str(synthetic_kwcoco),
            "dst": str(tmp_path / "x.kwcoco.zip"),
            "mode": "INVALID",
            "category_names": "widget",
            "progress": False,
        },
    )
    with pytest.raises(Exception):
        # kwconf's choices may reject INVALID at parse-time; otherwise
        # _run_* dispatches and falls through to the ValueError.
        run(cfg)


# ---------------------------------------------------------------------------
# Multiscale invariants — produces positives + (optionally) negatives
# ---------------------------------------------------------------------------


def test_multiscale_emits_positive_and_negative_roles(synthetic_kwcoco, tmp_path):
    dst = tmp_path / "ms.kwcoco.zip"
    dset = _tile_run(
        synthetic_kwcoco, dst, mode="multiscale",
        category_names="widget", progress=False,
        tile_size=128, source_scales="1.0,0.5",
        stride_frac=0.5, min_gt_area_frac=0.001, keep_negative=True,
    )
    roles = {img.get("tile_role") for img in dset.images().objs}
    assert "positive" in roles, "expected at least one positive tile"


def test_multiscale_keep_negative_false_drops_negatives(synthetic_kwcoco, tmp_path):
    dst = tmp_path / "ms_pos.kwcoco.zip"
    dset = _tile_run(
        synthetic_kwcoco, dst, mode="multiscale",
        category_names="widget", progress=False,
        tile_size=128, source_scales="1.0",
        stride_frac=0.5, min_gt_area_frac=0.001, keep_negative=False,
    )
    roles = {img.get("tile_role") for img in dset.images().objs}
    assert "negative" not in roles, f"keep_negative=False but roles include negative: {roles}"


# ---------------------------------------------------------------------------
# Oversize_factor — disk tile size = ceil(model_input_size * oversize_factor)
# ---------------------------------------------------------------------------


def test_oversize_factor_writes_larger_disk_tiles(synthetic_kwcoco, tmp_path):
    """oversize_factor=1.4 -> tiles emitted at ~448x448 even though model input is 320x320."""
    dst = tmp_path / "ms_oversize.kwcoco.zip"
    dset = _tile_run(
        synthetic_kwcoco, dst, mode="multiscale",
        category_names="widget", progress=False,
        tile_size=64, source_scales="1.0",
        oversize_factor=1.5,
        stride_frac=1.0, min_gt_area_frac=0.0001, keep_negative=False,
    )
    # On-disk image record width should be ~96 (= 64 * 1.5).
    assert dset.n_images > 0
    img = next(iter(dset.images().objs))
    assert int(img["width"]) == 96, (
        f"tile_size=64 * oversize 1.5 -> disk 96; got {img['width']}"
    )
    # The model_input_size metadata is 64x64 — that's the size the model
    # will see after the load-time crop.
    assert img["tile_model_input_size"] == [64, 64]


def test_full_only_produces_one_image_per_source(synthetic_kwcoco, tmp_path):
    src = kwcoco.CocoDataset.coerce(str(synthetic_kwcoco))
    n_src = src.n_images
    dst = tmp_path / "full.kwcoco.zip"
    dset = _tile_run(
        synthetic_kwcoco, dst, mode="full_only",
        category_names="widget", progress=False,
        full_dim=128,
    )
    assert dset.n_images == n_src, (
        f"full_only should emit exactly one tile per source image; got {dset.n_images} vs {n_src}"
    )
    for img in dset.images().objs:
        assert img.get("tile_role") == "full"


# ---------------------------------------------------------------------------
# Quadrant mode — NxN grid with overlap
# ---------------------------------------------------------------------------


def test_quadrant_emits_full_plus_grid_tiles(synthetic_kwcoco, tmp_path):
    src = kwcoco.CocoDataset.coerce(str(synthetic_kwcoco))
    n_src = src.n_images
    dst = tmp_path / "quad.kwcoco.zip"
    dset = _tile_run(
        synthetic_kwcoco, dst, mode="quadrant",
        category_names="widget", progress=False,
        tile_grid=2, tile_overlap=0.20, tile_output_dim=128, keep_full=True,
    )
    # Each source -> 1 full + 4 tiles = 5 emitted images per source
    assert dset.n_images == n_src * 5


def test_quadrant_keep_full_false_drops_full_view(synthetic_kwcoco, tmp_path):
    src = kwcoco.CocoDataset.coerce(str(synthetic_kwcoco))
    n_src = src.n_images
    dst = tmp_path / "quad_no_full.kwcoco.zip"
    dset = _tile_run(
        synthetic_kwcoco, dst, mode="quadrant",
        category_names="widget", progress=False,
        tile_grid=2, tile_overlap=0.20, tile_output_dim=128, keep_full=False,
    )
    # Each source -> 4 tiles = 4 emitted images per source
    assert dset.n_images == n_src * 4
    for img in dset.images().objs:
        assert img.get("tile_role") == "tile"


# ---------------------------------------------------------------------------
# Annotation bbox warping — boxes are clipped, kept_fraction filters
# ---------------------------------------------------------------------------


def test_clipped_annotation_stays_inside_tile(synthetic_kwcoco, tmp_path):
    dst = tmp_path / "ms_check_anns.kwcoco.zip"
    dset = _tile_run(
        synthetic_kwcoco, dst, mode="multiscale",
        category_names="widget", progress=False,
        tile_size=64, source_scales="1.0",
        stride_frac=0.5, min_gt_area_frac=0.001, keep_negative=False,
        min_keep_fraction=0.10,
    )
    for ann in dset.annots().objs:
        bx, by, bw, bh = ann["bbox"]
        assert bx >= 0 and by >= 0
        assert bx + bw <= 64 + 1e-3
        assert by + bh <= 64 + 1e-3


def _make_boundary_polygon_bundle(tmp_path):
    import kwimage
    import numpy as np

    bundle = tmp_path / "boundary_bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    kwimage.imwrite(str(assets / "im.png"), image)
    dset = kwcoco.CocoDataset()
    dset.fpath = str(bundle / "data.kwcoco.zip")
    cid = dset.add_category(name="widget")
    gid = dset.add_image(file_name="assets/im.png", width=128, height=128)
    # This straddles the x=64 boundary. With min_keep_fraction > 0.5,
    # neither fragment is valid positive supervision.
    polygon = {
        "exterior": [[60, 10], [68, 10], [68, 50], [60, 50], [60, 10]],
        "interiors": [],
    }
    dset.add_annotation(
        image_id=gid,
        category_id=cid,
        bbox=[60, 10, 8, 40],
        segmentation=polygon,
        area=320,
        iscrowd=0,
    )
    dset.dump()
    return Path(dset.fpath)


def test_unsuitable_intersecting_target_is_ignored_not_negative(tmp_path):
    src = _make_boundary_polygon_bundle(tmp_path)
    dst = tmp_path / "boundary_tiles.kwcoco.zip"
    dset = _tile_run(
        src,
        dst,
        mode="multiscale",
        category_names="widget",
        progress=False,
        tile_size=64,
        source_scales="1.0",
        stride_frac=1.0,
        min_keep_fraction=0.75,
        min_gt_area_frac=0.0001,
        keep_negative=True,
    )
    role_counts = dset.dataset["info"][0]["tile_role_counts"]
    assert role_counts["ignore"] == 2
    for img in dset.images().objs:
        if img["tile_role"] == "negative":
            x0, y0, x1, y1 = img["tile_extent_xyxy_in_source"]
            # No emitted negative may overlap the known target rectangle.
            overlaps = min(x1, 68) > max(x0, 60) and min(y1, 50) > max(y0, 10)
            assert not overlaps, img
            assert img["tile_num_intersecting_anns"] == 0


def test_multiscale_preserves_segmentation_and_recomputes_geometry(tmp_path):
    src = _make_boundary_polygon_bundle(tmp_path)
    dst = tmp_path / "seg_tiles.kwcoco.zip"
    dset = _tile_run(
        src,
        dst,
        mode="multiscale",
        category_names="widget",
        progress=False,
        tile_size=128,
        source_scales="1.0",
        stride_frac=1.0,
        min_keep_fraction=0.1,
        min_gt_area_frac=0.0001,
        keep_negative=True,
    )
    assert dset.n_annots == 1
    ann = dset.dataset["annotations"][0]
    assert ann.get("segmentation") is not None
    assert np.allclose(ann["bbox"], [60, 10, 8, 40])
    assert np.isclose(ann["area"], 320)


def test_realized_resize_uses_distinct_xy_scales(tmp_path):
    src = _make_boundary_polygon_bundle(tmp_path)
    import kwcoco
    dset = kwcoco.CocoDataset.coerce(src)
    dset.imgs[1]["width"] = 129
    dset.imgs[1]["height"] = 127
    # Keep the actual asset and manifest dimensions aligned.
    import kwimage
    image_path = dset.get_image_fpath(1)
    kwimage.imwrite(image_path, np.zeros((127, 129, 3), dtype=np.uint8))
    dset.dump()
    out = _tile_run(
        src, tmp_path / "odd_scale.kwcoco.zip", mode="multiscale",
        category_names="widget", progress=False, tile_size=128,
        source_scales="0.333", stride_frac=1.0,
        min_source_scale_long_side=1,
        min_keep_fraction=0.1, min_gt_area_frac=0.0001,
        keep_negative=True,
    )
    image = out.images().objs[0]
    sx, sy = image["tile_actual_scale_xy"]
    assert sx != sy
    assert np.isclose(sx, round(129 * .333) / 129)
    assert np.isclose(sy, round(127 * .333) / 127)

    import kwimage
    source_to_scaled = kwimage.Affine.scale((sx, sy))
    expected_poly = kwimage.Polygon(
        exterior=[[60, 10], [68, 10], [68, 50], [60, 50], [60, 10]],
    ).warp(source_to_scaled)
    ann = out.annots(gid=image["id"]).objs[0]
    got_poly = kwimage.MultiPolygon.coerce(ann["segmentation"]).data[0]
    got_vertices = sorted(map(tuple, got_poly.exterior.data[:-1]))
    expected_vertices = sorted(map(tuple, expected_poly.exterior.data[:-1]))
    assert np.allclose(got_vertices, expected_vertices)
    assert np.allclose(ann["bbox"], expected_poly.box().to_xywh().data)

    scaled_crop = kwimage.Boxes([[0, 0, 128, 128]], "ltrb")
    expected_source_extent = scaled_crop.warp(source_to_scaled.inv()).to_ltrb().data[0]
    expected_source_extent = [int(round(v)) for v in expected_source_extent]
    assert image["tile_extent_xyxy_in_source"] == expected_source_extent


def test_negative_keep_fraction_is_deterministic_and_prewrite(synthetic_kwcoco, tmp_path):
    common = {
        "mode": "multiscale",
        "category_names": "widget",
        "progress": False,
        "tile_size": 64,
        "source_scales": "1.0",
        "stride_frac": 0.5,
        "min_gt_area_frac": 0.0001,
        "keep_negative": True,
        "negative_keep_fraction": 0.35,
        "seed": 991,
    }
    first = _tile_run(synthetic_kwcoco, tmp_path / "sample1.kwcoco.zip", **common)
    second = _tile_run(synthetic_kwcoco, tmp_path / "sample2.kwcoco.zip", **common)
    first_neg = [
        (img["tile_source_gid"], img["tile_scale_name"], img["tile_extent_xyxy_in_source"])
        for img in first.images().objs if img["tile_role"] == "negative"
    ]
    second_neg = [
        (img["tile_source_gid"], img["tile_scale_name"], img["tile_extent_xyxy_in_source"])
        for img in second.images().objs if img["tile_role"] == "negative"
    ]
    assert first_neg == second_neg
    counts = first.dataset["info"][0]["tile_role_counts"]
    assert first.dataset["info"][0]["source_manifest_sha256"]
    assert first.dataset["info"][0]["source_dataset_fingerprint"]
    assert counts["negative"] == len(first_neg)
    assert counts["dropped_negative"] > 0
    assert counts["negative"] > 0

    none = _tile_run(
        synthetic_kwcoco,
        tmp_path / "sample_none.kwcoco.zip",
        **{**common, "negative_keep_fraction": 0.0},
    )
    assert not [img for img in none.images().objs if img["tile_role"] == "negative"]


def test_materialization_cache_is_reused_across_manifests(synthetic_kwcoco, tmp_path):
    cache_dpath = tmp_path / "tile_cache"
    kwargs = {
        "mode": "multiscale",
        "category_names": "widget",
        "progress": False,
        "tile_size": 128,
        "source_scales": "1.0",
        "min_source_scale_long_side": 32,
        "cache_dpath": str(cache_dpath),
    }
    first = _tile_run(synthetic_kwcoco, tmp_path / "round0.kwcoco.zip", **kwargs)
    cached_paths = [Path(img["file_name"]) for img in first.images().objs]
    mtimes = {path: path.stat().st_mtime_ns for path in cached_paths}
    second = _tile_run(synthetic_kwcoco, tmp_path / "round1.kwcoco.zip", **kwargs)
    assert [img["tile_id"] for img in first.images().objs] == [
        img["tile_id"] for img in second.images().objs
    ]
    assert all(path.is_absolute() and path.is_file() for path in cached_paths)
    assert mtimes == {path: path.stat().st_mtime_ns for path in cached_paths}


def test_stable_source_dataset_fingerprint_reuses_regenerated_selection(tmp_path):
    import kwcoco

    source1 = kwcoco.CocoDataset.demo("shapes2", image_size=(128, 128))
    source1.reroot(absolute=True)
    source1.fpath = tmp_path / "selection1.kwcoco.zip"
    source1.dump()
    source2 = source1.copy()
    source2.dataset.setdefault("info", []).append({"regenerated": True})
    source2.fpath = tmp_path / "selection2.kwcoco.zip"
    source2.dump()
    common = {
        "mode": "multiscale",
        "category_names": "star",
        "progress": False,
        "tile_size": 64,
        "source_scales": "1.0",
        "cache_dpath": str(tmp_path / "cache"),
        "source_dataset_fingerprint": "canonical-dataset-v1",
    }
    first = _tile_run(source1.fpath, tmp_path / "tiles1.kwcoco.zip", **common)
    second = _tile_run(source2.fpath, tmp_path / "tiles2.kwcoco.zip", **common)
    assert {
        img["tile_materialization_id"] for img in first.images().objs
    } == {
        img["tile_materialization_id"] for img in second.images().objs
    }
    assert first.dataset["info"][0]["source_manifest_sha256"] != second.dataset["info"][0]["source_manifest_sha256"]
    assert first.dataset["info"][0]["source_dataset_fingerprint"] == "canonical-dataset-v1"


# ---------------------------------------------------------------------------
# Category filter — only the target category gets emitted
# ---------------------------------------------------------------------------


def test_unknown_category_raises(synthetic_kwcoco, tmp_path):
    from kwcoco_detector_kit.data.tile import TileConfig, run
    cfg = TileConfig.cli(
        argv=False,
        data={
            "src": str(synthetic_kwcoco), "dst": str(tmp_path / "x.kwcoco.zip"),
            "mode": "multiscale", "category_names": "MISSING_CATEGORY",
            "progress": False,
        },
    )
    with pytest.raises(RuntimeError):
        run(cfg)


# ---------------------------------------------------------------------------
# Tile-extent metadata round-trip
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Multi-class — category order, kept categories, dropped categories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["full_only", "quadrant", "multiscale"])
def test_multi_class_assigns_ids_in_order(synthetic_kwcoco_factory, tmp_path, mode):
    src = synthetic_kwcoco_factory(
        f"mc_{mode}", num_images=4, boxes_per_image=2,
        category_names=("widget", "gizmo"),
    )
    dst = tmp_path / f"mc_{mode}.kwcoco.zip"
    # Reverse CLI order so gizmo gets id=1, widget gets id=2.
    dset = _tile_run(
        src, dst, mode=mode, category_names="gizmo,widget", progress=False,
        tile_size=128, source_scales="1.0", min_source_scale_long_side=32,
        min_gt_area_frac=0.0001, keep_negative=False,
        tile_grid=2, tile_overlap=0.20, tile_output_dim=128, keep_full=False,
        full_dim=128,
    )
    cats = {c["id"]: c["name"] for c in dset.dataset["categories"]}
    assert cats == {1: "gizmo", 2: "widget"}
    cat_ids_used = {ann["category_id"] for ann in dset.annots().objs}
    assert cat_ids_used == {1, 2}, (
        f"expected both classes in tiled annotations; got {cat_ids_used}"
    )


def test_multi_class_drops_unrequested_category(synthetic_kwcoco_factory, tmp_path):
    src = synthetic_kwcoco_factory(
        "mc_drop", num_images=4, boxes_per_image=2,
        category_names=("widget", "gizmo", "doodad"),
    )
    dst = tmp_path / "mc_drop.kwcoco.zip"
    dset = _tile_run(
        src, dst, mode="multiscale",
        category_names="widget,gizmo",  # drop doodad
        progress=False,
        tile_size=128, source_scales="1.0", min_source_scale_long_side=32,
        min_gt_area_frac=0.0001, keep_negative=False,
    )
    cat_names = {c["name"] for c in dset.dataset["categories"]}
    assert cat_names == {"widget", "gizmo"}, cat_names


# ---------------------------------------------------------------------------
# Tile-extent metadata round-trip
# ---------------------------------------------------------------------------


def test_tile_extent_is_recorded_for_quadrant(synthetic_kwcoco, tmp_path):
    dst = tmp_path / "q_extent.kwcoco.zip"
    dset = _tile_run(
        synthetic_kwcoco, dst, mode="quadrant",
        category_names="widget", progress=False,
        tile_grid=2, tile_overlap=0.20, tile_output_dim=128, keep_full=False,
    )
    for img in dset.images().objs:
        ext = img.get("tile_extent_xyxy_in_source")
        assert ext is not None and len(ext) == 4
        x0, y0, x1, y1 = ext
        assert x1 > x0 and y1 > y0


# ---------------------------------------------------------------------------
# source_category passthrough — regression for the May-2026 baseline-empty bug
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["full_only", "quadrant", "multiscale"])
def test_source_category_is_stamped_from_src_dset(synthetic_kwcoco, tmp_path, mode):
    """Annotations in the source bundle carry class info via category_id only
    (no pre-set source_category). The tile writer must stamp source_category
    from the source dataset's category lookup so apply_scheme can collapse
    classes downstream. Regression for the May-2026 cycle where tiled output
    had 903,603 anns and zero source_category, producing empty MSCOCO and
    a 48h x 3-job dud.
    """
    dst = tmp_path / f"sc_{mode}.kwcoco.zip"
    dset = _tile_run(
        synthetic_kwcoco, dst, mode=mode, category_names="widget",
        progress=False,
        tile_size=128, source_scales="1.0", min_source_scale_long_side=32,
    )
    anns = list(dset.dataset.get("annotations", []))
    assert anns, f"{mode}: no annotations emitted"
    missing = [a for a in anns if not a.get("source_category")]
    assert not missing, (
        f"{mode}: {len(missing)}/{len(anns)} annotations missing "
        f"source_category (sample: {missing[:1]})"
    )
    assert all(a["source_category"] == "widget" for a in anns), \
        f"{mode}: source_category should be the raw source class name"
