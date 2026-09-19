import kwcoco


def test_negative_fp_report_operates_per_source_image():
    from kwcoco_detector_kit.eval.kwcoco_eval import negative_image_fp_report
    true = kwcoco.CocoDataset()
    cid = true.add_category(name="poop")
    gids = [true.add_image(file_name=f"im{i}.jpg") for i in range(3)]
    true.add_annotation(image_id=gids[0], category_id=cid, bbox=[0, 0, 1, 1])
    pred = kwcoco.CocoDataset()
    pcid = pred.add_category(name="poop")
    for gid in gids:
        pred.add_image(id=gid, file_name=f"im{gid}.jpg")
    # Two overlapping-window remnants on one negative source image count as
    # one affected image but two predictions.
    pred.add_annotation(image_id=gids[1], category_id=pcid, bbox=[0, 0, 1, 1], score=.8)
    pred.add_annotation(image_id=gids[1], category_id=pcid, bbox=[1, 1, 1, 1], score=.7)
    report = negative_image_fp_report(
        true, pred, category_names=["poop"], thresholds=[.5, .9],
    )
    assert report["num_negative_source_images"] == 2
    assert report["thresholds"]["0.5"]["num_images_with_fp"] == 1
    assert report["thresholds"]["0.5"]["num_predictions"] == 2
    assert report["thresholds"]["0.9"]["num_images_with_fp"] == 0
