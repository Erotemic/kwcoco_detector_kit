#!/usr/bin/env bash
git-well archive-source --submodule_depth '
stack/kwimage: 1
stack/kwarray: 1
stack/kwplot: 1
stack/delayed_image: 1
stack/kwcoco: 100
stack/kwcoco_dataloader: 100
tpl/DEIMv2: 10
tpl/MaskDINO: 10
tpl/YOLO-v9: 10
tpl/YOLOX: 10
tpl/rf-detr: 10
#tpl/segment-anything-2: 10
#tpl/Open-GroundingDino: 10
' \
    --exclude_submodule tpl/segment-anything-2 tpl/Open-GroundingDino \
    "${@}"
