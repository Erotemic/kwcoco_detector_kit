# RF-DETR image

This image pins the KDK checkout and `tpl/rf-detr` submodule, installs
RF-DETR's training extras, and records both revisions in image labels and
`/etc/kcd_provenance.json`. It is intentionally separate from the
OpenGroundingDINO image: RF-DETR requires Transformers 5.x and has no custom
CUDA extension to compile.

Build on the aiq Blackwell host:

```bash
bash docker/rfdetr/build_aiq_cuda132_blackwell.sh
```

Before creating large tile pools, verify all four devices and a tiny native
segmentation forward/backward run inside this image. The ShitSpotter driver
provides the experiment-specific commands; this directory remains generic.
