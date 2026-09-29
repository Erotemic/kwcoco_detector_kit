# ADR-0002: KDK owns the kwcoco data plane; LibreYOLO is the primary model engine

**Status:** accepted

## Context

KDK needs two things that are easy to couple accidentally:

1. rapid access to current detector/segmenter families and their real training,
   DDP, checkpoint, inference, and export implementations; and
2. kwcoco's richer dataset semantics: large source images, delayed/auxiliary
   assets, explicit coordinate transforms, windows, ignore/truth policy,
   hard-negative replay, and source-space evaluation.

Implementing every model family directly in KDK duplicates model-specific work.
Moving KDK's spatial/dataset semantics into a model framework would instead
flatten the kwcoco source of truth to a conventional image/label dataset.

LibreYOLO already provides a common implementation layer for RF-DETR, D-FINE,
DEIMv2, YOLOv9, GTR, TinyFormer, and other families, including native training
and export paths.

## Decision

KDK and LibreYOLO have an asymmetric boundary:

- **KDK owns the data/control plane.** Kwcoco is authoritative. KDK owns
  preparation, sampling, windowing, delayed-image reads, truth semantics,
  coordinate transforms, mining/replay, campaign state, provenance,
  source-space evaluation, and package acceptance.
- **LibreYOLO owns the model engine.** It owns family construction, losses,
  optimizer/scheduler recipes, family augmentations, AMP/DDP, checkpoint
  lifecycle, native inference, and family-native export.
- The initial bridge is **native COCO JSON** generated from the KDK-prepared
  kwcoco view. It references existing image paths and does not create a second
  tile corpus. Instance segmentation uses COCO segmentation/RLE so holes and
  multipart geometry are not degraded to YOLO TXT rings.
- KDK's `libreyolo` trainer is a **single backend with a declarative variant
  catalog**, not one KDK trainer implementation per model family.
- Prediction results return through KDK's predictor interface, after which
  `PredictionSpace`, tiled merging, source-coordinate restoration, evaluation,
  and mining are unchanged.
- A future lazy sample-provider bridge may remove the temporary COCO JSON
  materialization, but it must preserve this ownership boundary.

## Consequences

- Adding a compatible LibreYOLO detector should normally require a catalog
  entry rather than a new KDK training stack.
- KDK can retire direct family integrations after parity/campaign evidence is
  sufficient, while retaining them temporarily as reference/oracle paths.
- The first LibreYOLO catalog is RGB because those pretrained model stems are
  RGB. Multispectral/auxiliary assets remain first-class in the source kwcoco;
  a non-RGB model is admitted only with an explicit input contract rather than
  silently discarding channels.
- Model-family limitations remain visible. LibreYOLO support for a family/task
  does not imply KDK supports every task exposed by that family.
- LibreYOLO's ONNX export remains family-native. KDK parity wraps it through
  LibreYOLO's ONNX backend and records the `libreyolo_native_v1` contract;
  KDK still decides whether the artifact is acceptable for packaging.

## Alternatives considered

**Keep direct KDK adapters for every model.** Rejected because it duplicates
training and deployment plumbing and makes model velocity proportional to KDK
maintenance work.

**Make LibreYOLO understand kwcoco directly.** Rejected because it pushes
large-image, channel/asset, truth, campaign, and coordinate semantics into the
model engine and creates a second implementation of KDK's data plane.

**Convert the source dataset permanently to YOLO/COCO.** Rejected because the
prepared detector view is derivative. The original kwcoco dataset and its
coordinate/asset semantics remain authoritative.
