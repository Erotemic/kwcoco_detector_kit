# kwcoco-detector-kit — handoff plan for the next agent

This is the complete brief. Paste it into the new agent's context.

---

## 1. Mission

Build **`kwcoco-detector-kit`** — a clean, domain-agnostic Python package that turns any kwcoco detection dataset into a trained detector via a configurable Pareto sweep, optional multi-scale tile augmentation, and round-based hard-negative mining. The kit must scale from **tiny mobile detectors on a single 12-16 GB GPU** all the way to **DINOv2/DINOv3-backed transformer detectors on multi-GPU clusters** (4× Titan, 4× 24 GB Ampere, 4× 80 GB A100, 4× 96 GB Blackwell, A100/H100 cloud). Mobile deploy is only one of several deploy targets; aerial sea-lion detection, satellite mosaics, and multispectral remote sensing are first-class citizens.

The kit was prototyped inside `/home/joncrall/code/shitspotter/`. **Two prototype lineages** to harvest:

- `experiments/mobile_app_training_v{4,5}/` — DEIMv2 (mobile + small) + tile augmentation + hard-neg mining + Pareto sweep. Recent (2026-05), well-tested (~62 pytest tests).
- `experiments/foundation_detseg_v3/` — OpenGroundingDINO (DINOv2 + BERT + DETR) + SAM2. Older lineage; reached AP=0.766 on simplified test GT. The canonical "big DINO detector" training pattern.

The shitspotter prototype is **the reference implementation, not an example to ship.** Public examples in the kit:

1. `examples/kwcoco_demo/` — uses `kwcoco.CocoDataset.demo('shapes8')` for cold CI smoke.
2. `examples/sealion_aerial/` — single-class aerial detection on the NOAA Steller Sea Lion dataset.
3. `examples/multispectral_<TBD>/` — placeholder for v1.1 (ask user which dataset).

---

## 2. Source map — where to harvest from

All paths absolute on the VM. **Read-only sources** — never modify under `/home/joncrall/code/shitspotter/` or `/home/agent/code/kwcoco_dataloader/`. Copy out, then port.

### 2.1 Small-detector lineage (mobile, single-GPU)

```text
/home/joncrall/code/shitspotter/experiments/mobile_app_training_v4/
├── tile_kwcoco.py                    v4 tile extractor (variable-size, NxN grid)
├── _train_deimv2_variant.sh          THE most important file in v4. Contains:
│                                       variant-keyed batch/epoch table,
│                                       per-(variant,input_size) memory tuning,
│                                       RLIMIT_NOFILE/FD-storm guards,
│                                       multi-scale policy parser,
│                                       YAML config generator,
│                                       HGNetv2 fixed-input constraint enforcement,
│                                       OOM fail-hint, GPU pinning.
├── 02_sweep.sh                       Pareto sweep state machine
├── 03_export_onnx.sh                 DEIMv2 ONNX export + modelspec sidecar
├── 04_eval_on_test.sh                kwcoco eval driver
├── 05_desktop_onnx_parity.py         torch ↔ ONNX parity guard
├── 06_benchmark_onnx_desktop.py      desktop CPU latency probe
├── eligibility_manifest.py           Pareto winner selection state machine
│                                       (NOT_READY / HOST_PROMISING /
│                                        PHONE_ELIGIBLE / PHONE_INELIGIBLE,
│                                        candidate_kind=real|smoke)
├── 00_setup.sh                       env probe + dep installer
├── 01_make_tile_augmented_kwcoco.sh  tile + simplify orchestration
├── run_all.sh                        one-shot driver
├── common.sh + setup_env.sh          env scaffold (idempotent PYTHONPATH)
├── v4_mock.py                        v4_mock_tiny scriptable detector for CPU CI
├── _train_v4_mock_variant.sh         mock trainer dispatch
├── README.md + DESIGN.md + AUDIT.md  READ THESE IN FULL BEFORE STARTING

/home/joncrall/code/shitspotter/experiments/mobile_app_training_v5/
├── v5_tile.py                        multi-scale fixed-size tile extractor
├── v5_merge.py                       pos + neg union for round training
├── v5_mine.py                        hard-negative mining
├── 02_train_round.sh                 single-round driver (dispatches v4 trainer)
├── 03_mine_hard_negatives.sh
├── run_round_loop.sh                 N-round mining loop
├── run_all.sh
├── README.md + DESIGN.md

/home/joncrall/code/shitspotter/tests/mobile_app_training_v4/
├── conftest.py                       synthetic_kwcoco hand-built fixture pattern
├── test_tile_kwcoco.py               (8 tests)
├── test_train_policy_parser.py       (7 tests)
├── test_eligibility_manifest.py      (11 tests)
├── test_v4_mock.py                   (5 tests)
├── test_candidate_kind.py            (6 tests)
├── test_simplify_status.py           (5 tests; SKIP — shitspotter-specific)

/home/joncrall/code/shitspotter/tests/mobile_app_training_v5/
├── conftest.py                       session-scope v5_tile_bundle fixture
├── test_v5_tile.py                   (11 tests)
├── test_v5_merge.py                  (5 tests)
├── test_v5_mine_offline.py           (4 tests)
```

### 2.2 Big-detector lineage (DINOv2/DINOv3 + DETR, multi-GPU)

This is **the right reference for sealion / aerial / big-DINO** use cases. The shitspotter v9 result (AP=0.766) was achieved here.

```text
/home/joncrall/code/shitspotter/experiments/foundation_detseg_v3/
├── v9_train_eval_opengroundingdino_sam2.sh   THE canonical OpenGroundingDINO trainer
│                                              + SAM2 co-training. ~900 lines, full
│                                              checkpoint-selection sweep, dense and
│                                              simplified-GT eval. READ IN FULL before
│                                              Phase 2.
├── train_deimv2_detector.sh          DEIMv2-S/M training entrypoint (DINOv3-backed)
├── train_deimv2_detector_from_coco.sh  variant for COCO-style inputs
├── train_sam2_segmenter.sh           SAM2 fine-tuning (parallel)
├── train_maskdino_baseline.sh        MaskDINO-R50 baseline (instance seg)
├── run_deimv2_boxes_on_{vali,test}.sh
├── run_deimv2_sam2_on_{vali,test}.sh
├── run_maskdino_on_{vali,test}.sh
├── run_bootstrap_new_cohort.sh       annotation-seeding pipeline
├── run_gdino_sam2_sweep.sh           hyperparameter sweep for OGDino
├── aggregate_foundation_results.sh
├── download_foundation_assets.sh
├── common.sh, setup_environment.sh
├── KARPATHY_LOOP.md                  iteration-discipline doc
├── README.md                         full v3 rationale
├── UPSTREAM_ENVIRONMENT_OVERRIDES.md env-tuning notes for heavier deps
└── packages/                         per-trained-variant deployment package YAMLs

/home/joncrall/code/shitspotter/shitspotter/algo_foundation_v3/
├── detector_deimv2.py                DEIMv2Predictor + DEIMv2 trainer dispatch
├── detector_opengroundingdino.py     OpenGroundingDINO predictor + trainer.
│                                       CRITICAL — the working DINOv2+BERT+DETR pipeline.
├── segmenter_sam2.py                 SAM2 trainer + predictor (instance segm).
│                                       Optional for v1.0; v1.2 segm support.
├── baseline_maskdino.py              MaskDINO trainer (alternative big detector)
├── cli_train.py                      top-level training CLI; dispatches by variant name
├── cli_predict.py / cli_predict_boxes.py / cli_predict_gtboxes.py
├── cli_package.py                    builds a "package YAML" describing a trained model
│                                       + postprocess hyperparameters. ADOPT this pattern.
├── coco_adapter.py                   _build_coco_export: kwcoco → MSCOCO json
├── datasets.py                       unified dataset preparation
├── kwcoco_adapter.py                 kwcoco I/O utilities
├── postprocess.py                    detector postprocessing (NMS, score thresh, etc.)
├── polygon_utils.py
├── merge_nearby_anns.py              annotation cluster-merging (more general than
│                                       shitspotter's simplify_kwcoco). Optional preprocess.
├── model_registry.py                 canonical model-variant registry. THE PATTERN for
│                                       the kit's trainer registry.
├── packaging.py
└── tests/
```

### 2.3 DEIMv2 model zoo

The shitspotter prototype trains seven HGNetv2 variants (Atto → X) and four DINOv3 variants (S/M/L/X). The kit registers all of these.

```text
/home/joncrall/code/shitspotter/tpl/DEIMv2/                REFERENCE ONLY, do NOT copy.
                                                            Submodule or optional dep.
├── configs/deimv2/
│   ├── deimv2_hgnetv2_{atto,femto,pico,n,s,m,l,x}_coco.yml
│   └── deimv2_dinov3_{s,m,l,x}_coco.yml
├── train.py                          subprocess; never re-implement
└── tools/deployment/export_onnx.py   subprocess for ONNX export
```

### 2.4 Webdataset / efficient batch storage (NEW — see §6)

```text
/home/agent/code/kwcoco_dataloader/                        READ-ONLY reference.
├── kwcoco_dataloader/
│   ├── cli/
│   │   └── build_webdataset.py       1472 lines. The existing kwcoco → webdataset
│   │                                   conversion pipeline. CRITICAL READ. Header
│   │                                   docstring lays out the design rationale +
│   │                                   open issues (random vs sequential sampling,
│   │                                   class-balanced over/undersampling, lost
│   │                                   metadata, lost spatial augmentation).
│   ├── tasks/fusion/datamodules/
│   │   ├── kwcoco_datamodule.py      907 lines. Lightning DataModule wrapping
│   │   │                               the windowed sampler.
│   │   ├── kwcoco_dataset.py         4786 lines. THE sampler core. Heterogeneous
│   │   │                               image sequences, channels, sensors, balanced
│   │   │                               sampling, pixelwise weighting.
│   │   ├── balanced_sampling.py      744 lines. Class-balanced sampler that handles
│   │   │                               under/over-represented classes.
│   │   ├── data_augment.py
│   │   ├── data_utils.py
│   │   ├── dimension_metadata.py
│   │   ├── dynamic_channel_handler.py
│   │   └── batch_visualization.py
│   └── utils/
│       ├── kwcoco_extensions.py
│       ├── util_bands.py
│       ├── util_fsspec.py            cloud-mount kwcoco support (S3, GCS)
│       ├── util_kwarray.py / util_kwimage.py / util_raster.py
│       └── ...
├── README.rst                        Module overview
└── tests/
```

### 2.5 Hard-won lessons

```text
/home/joncrall/code/shitspotter/dev/journals/lessons_learned.md
    2026-05-10 and 2026-05-11 entries. 19 documented failure modes (§7 below).

/home/joncrall/code/shitspotter/dev/benchmark-candidates/pipeline-bootstrap-questions.md
    Four benchmark candidates with invariants + acceptance criteria.

/home/joncrall/code/shitspotter/dev/AGENT_BENCHMARK_DISCIPLINE.md
    Framework for promoting lessons → benchmark questions. Adopt in the kit.
```

---

## 3. Scale-tier matrix — what runs where

| tier | example hardware | aggregate VRAM | recommended variant family | sample (variant, input_hw, batch) |
|---|---|---|---|---|
| **S** legacy single-GPU | 1× GTX 1080 Ti / Titan X (12 GB) | 12 GB | DEIMv2 HGNetv2 Atto/Femto/Pico | (atto, 320, 32), (pico, 320, 32) |
| **M** consumer single-GPU | 1× RTX 3090/4090 (24 GB) | 24 GB | DEIMv2 HGNetv2 N/S, DEIMv2 DINOv3-S | (n, 320, 32), (s, 416, 16), (dinov3_s, 640, 8) |
| **L** workstation | 1× RTX 6000 Ada / L40S (48 GB) | 48 GB | DEIMv2 DINOv3-M, OGDino-Swin-Tiny | (dinov3_m, 640, 16), (ogdino_swint, 800, 4) |
| **XL** single server | 1× A100/H100 (80 GB) | 80 GB | DEIMv2 DINOv3-L/X, OGDino-Swin-Base | (dinov3_l, 800, 16), (ogdino_swinb, 1024, 4) |
| **2-4×L** small cluster | 2-4× 24-48 GB | 96-192 GB | same as L per GPU, DDP for bigger batches | DDP × 4 |
| **4×XL** mid cluster | 4× A100 80 GB / 4× H100 / 4× B200 96 GB | 320-384 GB | DINOv3-X, full OGDino sweep | DDP × 4, large batches |
| **cloud** | N×A100/H100 via SLURM/k8s | varies | same as L-XL with cloud-mount kwcoco | document SLURM submit pattern |

### Tier-aware automation

The shitspotter `_train_deimv2_variant.sh` already has the single-GPU tier-M memory table. For the kit:

1. **Expand the table** to all variants × all tiers × all input sizes (~12 × 7 × 4 = 336 cells; not all interesting, but a dense lookup is cheap).
2. **Multi-GPU**: use the v4 trainer's `torch.distributed.run` pattern. Per-GPU batch × num_gpus = effective batch.
3. **FSDP / sharded** not needed for v1 — DINOv3-X (50M params + 800×800) fits in 80 GB. Defer.
4. **Auto-tier-detect**: at trainer launch, query `torch.cuda.mem_get_info()` on rank 0 × world_size, look up closest tier, default conservative. `--tier S/M/L/XL/cluster` overrides.
5. **AMP**: True for tier ≥ M; off for tier S. Already in v4 via `V4_USE_AMP`.

### Heterogeneous-GPU warning (failure #17)

The shitspotter host has 2× RTX 3090 but GPU 1 is on 2× PCIe — multi-GPU all-reduce was SLOWER than single-GPU. Kit's tier detection:

- Probe PCIe link width per device (`nvidia-smi --query-gpu=pcie.link.width.current`).
- **Warn** if any active device has < 8× PCIe AND `num_gpus > 1`.
- Default `CUDA_VISIBLE_DEVICES=0` for single-host non-cluster.

---

## 4. Architecture

```text
kwcoco-detector-kit/
├── pyproject.toml                    Python 3.10+ supported.
├── README.md
├── DESIGN.md                         Lift v4+v5+v3/v9 DESIGN.md sections, scrubbed.
├── kwcoco_detector_kit/
│   ├── __init__.py
│   ├── _version.py
│   ├── data/
│   │   ├── tile.py                   union v4 tile_kwcoco.py + v5_tile.py:
│   │   │                               mode="quadrant" (v4 NxN grid)
│   │   │                               mode="multiscale" (v5)
│   │   │                               mode="full_only" (no tiling)
│   │   ├── merge.py                  from v5_merge.py
│   │   ├── mine.py                   from v5_mine.py (predictor-adapter-aware)
│   │   ├── coco_export.py            from coco_adapter.py
│   │   ├── merge_nearby.py           from merge_nearby_anns.py (optional preprocess)
│   │   ├── stats.py                  NEW: per-channel mean/std probe (multispectral)
│   │   ├── webdataset_writer.py      NEW: lift + simplify build_webdataset.py.
│   │   │                               See §6 — turns the tile pool into wds shards.
│   │   └── webdataset_reader.py      NEW: torch IterableDataset wrapping wds shards.
│   ├── trainers/
│   │   ├── _interface.py             BaseTrainer Protocol
│   │   ├── _registry.py              @register_trainer decorator
│   │   ├── _tier.py                  scale-tier detection + lookup
│   │   ├── deimv2.py                 from _train_deimv2_variant.sh.
│   │   │                               PORTED TO PYTHON (failure #13).
│   │   │                               All 12 DEIMv2 variants.
│   │   └── opengroundingdino.py      from v9_train_eval_opengroundingdino_sam2.sh
│   │                                   + algo_foundation_v3/detector_opengroundingdino.py
│   ├── predictors/
│   │   ├── _interface.py             BasePredictor Protocol
│   │   ├── deimv2.py                 from detector_deimv2.py
│   │   └── opengroundingdino.py      from detector_opengroundingdino.py
│   ├── export/
│   │   ├── onnx.py                   from 03_export_onnx.sh's heredoc. Python.
│   │   ├── parity.py                 from 05_desktop_onnx_parity.py
│   │   ├── modelspec.py              from 03's sidecar writer
│   │   └── package.py                from cli_package.py — deployment package YAML
│   ├── eval/
│   │   ├── kwcoco_eval.py            from 04_eval_on_test.sh's python heredoc
│   │   ├── checkpoint_select.py      from v9 "checkpoint shortlist on validation"
│   │   └── bench.py                  from 06_benchmark_onnx_desktop.py
│   ├── orchestration/
│   │   ├── pareto_sweep.py           from 02_sweep.sh. Python.
│   │   ├── round_loop.py             from v5/run_round_loop.sh. Python.
│   │   ├── eligibility.py            from eligibility_manifest.py
│   │   └── setup_audit.py            from 00_setup.sh
│   ├── cli/
│   │   └── __main__.py               thin click/scriptconfig CLI
│   └── _env.py
├── examples/
│   ├── kwcoco_demo/                  always-runnable, no external data
│   │   ├── README.md
│   │   ├── run_smoke.sh
│   │   └── config.yaml
│   └── sealion_aerial/               real-data example, tier L/XL
│       ├── README.md
│       ├── prepare_kwcoco.py
│       ├── config.yaml               multi-scale params + DINOv3 variant
│       └── run_all.sh
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_tile.py
│   │   ├── test_merge.py
│   │   ├── test_mine_offline.py
│   │   ├── test_train_config_gen.py
│   │   ├── test_tier_detection.py
│   │   ├── test_supports_dynamic.py
│   │   ├── test_eligibility.py
│   │   ├── test_webdataset_writer.py NEW
│   │   └── test_webdataset_reader.py NEW
│   ├── integration/
│   │   ├── test_kwcoco_demo_end_to_end.py
│   │   ├── test_round_loop.py
│   │   └── test_webdataset_pipeline.py NEW
│   └── candidates/pipeline-bootstrap-questions.md
└── docs/
    ├── architecture.md
    ├── scale_tiers.md
    ├── multi_gpu.md
    ├── multispectral.md              v1.1 doc
    ├── webdataset.md                 NEW: storage-format tradeoffs (§6)
    ├── ci.md
    └── lessons.md
```

---

## 5. Three abstractions the kit must get right

### 5.1 Trainer plugin interface

```python
# kwcoco_detector_kit/trainers/_interface.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class DetectorTrainer(Protocol):
    name: str               # "deimv2", "opengroundingdino"
    variants: dict          # variant_name -> arch params
    supports_onnx_export: bool

    def generate_config(
        self, train_kwcoco_fpath, vali_kwcoco_fpath, workdir, *,
        variant: str,
        input_hw: tuple[int, int],
        train_policy: str,             # "fixed" | "multiscale" | "multiscale_<lo>_<hi>"
        num_classes: int,
        batch_size: int,
        val_batch_size: int,
        num_epochs: int,
        lr: float, backbone_lr: float,
        use_amp: bool,
        channels: str,                 # "r|g|b" or "B02|B03|B04|B08"
        scale_tier: str,               # "S"|"M"|"L"|"XL"|"cluster"
        num_gpus: int,
        data_format: str,              # NEW: "kwcoco" | "webdataset"
        extra: dict,
    ) -> Path: ...

    def launch(self, config_fpath, *, init_checkpoint=None, resume=None,
               num_gpus=1, distributed=False) -> Path: ...

    def find_checkpoint(self, workdir) -> Path: ...

    def supports_dynamic_input(self, variant: str) -> bool:
        """DEIMv2 HGNetv2 = False; DEIMv2 DINOv3 = True; OGDino DETR = True."""

    def memory_tier_default_batch(self, variant, input_hw, total_vram_gb) -> int:
        """Per-GPU batch lookup; multi-GPU multiplies."""

    def supports_webdataset_input(self) -> bool:
        """True iff the trainer can read pre-rendered webdataset shards."""
```

Initial trainer plugins:

- `kwcoco_detector_kit.trainers.deimv2` — DEIMv2 family (mobile + edge + DINOv3 server-class).
- `kwcoco_detector_kit.trainers.opengroundingdino` — OGDino (DINOv2 + BERT + DETR).

Stretch: `maskdino`, `yolox`, `rt-detr`.

### 5.2 Predictor plugin interface

```python
class DetectorPredictor(Protocol):
    def __init__(self, ckpt_fpath, config_fpath, device): ...
    def predict_image(self, image_np, orig_size) -> list[dict]:
        """[{'label': int, 'bbox_xyxy': [...], 'score': float}, ...]"""
    @property
    def eval_spatial_size(self) -> tuple[int, int]: ...
```

### 5.3 Channel system (multispectral, v1.1)

```python
class TileConfig:
    channels: str = "r|g|b"
    normalization: dict | None = None    # if None, probe via stats.py
```

Trainer plugin handles non-3-channel input by replacing the first conv's `in_channels` and the modelspec normalization. **v1.0 ships RGB only.** Document in `docs/multispectral.md` as v1.1 expansion path.

---

## 6. Webdataset extension — efficient batch storage

The v4/v5 prototype writes one .jpg per tile. For large datasets and long training runs, this is I/O-bound; webdataset (tar shards) is the standard fix.

`/home/agent/code/kwcoco_dataloader/kwcoco_dataloader/cli/build_webdataset.py` (1472 lines) is an EXISTING kwcoco → webdataset pipeline by the same author. **Read it in full** before designing the kit's version — the docstring lays out the design tradeoffs in detail.

### What the existing build_webdataset.py already solves

- Sequential reads (no random-access penalty on spinning disks).
- Pre-computed truth/metadata/tensor packing/normalization (saves dataloader CPU).
- Class-balanced over/undersampling via separate shards per class + bucket sampling.
- Heterogeneous-channel support (kwcoco_dataloader was built for geowatch's multi-sensor satellite work).
- WebDataset .index sidecar handling for wids random access.
- DDP integration via webdataset's distributed-aware sampling.

### Open issues called out in the existing code (lift verbatim into kit docs)

| issue | upstream note | kit response |
|---|---|---|
| Lost spatial augmentation — pre-rendered tiles can't shift sampling window | "sample larger windows + crop at load time" | Implement load-time crop in `webdataset_reader.py` |
| Lost rich kwcoco metadata | "add extra fields to webdataset records" | Same; document the per-tile metadata schema |
| Balanced sampling now means duplicated data on disk | "use wds index files + bucket headers/footers" | Document the disk-size tradeoff |
| Heterogeneous image sizes / time / sensor / channel | "this webdataset path assumes homogeneous outputs" | Kit's webdataset path is also homogeneous; multi-modal stays on the kwcoco-direct path |
| `.index` sidecar only with `use_mmap=False` | upstream link | Document |
| DDP + WebDataset semantics nontrivial | links to wds/lightning examples | Document the canonical recipe |

### Kit design: storage backends as a knob

Two data-loading backends, selectable per training run:

```python
# kwcoco_detector_kit/data/webdataset_writer.py
def write_tile_shards(
    tile_kwcoco_fpath: Path,
    out_dpath: Path,
    *,
    shard_size_mb: int = 256,         # ~256 MB shards are wds convention
    n_buckets_per_class: int = 1,     # >1 enables class-balanced sampling
    include_metadata: list[str] = ('tile_role', 'tile_scale_name',
                                    'tile_source_gid', 'tile_extent_xyxy'),
    progress: bool = True,
) -> Path:
    """Convert a kwcoco tile bundle to webdataset shards.
    Returns the shard-index directory path."""

# kwcoco_detector_kit/data/webdataset_reader.py
class WebdatasetTileLoader:
    """torch IterableDataset wrapping the shards.
    Supports DDP-aware sampling, load-time crop augmentation
    (compensates for lost on-the-fly window shifting), and the
    class-balanced bucket sampler."""
```

The trainer plugin's `data_format` knob selects between:
- `data_format="kwcoco"` (default) — the kwcoco-direct path. Slower but supports full augmentation.
- `data_format="webdataset"` — the pre-rendered shard path. Faster but augmentation is load-time only.

For tier S–M (small datasets, single-GPU): `kwcoco` is fine. For tier L–XL or multi-epoch big-data: `webdataset` saves I/O.

### Improvements the kit could land back upstream

Where `kwcoco_dataloader/cli/build_webdataset.py` has explicit TODO/FIXME comments:

1. **Line 232**: "need better customizable translation to webdataset" — the kit's `webdataset_writer.py` should be the canonical helper, then upstream to `kwcoco_dataloader`.
2. **Line 1142**: "TODO" block in the shard-bucket-sampling code — the kit's experience with hard-neg mining (rounds that mix pos + hard-neg pools at different ratios) maps directly onto bucket sampling. Worth contributing back.
3. **Line 1296**: "not maintainable, we need metadata about each" — the kit's modelspec-sidecar pattern is a nice precedent for "tile-side metadata that survives the shard write".
4. **Line 1400**: "Compose spatial provenance if present (TODO)" — the kit's `tile_source_gid` + `tile_extent_xyxy_in_source` schema (from v5_tile.py) is exactly what this needs. Lift and contribute upstream.

### Strategy for the kit

- **Don't fork `kwcoco_dataloader`.** Depend on it.
- **Wrap, don't replace.** `webdataset_writer.py` calls into `kwcoco_dataloader.cli.build_webdataset` (refactored to expose a Python API rather than a CLI-only entry point — that itself is a worthwhile upstream contribution).
- **Document the open issues in `docs/webdataset.md`** so kit users know which augmentations they're trading away.
- **Defer to v1.2** if the v1.0 acceptance bar (kwcoco_demo end-to-end + sealion working) is at risk. Webdataset is an optimization, not a correctness gate.

---

## 7. The 19 failure modes to NOT relearn

Lifted from `dev/journals/lessons_learned.md` 2026-05-11. Each cost real time in the shitspotter prototype.

| # | failure | mitigation |
|---|---|---|
| 1 | gdown 6.x dropped `--fuzzy` | use bare file ID |
| 2 | `kwimage.imresize(img, (W, H))` positional 2 is `scale=`, not `dsize=` | always pass `dsize=` kwarg |
| 3 | `kwimage.imwrite(..., imwrite_params=...)` not a kwarg cv2 accepts | use cv2's flat `params=[FLAG, val]` |
| 4 | gdown silently writes HTML quota-error pages | post-download size guard (≥ 1 MiB) |
| 5 | `kwimage.imresize(interp='area')` fails on skimage backend without cv2 | try/except, fall back to 'linear' |
| 6 | `geowatch.__init__` hard-imports osgeo | OPTIONAL geowatch — soft import |
| 7 | pip 25+ rejects girder index (PEP 700) | document direct-wheel URL pattern |
| 8 | torch/torchvision ABI mismatch (env-local) | pin matching pair in pyproject |
| 9 | `torch.onnx.export` on torch ≥ 2.5 needs `onnxscript` at function-call time | declare `onnxscript` install_requires |
| 10 | DEIMv2's exporter needs `onnxsim`; opset 17 incompatible with torch ≥ 2.11 (Pad has no adapter) | install_requires onnxsim; default opset 18 |
| 11 | DEIMv2 trainer needs faster_coco_eval / calflops / transformers / tensorboard (undeclared) | install_requires pulls all of `tpl/DEIMv2/requirements.txt` |
| 12 | sweep cell can silently record `status=ok` after a failed stage | per-stage exit-code check |
| 13 | YAML `collate_fn` indent-leak makes DEIMv2 pass it to CocoDetection.__init__ | parse-validate generated configs; **port YAML-gen to Python** |
| 14 | HGNetv2 hybrid encoder doesn't support multi-scale (pre-bakes pos_embed); upstream sets `base_size_repeat: ~` | trainer plugin exposes `supports_dynamic_input()`; if False, force `train_policy="fixed"` |
| 15 | non-root users can't raise RLIMIT_NOFILE above `ulimit -Hn` | clamp to hard cap; document `*_TORCH_MP_SHARING=file_system` fallback |
| 16 | DEIMv2 OOMs on 24 GB GPU with naïve upstream batch sizes | per-(variant, input_size, tier) memory table |
| 17 | Multi-GPU all-reduce bottlenecks on mismatched PCIe lanes | default `CUDA_VISIBLE_DEVICES=0` for single-host; warn on PCIe mismatch |
| 18 | DEIMv2 fixed-input encoders need eval_spatial_size = train Resize = val Resize = collate base_size = Mosaic output_size | one input-size knob drives all five via the config generator |
| 19 | `kwcoco subset --select_images "..."` requires undeclared `jq` Python pkg | document `--gids 1,2,3` form as canonical |

---

## 8. Re-engineering decisions during the port

The shitspotter prototype is shell-driven with bash heredocs for YAML. **Do not preserve this.**

### Port to Python

| from (shell) | to (Python) | why |
|---|---|---|
| `_train_deimv2_variant.sh` YAML generator | `trainers/deimv2.py:generate_config()` | failure #13 |
| `v9_train_eval_opengroundingdino_sam2.sh` ODVG-generator + trainer dispatch | `trainers/opengroundingdino.py` | same |
| `02_sweep.sh` | `orchestration/pareto_sweep.py` | testable, async cells |
| `run_round_loop.sh` | `orchestration/round_loop.py` | testable |
| `00_setup.sh` probe-and-install | `orchestration/setup_audit.py` | reuse via `--check-env` CLI |
| v3's checkpoint-select sweep (v9 lines ~696-735) | `eval/checkpoint_select.py` | generic |

### Keep as shell

| stays as shell | why |
|---|---|
| `examples/*/run_all.sh` thin drivers | 30-line config overlays |

### Subprocess upstream

- DEIMv2's `tools/deployment/export_onnx.py` — subprocess. Trap onnxsim ImportError (#10); recover .onnx from staging dir if present.
- DEIMv2's `train.py` — subprocess.
- OpenGroundingDINO's `train_dist.sh` — subprocess.
- `kwcoco eval` CLI — subprocess.

---

## 9. Example projects

### 9.1 `examples/kwcoco_demo/`

```bash
#!/usr/bin/env bash
set -euo pipefail
KCD_ROOT=${KCD_ROOT:-/tmp/kcd_demo_smoke}
mkdir -p "$KCD_ROOT"

python -m kwcoco_detector_kit.cli demo-data \
    --dst "$KCD_ROOT/train.kwcoco.zip" \
    --num_images 16 --num_categories 1

python -m kwcoco_detector_kit.cli run-all \
    --train_kwcoco "$KCD_ROOT/train.kwcoco.zip" \
    --vali_kwcoco  "$KCD_ROOT/train.kwcoco.zip" \
    --test_kwcoco  "$KCD_ROOT/train.kwcoco.zip" \
    --category_name star \
    --trainer mock_tiny \
    --tier S \
    --input_hw 320 320 \
    --num_epochs 2 \
    --workdir "$KCD_ROOT/run"
```

CI smoke on every commit. No GPU needed; uses `mock_tiny` (lifted from `v4_mock.py`).

### 9.2 `examples/sealion_aerial/`

```python
# examples/sealion_aerial/prepare_kwcoco.py
# Convert NOAA Steller dataset. Each image ~5616×3744 RGB.
# Annotations: colored dots → small fixed-size bboxes (~20-50 px).
```

```yaml
# examples/sealion_aerial/config.yaml
category_name: sealion
trainer: deimv2
variant: deimv2_dinov3_s
input_hw: [640, 640]
train_policy: multiscale_512_768
scale_tier: L

tile:
  mode: multiscale
  tile_size: 640
  source_scales: [1.0, 0.5, 0.25, 0.125, 0.0625]
  min_gt_area_frac: 0.001

data_format: kwcoco       # or "webdataset" for tier L+ with long training

round_loop:
  num_rounds: 4
  round0_neg_over_pos: 5.0
  mine_score_thresh: 0.25
  max_hard_per_round: 10000

export:
  onnx: false
  package: true            # deployment package YAML for offline batch

multi_gpu:
  num_gpus: 1
  distributed: false
```

### 9.3 `examples/multispectral_<TBD>/`

v1.1 placeholder. Ask user.

---

## 10. Tests

| from | port to | adjustments |
|---|---|---|
| `tests/mobile_app_training_v4/test_tile_kwcoco.py` | `tests/unit/test_tile.py` | drop `'poop'`; parameterize over kwcoco_demo |
| `tests/mobile_app_training_v4/test_train_policy_parser.py` | `tests/unit/test_train_config_gen.py` | tests Python YAML-gen now |
| `tests/mobile_app_training_v4/test_eligibility_manifest.py` | `tests/unit/test_eligibility.py` | full state machine |
| `tests/mobile_app_training_v4/test_candidate_kind.py` | merge into above | |
| `tests/mobile_app_training_v4/test_simplify_status.py` | drop | shitspotter-specific |
| `tests/mobile_app_training_v5/test_v5_tile.py` | `tests/unit/test_tile.py` | merge with v4 |
| `tests/mobile_app_training_v5/test_v5_merge.py` | `tests/unit/test_merge.py` | direct port |
| `tests/mobile_app_training_v5/test_v5_mine_offline.py` | `tests/unit/test_mine_offline.py` | direct port |

**New** tests:

- `test_train_config_gen.py`: Python config-gen round-trip → yaml.safe_load → structural invariants. Catches #13.
- `test_tier_detection.py`: VRAM lookup picks right tier; warn on PCIe mismatch.
- `test_supports_dynamic_input.py`: per-variant attribute honest; round-loop coerces multiscale → fixed.
- `test_setup_audit.py`: probe-and-install logic.
- `test_channels_rgb.py`: 3-channel default unchanged when `channels=` unspecified.
- `test_kwcoco_demo_end_to_end.py`: full pipeline, mock_tiny CPU fallback.
- `test_webdataset_writer.py`: kwcoco tile bundle → wds shards; per-tile metadata round-trips.
- `test_webdataset_reader.py`: shards → torch IterableDataset; class-balanced sampling honored.

Target: **≥ 80 tests passing on CPU in under 60 seconds.**

---

## 11. Acceptance criteria

1. `pip install kwcoco-detector-kit && python -m pytest` passes on clean Python 3.10-3.13.
2. `bash examples/kwcoco_demo/run_smoke.sh` produces ONNX + populated eligibility manifest with one HOST_PROMISING candidate in <90 s on a 1-CPU laptop.
3. `bash examples/sealion_aerial/run_all.sh` runs end-to-end on single 24 GB GPU (tier M with deimv2_dinov3_s + smaller batch). Doesn't need a specific AP — must not crash, must produce non-trivial manifest.
4. Same `sealion_aerial` config trains under DDP on 4× 24 GB cluster with `--num_gpus 4 --distributed`.
5. Same config with `data_format=webdataset` writes shards, reads them back, and trains end-to-end.
6. README documents adding a new trainer plugin in ≤ 50 lines.
7. No kit source file contains `poop`, `shitspotter`, `mobile_app_training`, `v9 baseline`, `Pixel 5`, `tpl/poop_models`, `tpl/Open-GroundingDino`.
8. Each of §7's 19 failure modes is caught by setup-time probe or pytest test.
9. `docs/lessons.md` enumerates all 19 with mitigations.
10. `docs/scale_tiers.md` documents the tier system with concrete recommendations.
11. `docs/webdataset.md` documents the storage-backend tradeoffs and the upstream-contribution opportunities (§6).

---

## 12. Phased delivery

### Phase 1 — port + 3-channel RGB + tier M/L (2-3 weeks)

1. Scaffold pyproject + module layout.
2. Lift `v5_tile.py` → `data/tile.py` with quadrant/multiscale modes.
3. Lift `v5_merge.py` + `v5_mine.py` → `data/`.
4. Port `_train_deimv2_variant.sh` → `trainers/deimv2.py` IN PYTHON. **Highest-risk task; write tests first.** Covers all 12 DEIMv2 variants.
5. Lift `eligibility_manifest.py` → `orchestration/eligibility.py`.
6. Port `02_sweep.sh` → `orchestration/pareto_sweep.py`.
7. Port `run_round_loop.sh` → `orchestration/round_loop.py`.
8. Build kwcoco_demo example + run-smoke.
9. Port v4 + v5 tests.

Phase 1 acceptance: kwcoco_demo end-to-end smoke; deimv2_n single-GPU on tier M.

### Phase 2 — big DINO + multi-GPU + sealion (2-3 weeks)

1. Lift `algo_foundation_v3/detector_opengroundingdino.py` → `predictors/opengroundingdino.py`.
2. Port `v9_train_eval_opengroundingdino_sam2.sh` ODVG-gen + dispatch → `trainers/opengroundingdino.py`. Skip SAM2 for v1.0.
3. Add DDP support at kit level (already in v4; expose via `--num_gpus`).
4. Scale-tier detection + GPU lookup.
5. `examples/sealion_aerial/prepare_kwcoco.py`.
6. Validate sealion end-to-end on tier M, L, 4×L (DDP).
7. `--check-env` CLI.
8. Documentation.

Phase 2 acceptance: sealion runs on tier L; same config runs on 4× cluster via DDP.

### Phase 3 — webdataset + multispectral + cloud (3-4 weeks)

1. Implement `data/webdataset_writer.py` and `data/webdataset_reader.py` wrapping `kwcoco_dataloader`'s build_webdataset.
2. Add `data_format` knob through trainer plugins.
3. Implement `data/stats.py` per-channel normalization probing.
4. Add `channels=` knob end-to-end.
5. Add multispectral example.
6. Document SLURM submit pattern + cloud-mount kwcoco recipe.
7. **Contribute back to `kwcoco_dataloader`**: the TODOs/FIXMEs in §6 that the kit's work would address.

Phase 3 acceptance: at least one non-RGB example trains end-to-end; webdataset path validated on sealion; H100/A100 cloud-launch documented.

---

## 13. What NOT to bring

- `shitspotter.cli.simplify_kwcoco` and any "v9 baseline = 0.766" reference. (`merge_nearby_anns.py` OK as optional preprocess.)
- `experiments/foundation_detseg_v3/` references by name (lift patterns, drop v3 naming).
- Phone-app deploy contract (`PostprocessType.DEIMV2`, KMP+Compose).
- `Pixel 5`, `mobile_app_training_v4/v5`, `foundation_detseg_v3` names.
- `v4_mock_tiny` rename to `mock_tiny`; keep `candidate_kind="smoke"`.
- Any reference to `github.com/Erotemic/shitspotter`.

---

## 14. Things to ask the user before starting

- **NOAA Steller dataset location**: kwcoco bundle on disk, or build from scratch?
- **Multispectral dataset for v1.1**: Sentinel-2 burn scars? Landsat ship? Worldview cars?
- **Package owner / git host**: github org? gitlab.kitware?
- **Python version range**: confirm 3.10-3.13.
- **Trainer plugins beyond DEIMv2 + OGDino**: YOLOX or RT-DETR in v1, or defer?
- **SAM2 segmenter co-training**: required v1 (v9 result depended on it), or v1.1+?
- **Cluster launch backend**: just `torch.distributed.run`, or SLURM/k8s helpers?
- **Webdataset/kwcoco_dataloader integration**: pin to a specific kwcoco_dataloader version, or develop alongside?
- **Upstream PRs to `kwcoco_dataloader`**: in scope for the kit's Phase 3, or separate effort?

---

## 15. Estimated effort

| phase | hours | running total |
|---|---|---|
| Phase 1 (port + RGB + tier M/L single-GPU) | 60-90 | 60-90 |
| Phase 2 (big DINO + multi-GPU + sealion) | 60-90 | 120-180 |
| Phase 3 (webdataset + multispectral + cloud) | 80-120 | 200-300 |

**One agent full-time: 6-10 weeks.** Highest-risk task: the Python port of the bash YAML generators (failure #13). Budget a full day of pytest-driven dev before connecting to the rest of the pipeline.

---

## 16. Final paste-in for the next agent's first prompt

> You are building **kwcoco-detector-kit** — a clean, domain-agnostic Python package for training detectors on kwcoco datasets. Read the handoff plan at `/data/tmp/kwcoco-detector-kit-plan.md`. The reference implementation lives at `/home/joncrall/code/shitspotter/experiments/mobile_app_training_v{4,5}/` AND `/home/joncrall/code/shitspotter/experiments/foundation_detseg_v3/` — READ-ONLY. The first lineage is small/mobile detectors; the second is big DINOv2/DINOv3-backed detectors. Both must be supported. The webdataset-extension prototype lives at `/home/agent/code/kwcoco_dataloader/` — also READ-ONLY; the kit depends on it and contributes back. Ship Phase 1 first: scaffold + port + RGB-only + kwcoco_demo + DEIMv2 trainer (mobile + edge + DINOv3 variants) + pytest. Confirm with the user before starting Phase 2 (big-DINO + multi-GPU + sealion) or Phase 3 (webdataset + multispectral + cloud). The single biggest correctness risk is the Python YAML-config generators for DEIMv2 and OpenGroundingDINO; write their tests first. The 19 documented failure modes in §7 of the handoff plan must each be either caught by a setup-time probe or covered by a pytest test. Scale targets range from a single 12 GB GTX 1080 Ti up to 4× 96 GB Blackwell or A100/H100 cloud — see the §3 tier matrix. Do not bring `poop`, `shitspotter`, `Pixel 5`, `v9`, or `mobile_app_training` into the kit.

---

End of plan.
