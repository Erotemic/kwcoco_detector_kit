# Architecture

```
kwcoco_detector_kit/
├── data/          tile / merge / mine / coco_export — kwcoco-side data plumbing
├── trainers/      _interface + _registry + _tier + deimv2 + mock_tiny
├── predictors/    _interface + per-trainer checkpoint inference adapters
├── export/        onnx + parity + modelspec + package
├── eval/          kwcoco_eval + checkpoint_select + bench
├── orchestration/ pareto_sweep + round_loop + eligibility + setup_audit
└── cli/           scriptconfig-based __main__
```

## Decision lines

### Trainer plugin (`trainers/_interface.py`)

```python
@runtime_checkable
class DetectorTrainer(Protocol):
    name: str
    variants: dict
    supports_onnx_export: bool
    def generate_config(...) -> Path
    def launch(config_fpath, *, init_checkpoint=None, resume=None,
               num_gpus=1, distributed=False) -> Path
    def find_checkpoint(workdir) -> Path
    def supports_dynamic_input(variant) -> bool
    def memory_tier_default_batch(variant, input_hw, total_vram_gb) -> int
    def supports_webdataset_input() -> bool
```

The trainer registry (`_registry.py`) maps `name → DetectorTrainer` so the CLI / orchestration can dispatch by string. Adding a new trainer is a single new module with `@register_trainer("foo")`.

### Predictor plugin (`predictors/_interface.py`)

```python
class DetectorPredictor(Protocol):
    def __init__(ckpt_fpath, config_fpath, device): ...
    def predict_image(image_np, orig_size) -> list[dict]:
        """[{'label': int, 'bbox_xyxy': [...], 'score': float}, ...]"""
    @property
    def eval_spatial_size(self) -> tuple[int, int]: ...
```

### Tile-store interface (Phase-3 hook)

Phase 1's `data/tile.py` writes JPEG + kwcoco manifest. Phase 3 will introduce `data/tile_store.py` with backends (`kwcoco_jpeg` = default, `webdataset` = optional, possibly `parquet` / `lmdb`). The `data_format=` knob on the trainer plugin selects the backend. **Oversized tiles** (`oversize_factor > 1.0`) reserve margin for load-time crop augmentation; this knob exists in Phase 1 even though the load-time crop runs in Phase 3.

### Eligibility state machine

`orchestration/eligibility.py` exposes:

```
NOT_READY          missing checkpoint / ONNX / eval / desktop bench
HOST_PROMISING     all host gates passed; worth taking to a real device
DEPLOY_ELIGIBLE    on-device validation passed (when --device_index supplied)
DEPLOY_INELIGIBLE  device data present but failed
```

Renamed from `PHONE_*` in the prior project; semantics preserved.

`candidate_kind=smoke` is excluded from winner-selection by default — the `mock_tiny` smoke detector should not accidentally win a real sweep.

## Scale tiering

`trainers/_tier.py` queries `torch.cuda.mem_get_info()` × world_size and the active GPU's PCIe link width. Auto-tier-detect picks the conservative tier; `--tier S/M/L/XL/cluster` overrides. The per-`(variant, input_hw, tier)` memory table lives in `trainers/deimv2.py:_BATCH_TABLE` — adding a new variant is one row of dict entries.
