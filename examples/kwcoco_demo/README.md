# kwcoco_demo — always-runnable smoke example

A self-contained, CPU-only smoke that exercises every Phase-1 module:

```
synth kwcoco bundle
   ↓                                    data/tile (multiscale)
tile-augmented kwcoco
   ↓                                    data/merge (round 0)
training kwcoco
   ↓                                    trainers/mock_tiny
trained checkpoint
   ↓                                    export/onnx + parity + modelspec
exported ONNX + modelspec sidecar
   ↓                                    eval/kwcoco_eval + bench
detect_metrics.json + bench.json
   ↓                                    orchestration/eligibility
manifest.tsv (with one HOST_PROMISING candidate)
```

## Run

```bash
bash examples/kwcoco_demo/run_smoke.sh
```

Acceptance target: **<90 s on a 1-CPU laptop**. No GPU required. Produces a `.onnx`, a populated eligibility manifest, and a `HOST_PROMISING` candidate of `candidate_kind=smoke`.

## Layout

| file | role |
|---|---|
| `run_smoke.sh` | Top-level driver. Calls into `python -m kwcoco_detector_kit ...` for every step. |
| `config.yaml` | scriptconfig-compatible overlay. Knobs: trainer, tier, input_hw, num_epochs, tile_mode. |
