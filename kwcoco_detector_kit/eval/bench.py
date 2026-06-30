"""
ONNX desktop benchmark — measures CPU inference latency on the exported
model. Result lives next to the .onnx as ``<name>.bench.json``.

The eligibility manifest reads ``mean_ms`` from this file as the desktop
CPU proxy gate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import kwconf


def run_onnx_bench(
    *,
    workdir: Path,
    warmup: int = 3,
    iters: int = 20,
    providers: tuple = ("CPUExecutionProvider",),
) -> Path:
    """Time a fixed-shape ONNX run for the candidate's exported model."""
    import numpy as np
    import onnxruntime as ort

    workdir = Path(workdir)
    export_dpath = workdir / "export"
    onnx_files = sorted(export_dpath.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"no .onnx in {export_dpath}")
    onnx_fpath = onnx_files[0]

    sess = ort.InferenceSession(str(onnx_fpath), providers=list(providers))
    inputs = sess.get_inputs()
    images_input = next(i for i in inputs if i.name == "images")
    size_input = next(i for i in inputs if i.name == "orig_target_sizes")
    # Resolve dynamic batch dim: 1.
    shape = []
    for d in images_input.shape:
        if isinstance(d, int):
            shape.append(int(d))
        elif d == "N" or d is None:
            shape.append(1)
        else:
            shape.append(int(d))
    if len(shape) != 4:
        raise RuntimeError(f"unexpected images-input shape {images_input.shape!r}")
    img = np.zeros(shape, dtype=np.float32)
    H, W = shape[2], shape[3]
    sz = np.array([[W, H]], dtype=np.int64)

    for _ in range(int(warmup)):
        sess.run(None, {"images": img, "orig_target_sizes": sz})

    timings = []
    for _ in range(int(iters)):
        t0 = time.perf_counter()
        sess.run(None, {"images": img, "orig_target_sizes": sz})
        timings.append((time.perf_counter() - t0) * 1000.0)

    out_fpath = onnx_fpath.with_suffix(".bench.json")
    payload = {
        "onnx_fpath": str(onnx_fpath),
        "warmup": int(warmup),
        "iters": int(iters),
        "providers": list(providers),
        "timings_ms": timings,
        "mean_ms": sum(timings) / len(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }
    out_fpath.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out_fpath} (mean={payload['mean_ms']:.1f}ms)")
    return out_fpath


class BenchConfig(kwconf.Config):
    """Benchmark the exported ONNX model's CPU inference latency."""

    workdir = kwconf.Value(None, position=1, required=True,
                         help="trained workdir (must contain export/*.onnx)")
    warmup = kwconf.Value(3, help="number of warm-up runs before timing")
    iters = kwconf.Value(20, help="number of timed iterations")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        workdir = Path(str(config.workdir)).expanduser().resolve()
        run_onnx_bench(workdir=workdir, warmup=int(config.warmup), iters=int(config.iters))


def run(config):
    BenchConfig.main(argv=False, **{k: v for k, v in config.items()})


__cli__ = BenchConfig
