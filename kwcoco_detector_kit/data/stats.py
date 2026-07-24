"""
Per-channel image statistics probe — required for multispectral
normalization (Phase 3).

Computes ``mean`` and ``std`` per channel across a sample of tiles from a
``TileStore``. Result is dumped as JSON and consumed by
``TileLoader(normalize=...)`` or by the trainer plugin's ``channels=``
knob.

The probe is sampling-based: we read up to ``sample_size`` tiles in
order and accumulate Welford's online statistics so memory stays O(C),
independent of the bundle size.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import kwconf


def compute_per_channel_stats(store, *, sample_size: int = 256) -> Dict[str, object]:
    """Compute per-channel mean + std over a sample of tiles.

    Uses Welford's online algorithm so we only ever hold one tile's
    worth of pixels in memory.

    Args:
        store: any TileStore.
        sample_size: max number of tiles to sample (0 = all).

    Returns:
        ``{"channels": str, "num_channels": int, "num_tiles_sampled": int,
        "mean": [c0, c1, ...], "std": [...], "num_pixels": int}``.
        Mean/std are in [0,1] (float32 after dividing uint8 by 255).
    """
    n_tiles_sampled = 0
    n_pixels = 0
    mean: Optional[np.ndarray] = None
    M2: Optional[np.ndarray] = None  # Welford accumulator

    for record in store:
        x = record.image_np.astype(np.float64) / 255.0
        C = x.shape[-1]
        flat = x.reshape(-1, C)
        if mean is None:
            mean = np.zeros(C, dtype=np.float64)
            M2 = np.zeros(C, dtype=np.float64)
        # Welford batch update (one tile = one batch of N pixels)
        Nb = flat.shape[0]
        new_n = n_pixels + Nb
        batch_mean = flat.mean(axis=0)
        delta = batch_mean - mean
        mean += delta * Nb / new_n
        # M2 update: M2 += sum((x - new_mean) * (x - old_mean))
        batch_M2 = ((flat - batch_mean) ** 2).sum(axis=0)
        M2 += batch_M2 + (delta ** 2) * (n_pixels * Nb / new_n)
        n_pixels = new_n
        n_tiles_sampled += 1
        if sample_size and n_tiles_sampled >= int(sample_size):
            break

    if mean is None:
        raise RuntimeError("store had no tiles to sample")
    std = np.sqrt(M2 / max(1, n_pixels - 1))
    channels = str(store.metadata.get("channels", "r|g|b"))
    return {
        "channels": channels,
        "num_channels": int(mean.shape[0]),
        "num_tiles_sampled": int(n_tiles_sampled),
        "num_pixels": int(n_pixels),
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
    }


class StatsConfig(kwconf.Config):
    """Probe per-channel mean/std over a tile bundle. Writes a JSON sidecar."""

    src = kwconf.Value(None, position=1, required=True,
                     help="kwcoco tile bundle OR a WebdatasetStore directory")
    out = kwconf.Value(None, position=2, required=True,
                     help="output JSON path (e.g. <bundle>.stats.json)")
    sample_size = kwconf.Value(256, help="max tiles to sample; 0 = all")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        from kwcoco_detector_kit.data.tile_store import open_store
        store = open_store(str(config.src))
        stats = compute_per_channel_stats(store, sample_size=int(config.sample_size))
        out = Path(str(config.out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=2))
        print(f"wrote {out} (channels={stats['channels']}, "
              f"mean={stats['mean']}, std={stats['std']}, "
              f"n_tiles={stats['num_tiles_sampled']}, n_pixels={stats['num_pixels']})")


__cli__ = StatsConfig
