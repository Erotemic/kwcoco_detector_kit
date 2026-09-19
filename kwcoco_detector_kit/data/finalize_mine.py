"""Finalize a complete set of virtual hard-negative mining shards."""
from __future__ import annotations

import kwconf


class FinalizeMineConfig(kwconf.Config):
    """Verify shard coverage, choose global top-K, and materialize winners."""

    candidate_index = kwconf.Value(None, required=True)
    ledgers = kwconf.Value(None, required=True, nargs="+")
    dst = kwconf.Value(None, required=True)
    cache_dpath = kwconf.Value(None, required=True)
    jpeg_quality = kwconf.Value(90)
    score_thresh = kwconf.Value(0.30)
    max_hard_per_round = kwconf.Value(5000)
    allow_failures = kwconf.Value(False)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        from kwcoco_detector_kit.data.mine import finalize_virtual_mining

        return finalize_virtual_mining(
            config.candidate_index,
            config.ledgers,
            config.dst,
            cache_dpath=config.cache_dpath,
            jpeg_quality=int(config.jpeg_quality),
            score_thresh=float(config.score_thresh),
            max_hard_per_round=int(config.max_hard_per_round),
            allow_failures=bool(config.allow_failures),
        )


__cli__ = FinalizeMineConfig


if __name__ == "__main__":
    __cli__.main()
