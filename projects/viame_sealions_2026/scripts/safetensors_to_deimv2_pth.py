#!/usr/bin/env python3
"""
Convert a HuggingFace ``model.safetensors`` to a DEIMv2-compatible
``.pth`` checkpoint.

DEIMv2's ``engine/solver/_solver.load_tuning_state(path)`` calls
``torch.load(path)`` and expects either ``state['model']`` (typical) or
``state['ema']['module']`` (EMA copy) to hold the state_dict. The HF
distribution stores the same parameters as a flat
``key -> tensor`` mapping in safetensors format. This script wraps that
mapping in the ``{'model': state_dict}`` envelope and writes a .pth.

Idempotent: if the destination .pth already exists with the same
parameter count, exits successfully.

Usage:

    python3 scripts/safetensors_to_deimv2_pth.py \\
        --src /path/to/model.safetensors \\
        --dst /path/to/model.pth

The fetch_pretrained.sh script calls this automatically — you should
rarely need to invoke it by hand.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def convert(src: Path, dst: Path, *, verify: bool = True) -> None:
    import torch
    from safetensors import safe_open

    state_dict: dict = {}
    with safe_open(str(src), framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    if not state_dict:
        raise RuntimeError(f"no tensors loaded from {src}")

    checkpoint = {"model": state_dict}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(dst))

    if verify:
        round_trip = torch.load(str(dst), map_location="cpu", weights_only=False)
        assert "model" in round_trip, "round-trip checkpoint missing 'model' key"
        assert len(round_trip["model"]) == len(state_dict), (
            "round-trip parameter count mismatch: "
            f"wrote {len(state_dict)}, loaded {len(round_trip['model'])}"
        )

    sample_keys = sorted(state_dict)[:5]
    print(f"wrote {dst}")
    print(f"  parameters: {len(state_dict)}")
    print(f"  first keys: {sample_keys}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True,
                        help="input model.safetensors")
    parser.add_argument("--dst", type=Path, required=True,
                        help="output .pth (DEIMv2 fine-tuning checkpoint)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing dst even if it looks valid")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip post-write round-trip load check")
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"src does not exist: {args.src}")

    if args.dst.exists() and not args.force:
        import torch
        try:
            existing = torch.load(str(args.dst), map_location="cpu", weights_only=False)
            n_existing = len(existing.get("model", {}))
        except Exception:
            n_existing = -1
        if n_existing > 0:
            print(f"{args.dst} already exists ({n_existing} params); skipping. "
                  f"Pass --force to overwrite.")
            return 0

    convert(args.src, args.dst, verify=not args.no_verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
