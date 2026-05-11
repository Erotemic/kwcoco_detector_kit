"""
Environment audit / `--check-env` probe + optional install.

Front-loads discovery of transitive runtime deps so the first GPU minute
isn't burned on a `ModuleNotFoundError` (failure #11). Covers:

- Core kwcoco / kwimage / torch / scriptconfig.
- ONNX trio (failures #9 + #10): onnx, onnxruntime, onnxscript, onnxsim.
- DEIMv2 trainer hidden deps: faster_coco_eval, calflops, transformers,
  tensorboard, scipy.
- OpenGroundingDINO + SAM2 (Phase 2): transformers, addict, yapf.

Each probe is a one-line ``importlib.util.find_spec`` check. The aggregate
report exits non-zero when anything required is missing.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import scriptconfig as scfg


@dataclass
class Probe:
    module: str
    pip_name: Optional[str]      # None means "no auto-install"
    group: str                   # "core", "onnx", "deimv2", "opengroundingdino"
    required_in: Tuple[str, ...] = ()


PROBES: List[Probe] = [
    # Core — required everywhere.
    Probe("kwcoco",         "kwcoco",         "core",      ("data",)),
    Probe("kwimage",        "kwimage",        "core",      ("data",)),
    Probe("ubelt",          "ubelt",          "core",      ("*",)),
    Probe("scriptconfig",   "scriptconfig",   "core",      ("*",)),
    Probe("numpy",          "numpy",          "core",      ("*",)),
    Probe("yaml",           "pyyaml",         "core",      ("trainers",)),
    Probe("torch",          None,             "core",      ("trainers",)),
    Probe("torchvision",    None,             "core",      ("trainers",)),
    Probe("cv2",            "opencv-python-headless", "core", ("data",)),
    # ONNX export trio — failures #9, #10.
    Probe("onnx",           "onnx",           "onnx",      ("export",)),
    Probe("onnxruntime",    "onnxruntime",    "onnx",      ("export", "bench")),
    Probe("onnxscript",     "onnxscript",     "onnx",      ("export",)),
    Probe("onnxsim",        "onnxsim",        "onnx",      ("export",)),
    # DEIMv2 hidden transitive deps — failure #11. These live under
    # the `deimv2` extras group; only required when running the DEIMv2 trainer.
    Probe("faster_coco_eval", "faster_coco_eval", "deimv2", ("trainers",)),
    Probe("calflops",         "calflops",         "deimv2", ("trainers",)),
    Probe("transformers",     "transformers",     "deimv2", ("trainers",)),
    Probe("tensorboard",      "tensorboard",      "deimv2", ("trainers",)),
    Probe("scipy",            "scipy",            "deimv2", ("trainers",)),
    # OpenGroundingDINO + SAM2 — Phase 2. Soft.
    Probe("addict",           "addict",           "opengroundingdino", ("trainers",)),
    Probe("yapf",             "yapf",             "opengroundingdino", ("trainers",)),
]


def _present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ValueError, ModuleNotFoundError, ImportError):
        return False


def probe_env(*, groups: Optional[Iterable[str]] = None) -> List[Probe]:
    """Return the list of probes that are MISSING in the current env.

    ``groups`` filters: pass ``("core", "onnx")`` to skip the DEIMv2 +
    OpenGroundingDINO checks.
    """
    selected = [p for p in PROBES if (groups is None or p.group in set(groups))]
    return [p for p in selected if not _present(p.module)]


def install_missing(missing: List[Probe], *, python_executable: Optional[str] = None) -> int:
    """pip install the missing deps that have a pip_name. Returns exit code (0 OK)."""
    pip_names = [p.pip_name for p in missing if p.pip_name]
    if not pip_names:
        return 0
    py = python_executable or sys.executable
    args = [py, "-m", "pip", "install", *pip_names]
    print(f"[setup_audit] {' '.join(args)}")
    return subprocess.call(args)


class CheckEnvConfig(scfg.DataConfig):
    """Audit the env for every transitive runtime dep the kit + its trainers need."""

    groups = scfg.Value(
        "core,onnx",
        help="comma-separated groups to probe: core,onnx,deimv2,opengroundingdino",
    )
    install = scfg.Value(
        False, isflag=True,
        help="attempt to pip install missing modules",
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        return run(config)


def run(config) -> int:
    groups = [g.strip() for g in str(config.groups).split(",") if g.strip()]
    missing = probe_env(groups=groups)
    if not missing:
        print(f"[check-env] all probes ok for groups={groups}")
        return 0

    print(f"[check-env] {len(missing)} missing modules across groups={groups}:")
    for p in missing:
        hint = f"pip install {p.pip_name}" if p.pip_name else "(install manually)"
        print(f"  - {p.module:25s} group={p.group:18s} -> {hint}")

    if not bool(config.install):
        print("\npass --install to attempt automatic pip install.")
        return 1

    rc = install_missing(missing)
    if rc != 0:
        print(f"[check-env] pip install returned {rc}")
        return rc

    # Re-probe
    still_missing = probe_env(groups=groups)
    if still_missing:
        names = ", ".join(p.module for p in still_missing)
        print(f"[check-env] still missing after install: {names}")
        return 2
    print("[check-env] all probes ok after install.")
    return 0


__cli__ = CheckEnvConfig
