"""
Environment audit / `--check-env` probe + optional install.

Front-loads discovery of transitive runtime deps so the first GPU minute
isn't burned on a `ModuleNotFoundError` (failure #11). Covers:

- Core kwcoco / kwimage / torch / scriptconfig.
- ONNX trio (failures #9 + #10): onnx, onnxruntime, onnxscript, onnxsim.
- DEIMv2 trainer hidden deps: faster_coco_eval, calflops, transformers,
  tensorboard, scipy.
- OpenGroundingDINO + SAM2 (Phase 2): transformers, addict, yapf, colorlog.

Each probe is a one-line ``importlib.util.find_spec`` check. The aggregate
report exits non-zero when anything required is missing.
"""
from __future__ import annotations

import importlib.util
import importlib.metadata
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
    version_spec: Optional[str] = None


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
    Probe("pycocotools",      "pycocotools",      "opengroundingdino", ("trainers",)),
    Probe("matplotlib",       "matplotlib",       "opengroundingdino", ("trainers",)),
    Probe("timm",             "timm",             "opengroundingdino", ("trainers",)),
    Probe("transformers",     "transformers>=4.35,<4.47", "opengroundingdino", ("trainers",), ">=4.35,<4.47"),
    Probe("colorlog",         "colorlog",         "opengroundingdino", ("trainers",)),
    Probe("jsonlines",        "jsonlines",        "opengroundingdino", ("trainers",)),
    # Webdataset — Phase 3 alternative TileStore backend.
    Probe("webdataset",       "webdataset",       "webdataset",        ("data",)),
    Probe("braceexpand",      "braceexpand",      "webdataset",        ("data",)),
]


def _present(module: str) -> bool:
    """Cheap check: is the module importable by `importlib.util.find_spec`?

    Returns False for missing modules and for modules whose spec can't be
    resolved (corrupt install). Does NOT trigger the module's __init__
    so it can't catch version-conflict errors that raise at import time.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ValueError, ModuleNotFoundError, ImportError):
        return False


def _strict_import(module: str) -> tuple[bool, Optional[str]]:
    """Actually import the module in this process; return (ok, error_str).

    Catches version-conflict errors that raise from __init__ — e.g.
    transformers requiring huggingface-hub < 1.0 (failure mode hit in
    the prior agent's host smoke on torch 2.10 + huggingface-hub 1.14).
    """
    try:
        __import__(module)
        return True, None
    except Exception as ex:
        return False, f"{type(ex).__name__}: {ex}"


def _satisfies_version_spec(module: str, spec: Optional[str]) -> tuple[bool, Optional[str]]:
    if not spec:
        return True, None
    try:
        from packaging.specifiers import SpecifierSet
    except Exception as ex:
        return False, f"cannot check version spec {spec!r}: {type(ex).__name__}: {ex}"
    dist_name = "pyyaml" if module == "yaml" else module
    try:
        version = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return False, f"{dist_name} distribution not found"
    ok = SpecifierSet(spec).contains(version, prereleases=True)
    if not ok:
        return False, f"{dist_name}=={version} does not satisfy {spec}"
    return True, None


# Groups for which find_spec isn't sufficient — actually try to import
# the canonical entry points so we catch transitive version conflicts.
_STRICT_IMPORT_GROUPS = {"deimv2", "opengroundingdino"}


def probe_env(*, groups: Optional[Iterable[str]] = None,
              strict_import: bool = False) -> List[Probe]:
    """Return the list of probes that are MISSING in the current env.

    Args:
        groups: subset of probe groups to run; None = all.
        strict_import: when True, do a real ``__import__`` for every
            probe (catches version-conflict raises). When False (default),
            do real imports only for groups in ``_STRICT_IMPORT_GROUPS``
            (deimv2, opengroundingdino) and find_spec for the rest.
    """
    selected_groups = set(groups) if groups is not None else None
    selected = [p for p in PROBES if (selected_groups is None or p.group in selected_groups)]
    missing: List[Probe] = []
    for p in selected:
        use_strict = strict_import or (p.group in _STRICT_IMPORT_GROUPS)
        if use_strict:
            ok, err = _strict_import(p.module)
            if not ok:
                # Annotate the probe with the import error so the CLI
                # can show actionable hints (e.g. "found vX but transformers
                # requires <vY"). We use a copy so PROBES stays clean.
                annotated = Probe(
                    module=p.module, pip_name=p.pip_name, group=p.group,
                    required_in=p.required_in, version_spec=p.version_spec,
                )
                # Stash error on the dataclass via __dict__ since the
                # frozen-ish Probe doesn't have an error field.
                annotated.__dict__["_error"] = err
                missing.append(annotated)
                continue
        else:
            if not _present(p.module):
                missing.append(p)
                continue
        ok, err = _satisfies_version_spec(p.module, p.version_spec)
        if not ok:
            annotated = Probe(
                module=p.module, pip_name=p.pip_name, group=p.group,
                required_in=p.required_in, version_spec=p.version_spec,
            )
            annotated.__dict__["_error"] = err
            missing.append(annotated)
    return missing


def install_missing(missing: List[Probe], *, python_executable: Optional[str] = None) -> int:
    """pip install the missing deps that have a pip_name. Returns exit code (0 OK)."""
    pip_names = [p.pip_name for p in missing if p.pip_name]
    if not pip_names:
        return 0
    py = python_executable or sys.executable
    args = [py, "-m", "pip", "install", *pip_names]
    print(f"[setup_audit] {' '.join(args)}")
    return subprocess.call(args)


def _parse_groups(value) -> List[str]:
    """Tolerate both 'a,b,c' and ['a','b','c'] inputs (scriptconfig smartcast)."""
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        items = str(value).strip("[]").split(",")
    return [g.strip().strip("'\"") for g in items if g.strip().strip("'\"")]


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
    strict_import = scfg.Value(
        False, isflag=True,
        help="real __import__ for every probe (catches version conflicts)",
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        return run(config)


def run(config) -> int:
    groups = _parse_groups(config.groups)
    missing = probe_env(groups=groups, strict_import=bool(config.strict_import))
    if not missing:
        print(f"[check-env] all probes ok for groups={groups}")
        return 0

    print(f"[check-env] {len(missing)} probe(s) failed across groups={groups}:")
    for p in missing:
        err = p.__dict__.get("_error")
        if err:
            # Real-import probe caught a version conflict or broken install.
            print(f"  - {p.module:25s} group={p.group:18s} import error: {err}")
            hint = _hint_for_error(p.module, err)
            if hint:
                print(f"      fix: {hint}")
        else:
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
    still_missing = probe_env(groups=groups, strict_import=bool(config.strict_import))
    if still_missing:
        names = ", ".join(p.module for p in still_missing)
        print(f"[check-env] still missing after install: {names}")
        return 2
    print("[check-env] all probes ok after install.")
    return 0


def _hint_for_error(module: str, err: str) -> Optional[str]:
    """Per-module heuristics for actionable fix hints based on the import error."""
    # transformers requires huggingface-hub < 1.0 in versions <= 4.46;
    # users with HF Hub 1.x get the conflict at import time.
    if module == "transformers" and "huggingface-hub" in err and ("<1.0" in err or "<1, " in err):
        return (
            "pip install -U transformers   "
            "# OR  pip install 'huggingface-hub<1.0'"
        )
    if "huggingface-hub" in err and "<1.0" in err:
        return "pip install 'huggingface-hub<1.0'"
    return None


__cli__ = CheckEnvConfig
