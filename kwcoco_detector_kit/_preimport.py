"""
Ordered pre-imports for binary-extension modules.

GDAL, pyproj, geopandas, rasterio, fiona, and torch each ship with native
libraries that conflict with each other when loaded in the wrong order.
The most common symptom on this project is the
"DelayedLoad may not be efficient without gdal" warning from
delayed_image, accompanied by 30-100x slower image reads during the
round-loop's negative-tile mining pass.

This is a port of the equivalent pre-import scaffold in geowatch (see
``geowatch.__init__``). Same env-var protocol with KCD_ prefix:

  KCD_PREIMPORT=auto           # (default) pick a known-good profile
  KCD_PREIMPORT=variant1       # gdal-first, broadest compatibility
  KCD_PREIMPORT=variant2       # CI-friendly subset
  KCD_PREIMPORT=variant3       # delay GDAL import
  KCD_PREIMPORT=0              # skip pre-imports entirely
  KCD_PREIMPORT=pyproj,gdal    # explicit comma-separated module list

``KCD_PREIMPORT_DEBUG=1`` prints what the scaffold did.
"""
from __future__ import annotations

import os
import sys
import warnings


PREIMPORT_VARIANTS = {
    "variant1": ["geopandas", "pyproj", "gdal"],   # broadest compat
    "variant2": ["pyproj", "gdal"],                # CI-friendly
    "variant3": ["geopandas", "pyproj"],           # delay gdal
    "none":     [],
    "0":        [],
}


def _debug_enabled() -> bool:
    return bool(os.environ.get("KCD_PREIMPORT_DEBUG", ""))


def _is_fast_cli() -> bool:
    """Short-circuit pre-imports for cli tools where startup time matters.

    Mirrors geowatch's heuristic — argcomplete, --help, finish_install,
    and the kit's own check-env probe (which has its own import-order
    semantics).
    """
    if os.environ.get("_ARGCOMPLETE", ""):
        return True
    argv = sys.argv or []
    if not argv:
        return False
    if any(a in ("--help", "-h") for a in argv):
        return True
    if any("finish_install" in a for a in argv):
        return True
    # Bare `kwcoco-detector-kit check-env` should still pre-import so the
    # runtime probe can verify the import order it relies on.
    return False


def _import_one(modname: str) -> None:
    if _debug_enabled():
        print(f"[kcd._preimport] import {modname}")
    if modname == "gdal":
        try:
            from osgeo import gdal as _gdal
        except ImportError:
            if _debug_enabled():
                print("[kcd._preimport] osgeo.gdal not installed; skip")
            return
        if not getattr(_gdal, "_UserHasSpecifiedIfUsingExceptions",
                       lambda: False)():
            if _debug_enabled():
                print("[kcd._preimport] gdal.UseExceptions()")
            _gdal.UseExceptions()
    elif modname == "pyproj":
        try:
            import pyproj  # noqa: F401
            from pyproj import CRS  # noqa: F401
        except ImportError:
            if _debug_enabled():
                print("[kcd._preimport] pyproj not installed; skip")
    elif modname == "geopandas":
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    ".*is incompatible with the GEOS version "
                    "PyGEOS was compiled with.*",
                )
                import geopandas  # noqa: F401
        except ImportError:
            if _debug_enabled():
                print("[kcd._preimport] geopandas not installed; skip")
    elif modname == "rasterio":
        try:
            import rasterio  # noqa: F401
        except ImportError:
            if _debug_enabled():
                print("[kcd._preimport] rasterio not installed; skip")
    elif modname == "fiona":
        try:
            import fiona  # noqa: F401
        except ImportError:
            if _debug_enabled():
                print("[kcd._preimport] fiona not installed; skip")
    elif modname == "torch":
        try:
            import torch  # noqa: F401
        except ImportError:
            if _debug_enabled():
                print("[kcd._preimport] torch not installed; skip")
    elif modname == "numpy":
        try:
            import numpy  # noqa: F401
        except ImportError:
            if _debug_enabled():
                print("[kcd._preimport] numpy not installed; skip")
    else:
        if _debug_enabled():
            print(f"[kcd._preimport] unknown module {modname!r}; skip")


def execute_ordered_preimports() -> None:
    """Read ``KCD_PREIMPORT`` and pre-import the listed modules in order."""
    # USE_PYGEOS=0 is the geopandas-2.x default but explicit beats implicit
    # when the kit ships on hosts with older geopandas. Harmless if unset.
    os.environ.setdefault("USE_PYGEOS", "0")

    raw = os.environ.get("KCD_PREIMPORT", "auto")
    if _debug_enabled():
        print(f"[kcd._preimport] KCD_PREIMPORT={raw!r}")

    if raw in ("0", "none", "false", "no", ""):
        if _debug_enabled():
            print("[kcd._preimport] disabled by env")
        return

    if raw == "auto":
        if _is_fast_cli():
            if _debug_enabled():
                print("[kcd._preimport] fast-cli context; skip pre-imports")
            return
        mods = PREIMPORT_VARIANTS["variant1"]
    elif raw in PREIMPORT_VARIANTS:
        mods = PREIMPORT_VARIANTS[raw]
    else:
        mods = [m.strip() for m in raw.split(",") if m.strip()]

    for m in mods:
        try:
            _import_one(m)
        except Exception as ex:
            # Pre-import failures are non-fatal -- the kit still works
            # without the optional geo modules. Warn so the user sees it.
            warnings.warn(
                f"[kcd._preimport] {m} pre-import failed: "
                f"{type(ex).__name__}: {ex}"
            )
