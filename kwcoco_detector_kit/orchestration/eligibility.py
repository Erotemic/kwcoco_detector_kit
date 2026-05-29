"""
Eligibility manifest — aggregate per-cell sweep outputs into a single
TSV/JSON, run the four-class state machine, and select the highest-AP
candidate that satisfies the deployment gates.

Four eligibility classes
------------------------
::

  NOT_READY           host pipeline incomplete — missing checkpoint, ONNX,
                      eval metrics, OR no desktop benchmark JSON. By default
                      a missing benchmark also lands here; override with
                      --allow_missing_desktop_bench.

  HOST_PROMISING      passes host gates: trained, exported, eval AP measured,
                      AND desktop CPU mean <= --max_desktop_ms. Worth taking
                      to the deploy target for real measurement.

  DEPLOY_ELIGIBLE     HOST_PROMISING AND --device_index supplied AND the
                      candidate's device fps >= --min_device_fps. Only this
                      class is safe to call "eligible to deploy".

  DEPLOY_INELIGIBLE   desktop mean > gate, OR device data supplied but fps
                      below gate.

The class names were renamed from the prior project's ``PHONE_*`` so the
kit is domain-neutral. Semantics are preserved: ``candidate_kind=smoke``
candidates are excluded from winner-selection by default (the kit's
mock_tiny CPU detector should not accidentally win a real sweep).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import scriptconfig as scfg


# Public state names
NOT_READY = "NOT_READY"
HOST_PROMISING = "HOST_PROMISING"
DEPLOY_ELIGIBLE = "DEPLOY_ELIGIBLE"
DEPLOY_INELIGIBLE = "DEPLOY_INELIGIBLE"
ELIGIBILITY_CLASSES = (NOT_READY, HOST_PROMISING, DEPLOY_ELIGIBLE, DEPLOY_INELIGIBLE)


MANIFEST_FIELDS = [
    "candidate_id",
    "variant",
    "candidate_kind",
    "export_input_h",
    "export_input_w",
    "train_resolution_policy",
    "requested_train_resolution_min",
    "requested_train_resolution_max",
    "train_resolution_min",
    "train_resolution_max",
    "train_resolution_choices",
    "tile_training_policy",
    "checkpoint_path",
    "onnx_path",
    "modelspec_path",
    "test_ap",
    "desktop_latency_ms_p50",
    "desktop_latency_ms_mean",
    "desktop_latency_ms_p99",
    "desktop_eligible",
    "device_latency_ms",
    "device_fps",
    "device_eligible",
    "eligibility_class",
    "device_model_id",
    "status",
    "reasons",
]


def _infer_candidate_kind(policy: dict) -> str:
    """Pull ``candidate_kind`` from policy.json, fall back to variant prefix.

    ``mock_tiny*`` variants are smoke artifacts. Anything else is treated as
    a real deployable detector.
    """
    kind = policy.get("candidate_kind")
    if kind in ("smoke", "real"):
        return kind
    variant = policy.get("variant", "")
    if variant.startswith("mock_tiny") or variant.startswith("v4_mock"):
        return "smoke"
    return "real"


@dataclass
class Row:
    candidate_id: str = ""
    variant: str = ""
    candidate_kind: str = ""
    export_input_h: Optional[int] = None
    export_input_w: Optional[int] = None
    train_resolution_policy: str = ""
    requested_train_resolution_min: Optional[int] = None
    requested_train_resolution_max: Optional[int] = None
    train_resolution_min: Optional[int] = None
    train_resolution_max: Optional[int] = None
    train_resolution_choices: str = ""
    tile_training_policy: str = ""
    checkpoint_path: str = ""
    onnx_path: str = ""
    modelspec_path: str = ""
    test_ap: Optional[float] = None
    desktop_latency_ms_p50: Optional[float] = None
    desktop_latency_ms_mean: Optional[float] = None
    desktop_latency_ms_p99: Optional[float] = None
    desktop_eligible: str = ""
    device_latency_ms: Optional[float] = None
    device_fps: Optional[float] = None
    device_eligible: str = "TODO"
    eligibility_class: str = ""
    device_model_id: str = ""
    status: str = ""
    reasons: list = field(default_factory=list)


class EligibilityConfig(scfg.DataConfig):
    """Aggregate sweep outputs + run the eligibility state machine."""

    sweep_index = scfg.Value(None, help="TSV index produced by pareto_sweep.py; takes precedence over --auto")
    auto = scfg.Value(False, isflag=True, help="Discover candidates under $KCD_ROOT/runs/")
    kcd_root = scfg.Value(
        None,
        help="root dir to scan in --auto mode (defaults to $KCD_ROOT)",
    )
    out = scfg.Value(None, help="output TSV path")
    out_json = scfg.Value(None, help="optional JSON output (richer)")

    max_desktop_ms = scfg.Value(80.0, help="desktop CPU mean ms gate (proxy for on-device latency)")
    min_device_fps = scfg.Value(10.0, help="on-device FPS gate (only when --device_index supplied)")
    device_index = scfg.Value(
        None,
        help="optional TSV with columns candidate_id\\tlatency_ms\\tfps from a real-device benchmark",
    )
    allow_missing_desktop_bench = scfg.Value(
        False, isflag=True,
        help="treat candidates with no desktop bench as HOST_PROMISING (default: NOT_READY)",
    )
    include_smoke_models = scfg.Value(
        False, isflag=True,
        help="include candidate_kind=smoke in the winner pool (default: real only)",
    )
    smoke_only = scfg.Value(
        False, isflag=True,
        help="restrict the winner pool to smoke candidates (CI usage)",
    )
    print_winner = scfg.Value(True, isflag=True, help="print the eligible winner to stdout")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------


def _iter_candidates_from_index(index_fpath: Path):
    with open(str(index_fpath), "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("status", "").strip() in ("fail", "skip"):
                continue
            yield row.get("candidate_id", "").strip(), Path(row.get("workdir", "").strip())


def _iter_candidates_from_auto(kcd_root: Path):
    runs_root = Path(kcd_root) / "runs"
    if not runs_root.exists():
        return
    for sub in sorted(runs_root.iterdir()):
        if not sub.is_dir():
            continue
        policy_fpath = sub / "policy.json"
        if not policy_fpath.exists():
            continue
        try:
            cid = json.loads(policy_fpath.read_text()).get("candidate_id") or sub.name
        except Exception:
            cid = sub.name
        yield cid, sub


# ---------------------------------------------------------------------------
# Per-candidate readers
# ---------------------------------------------------------------------------


def _load_policy(workdir: Path) -> dict:
    fpath = workdir / "policy.json"
    if not fpath.exists():
        return {}
    try:
        return json.loads(fpath.read_text())
    except Exception as ex:
        return {"_policy_load_error": str(ex)}


def _find_checkpoint(workdir: Path) -> str:
    for cand in ("best_stg2.pth", "best_stg1.pth", "last.pth"):
        p = workdir / cand
        if p.exists():
            return str(p)
    epochs = sorted(workdir.glob("checkpoint*.pth"))
    return str(epochs[-1]) if epochs else ""


def _find_onnx_and_modelspec(workdir: Path):
    export_dpath = workdir / "export"
    if not export_dpath.exists():
        return "", ""
    onnx_files = sorted(export_dpath.glob("*.onnx"))
    if not onnx_files:
        return "", ""
    onnx = onnx_files[0]
    modelspec = onnx.with_suffix(".modelspec.json")
    return str(onnx), (str(modelspec) if modelspec.exists() else "")


def _find_eval_ap(kcd_root: Path, candidate_id: str,
                  workdir: Optional[Path] = None) -> Optional[float]:
    """Locate detect_metrics.json under <kcd_root>/eval/<candidate_id>/eval/.

    Falls back to inferring the root from <workdir>/../../eval/<candidate_id>/
    if <kcd_root> was wrong (e.g. KCD_ROOT unset when invoked).

    Selection precedence: prefer the most-pruned sidecar metrics file when
    present (e.g. ``detect_metrics.northern_fur_seal.json``) so model
    selection runs on the corrected AP rather than the with-NFS AP.
    The original ``detect_metrics.json`` stays on disk as a diagnostic.
    """
    eval_dir = kcd_root / "eval" / candidate_id / "eval"
    if not eval_dir.exists() and workdir is not None:
        eval_dir = workdir.parent.parent / "eval" / candidate_id / "eval"
    if not eval_dir.exists():
        return None

    # Prefer sidecar metrics files (detect_metrics.<excluded>.json) over
    # the with-everything detect_metrics.json. Tie-break alphabetically
    # so the choice is deterministic across runs.
    sidecars = sorted(p for p in eval_dir.glob("detect_metrics.*.json"))
    metrics_fpath = sidecars[0] if sidecars else (eval_dir / "detect_metrics.json")
    if not metrics_fpath.exists():
        return None
    try:
        data = json.loads(metrics_fpath.read_text())
    except Exception:
        return None

    def find_ap(node):
        if isinstance(node, dict):
            if "nocls_measures" in node and isinstance(node["nocls_measures"], dict):
                v = node["nocls_measures"].get("ap")
                if v is not None:
                    return float(v)
            for v in node.values():
                r = find_ap(v)
                if r is not None:
                    return r
        elif isinstance(node, list):
            for v in node:
                r = find_ap(v)
                if r is not None:
                    return r
        return None

    return find_ap(data)


def _find_bench_metrics(workdir: Path) -> Dict[str, Optional[float]]:
    export_dpath = workdir / "export"
    if not export_dpath.exists():
        return {}
    candidates = sorted(export_dpath.glob("*.bench.json"))
    if not candidates:
        return {}
    try:
        bench = json.loads(candidates[0].read_text())
    except Exception:
        return {}
    timings = bench.get("timings_ms", [])
    if not timings:
        return {"mean_ms": bench.get("mean_ms")}
    sorted_t = sorted(timings)

    def pct(p):
        idx = max(0, min(len(sorted_t) - 1, int(round(p / 100.0 * (len(sorted_t) - 1)))))
        return sorted_t[idx]

    return {
        "mean_ms": bench.get("mean_ms", sum(timings) / len(timings)),
        "p50_ms": pct(50),
        "p99_ms": pct(99),
    }


def _device_model_id(policy: dict) -> str:
    """Canonical model ID for cross-device tracking. Mirrors prior project."""
    v = policy.get("variant", "")
    h = policy.get("export_input_h", "")
    w = policy.get("export_input_w", "")
    pol = policy.get("train_resolution_policy", "")
    return f"{v}-h{h}w{w}-{pol}"


def _load_device_index(fpath) -> Dict[str, Dict[str, Optional[float]]]:
    if fpath is None:
        return {}
    out: Dict[str, Dict[str, Optional[float]]] = {}
    with open(str(fpath), "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cid = row.get("candidate_id", "").strip()
            if not cid:
                continue
            out[cid] = {
                "latency_ms": _safe_float(row.get("latency_ms")),
                "fps": _safe_float(row.get("fps")),
            }
    return out


def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "", "None") else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core: classify a single row
# ---------------------------------------------------------------------------


def classify_row(
    row: Row,
    *,
    max_desktop_ms: float,
    allow_missing_desktop_bench: bool,
) -> Row:
    """Run the four-class state machine in-place on a Row."""
    reasons: List[str] = list(row.reasons)

    # Host gate
    if not row.checkpoint_path:
        row.status = "no_checkpoint"
        reasons.append("no .pth in workdir")
    elif not row.onnx_path:
        row.status = "no_onnx"
        reasons.append("no exported .onnx")
    elif row.test_ap is None:
        row.status = "no_eval"
        reasons.append("no test detect_metrics.json")
    else:
        row.status = "ok"

    # Desktop proxy gate
    mean_ms = row.desktop_latency_ms_mean
    if mean_ms is None:
        row.desktop_eligible = "missing"
        reasons.append("no desktop benchmark")
    elif mean_ms <= float(max_desktop_ms):
        row.desktop_eligible = "yes"
    else:
        row.desktop_eligible = "no"
        reasons.append(f"desktop mean {mean_ms:.1f}ms > {max_desktop_ms}ms")

    # Class assignment
    if row.status != "ok":
        row.eligibility_class = NOT_READY
    elif row.desktop_eligible == "missing" and not allow_missing_desktop_bench:
        row.eligibility_class = NOT_READY
    elif row.desktop_eligible == "no":
        row.eligibility_class = DEPLOY_INELIGIBLE
    elif row.device_eligible == "no":
        row.eligibility_class = DEPLOY_INELIGIBLE
    elif row.device_eligible == "yes":
        row.eligibility_class = DEPLOY_ELIGIBLE
    else:
        row.eligibility_class = HOST_PROMISING

    row.reasons = reasons
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(config):
    kcd_root = Path(
        config.kcd_root or os.environ.get("KCD_ROOT")
        or (Path.home() / "data" / "kcd")
    )
    device_index = _load_device_index(config.device_index)

    if config.sweep_index:
        cand_iter = list(_iter_candidates_from_index(Path(config.sweep_index)))
    elif config.auto:
        cand_iter = list(_iter_candidates_from_auto(kcd_root))
    else:
        raise SystemExit("Either --sweep_index or --auto is required")

    rows: List[Row] = []
    for candidate_id, workdir in cand_iter:
        if not workdir.exists():
            print(f"  skip: {candidate_id} workdir missing ({workdir})", file=sys.stderr)
            continue

        policy = _load_policy(workdir)
        if not candidate_id:
            candidate_id = policy.get("candidate_id", workdir.name)

        ckpt = _find_checkpoint(workdir)
        onnx, modelspec = _find_onnx_and_modelspec(workdir)
        ap = _find_eval_ap(kcd_root, candidate_id, workdir=workdir)
        bench = _find_bench_metrics(workdir)
        scales = policy.get("effective_train_scales", [])

        row = Row(
            candidate_id=candidate_id,
            variant=policy.get("variant", ""),
            candidate_kind=_infer_candidate_kind(policy),
            export_input_h=policy.get("export_input_h"),
            export_input_w=policy.get("export_input_w"),
            train_resolution_policy=policy.get("train_resolution_policy", ""),
            requested_train_resolution_min=policy.get("requested_train_resolution_min"),
            requested_train_resolution_max=policy.get("requested_train_resolution_max"),
            train_resolution_min=policy.get("effective_train_scale_min"),
            train_resolution_max=policy.get("effective_train_scale_max"),
            train_resolution_choices=",".join(str(s) for s in scales),
            tile_training_policy=policy.get("tile_training_policy", ""),
            checkpoint_path=ckpt,
            onnx_path=onnx,
            modelspec_path=modelspec,
            test_ap=ap,
            desktop_latency_ms_p50=bench.get("p50_ms"),
            desktop_latency_ms_mean=bench.get("mean_ms"),
            desktop_latency_ms_p99=bench.get("p99_ms"),
            device_model_id=_device_model_id(policy),
        )

        # Device gate (when supplied)
        dev = device_index.get(candidate_id)
        if dev is not None:
            row.device_latency_ms = dev.get("latency_ms")
            row.device_fps = dev.get("fps")
            if row.device_fps is not None and row.device_fps >= float(config.min_device_fps):
                row.device_eligible = "yes"
            else:
                row.device_eligible = "no"

        classify_row(
            row,
            max_desktop_ms=float(config.max_desktop_ms),
            allow_missing_desktop_bench=bool(config.allow_missing_desktop_bench),
        )
        rows.append(row)

    # Write outputs
    if config.out:
        out_fpath = Path(str(config.out))
        out_fpath.parent.mkdir(parents=True, exist_ok=True)
        with out_fpath.open("w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(MANIFEST_FIELDS)
            for row in rows:
                d = asdict(row)
                d["reasons"] = "; ".join(row.reasons)
                w.writerow([d.get(k, "") if d.get(k) is not None else "" for k in MANIFEST_FIELDS])
        print(f"wrote {out_fpath}")

    if config.out_json:
        Path(str(config.out_json)).parent.mkdir(parents=True, exist_ok=True)
        Path(str(config.out_json)).write_text(json.dumps([asdict(r) for r in rows], indent=2))

    # Summary table
    print()
    print(
        "candidate_id".ljust(50),
        "AP".rjust(7),
        "desk_ms".rjust(8),
        "desk_ok".rjust(8),
        "dev_fps".rjust(8),
        "dev_ok".rjust(6),
        "class".rjust(20),
        "status",
    )
    print("-" * 130)
    for row in sorted(rows, key=lambda r: (r.test_ap or -1), reverse=True):
        ap = "-" if row.test_ap is None else f"{row.test_ap:.3f}"
        ms = "-" if row.desktop_latency_ms_mean is None else f"{row.desktop_latency_ms_mean:.1f}"
        fps = "-" if row.device_fps is None else f"{row.device_fps:.1f}"
        print(
            row.candidate_id.ljust(50),
            ap.rjust(7),
            ms.rjust(8),
            row.desktop_eligible.rjust(8),
            fps.rjust(8),
            row.device_eligible.rjust(6),
            row.eligibility_class.rjust(20),
            row.status,
        )

    if not config.print_winner:
        return rows

    # Winner selection
    if bool(config.smoke_only):
        kind_filter = lambda r: r.candidate_kind == "smoke"
        kind_label = "SMOKE_ONLY"
    elif bool(config.include_smoke_models):
        kind_filter = lambda r: r.candidate_kind in ("real", "smoke")
        kind_label = "real + smoke"
    else:
        kind_filter = lambda r: r.candidate_kind == "real"
        kind_label = "real"

    n_excluded = sum(
        1 for r in rows
        if r.candidate_kind == "smoke" and not bool(config.include_smoke_models)
        and not bool(config.smoke_only)
    )
    if n_excluded:
        print(
            f"\nwinner pool: candidate_kind={kind_label} ({n_excluded} smoke "
            "candidate(s) excluded; --include_smoke_models to include)"
        )

    promising = [
        r for r in rows
        if r.eligibility_class == HOST_PROMISING
        and r.test_ap is not None
        and kind_filter(r)
    ]
    eligible = [
        r for r in rows
        if r.eligibility_class == DEPLOY_ELIGIBLE
        and r.test_ap is not None
        and kind_filter(r)
    ]

    if promising:
        winner = max(promising, key=lambda r: r.test_ap)
        _print_winner("host-promising winner", winner)
    else:
        print(f"\nno HOST_PROMISING candidate of kind={kind_label} yet")

    if device_index:
        if eligible:
            _print_winner("deploy-eligible winner", max(eligible, key=lambda r: r.test_ap))
        else:
            print(f"\nno DEPLOY_ELIGIBLE candidate of kind={kind_label}")
    else:
        print("\nno deploy-eligible winner can be selected without --device_index")

    return rows


def _print_winner(label: str, row: Row):
    print()
    print(f"=== {label} ===")
    print(f"  candidate_id          {row.candidate_id}")
    print(f"  variant               {row.variant}")
    print(f"  candidate_kind        {row.candidate_kind}")
    print(f"  export size           {row.export_input_h} x {row.export_input_w}")
    print(
        f"  train policy          {row.train_resolution_policy}"
        f"  scales={row.train_resolution_min}..{row.train_resolution_max}"
    )
    if row.test_ap is not None:
        print(f"  test AP               {row.test_ap:.3f}")
    if row.desktop_latency_ms_mean is not None:
        print(f"  desktop CPU mean      {row.desktop_latency_ms_mean:.1f} ms")
    if row.device_fps is not None:
        print(f"  device fps            {row.device_fps:.1f}")
    print(f"  device_model_id       {row.device_model_id}")
    print(f"  onnx                  {row.onnx_path}")
    print(f"  modelspec             {row.modelspec_path}")


__cli__ = EligibilityConfig
