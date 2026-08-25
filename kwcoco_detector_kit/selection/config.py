"""Per-project checkpoint-selection config and plan resolution.

``SelectionConfig`` is the declarative, serializable per-project knob
(spec: docs/planning/checkpoint_selection.md "Config schema"). Defaults
derive from project type via :func:`default_selection_config` and are
**always materialized** into the run's resolved config (write
``resolved.to_jsonable()`` next to the journal) — visible, never magic.

``resolve_plan`` binds the declarative config to one concrete run:
protocols get their ``Param``s (train input size), datasets get content
ids, fingerprints are computed, per-class buckets below the support
floor are auto-disabled (logged), and ``min_epoch_frac`` becomes an
epoch number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from kwcoco_detector_kit.eval.protocols import (
    DatasetBinding,
    EvalProtocol,
    definition_of,
    fingerprint,
    get_protocol,
    resolve_protocol,
)
from kwcoco_detector_kit.selection.boards import BucketSpec
from kwcoco_detector_kit.selection.rerank import Axis

__all__ = [
    "SelectionConfig", "Binding", "ResolvedPlan",
    "default_selection_config", "resolve_plan",
    "RARE_CLASS_SUPPORT_FLOOR",
]

RARE_CLASS_SUPPORT_FLOOR = 50


@dataclass
class SelectionConfig:
    """Declarative per-project selection config (see module docstring)."""
    # what the worker scores every epoch: [{protocol, dataset}]
    inloop: List[Dict[str, str]] = field(default_factory=list)
    # probe construction (when any inloop binding uses dataset: probe)
    probe: Dict[str, Any] = field(default_factory=dict)
    # leaderboards: [{protocol, dataset, metric, k}]
    buckets: List[Dict[str, Any]] = field(default_factory=list)
    # retention: {anchor_top_m}
    retention: Dict[str, Any] = field(default_factory=lambda: {"anchor_top_m": 2})
    # rerank: {axes: [...], policy, primary}
    rerank: Dict[str, Any] = field(default_factory=dict)
    min_epoch_frac: float = 0.1

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "inloop": self.inloop,
            "probe": self.probe,
            "buckets": self.buckets,
            "retention": self.retention,
            "rerank": self.rerank,
            "min_epoch_frac": self.min_epoch_frac,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelectionConfig":
        return cls(
            inloop=list(data.get("inloop") or []),
            probe=dict(data.get("probe") or {}),
            buckets=list(data.get("buckets") or []),
            retention=dict(data.get("retention") or {"anchor_top_m": 2}),
            rerank=dict(data.get("rerank") or {}),
            min_epoch_frac=float(data.get("min_epoch_frac", 0.1)),
        )


def default_selection_config(*, trains_on_tiles: bool) -> SelectionConfig:
    """Project-type-derived default (decided 2026-06-10).

    Tile-cache projects select in the tiled regime (probe-bound) with the
    whole-image lens as a secondary bucket; whole-image projects keep the
    whole-image lens only. Either way the result must be written into the
    run's resolved config — the knob is visible, not magic.
    """
    if trains_on_tiles:
        return SelectionConfig(
            inloop=[
                {"protocol": "true_tiled", "dataset": "probe"},
                {"protocol": "whole_resize", "dataset": "vali"},
            ],
            probe={"source": "vali", "frames": 50, "seed": 0, "empty_frac": 0.1},
            buckets=[
                {"protocol": "true_tiled", "dataset": "probe", "metric": "AP@0.5", "k": 3},
                {"protocol": "whole_resize", "dataset": "vali", "metric": "AP@0.5", "k": 2},
            ],
            retention={"anchor_top_m": 2},
            rerank={
                "axes": [
                    {"protocol": "true_tiled", "dataset": "vali_full", "metric": "AP@0.5"},
                    {"protocol": "whole_resize", "dataset": "vali_full", "metric": "AP@0.5"},
                ],
                "policy": "argmax",
                "primary": {"protocol": "true_tiled", "dataset": "vali_full",
                            "metric": "AP@0.5"},
            },
            min_epoch_frac=0.1,
        )
    return SelectionConfig(
        inloop=[{"protocol": "whole_resize", "dataset": "vali"}],
        buckets=[
            {"protocol": "whole_resize", "dataset": "vali", "metric": "AP@0.5", "k": 1},
        ],
        retention={"anchor_top_m": 1},
        rerank={
            "axes": [
                {"protocol": "whole_resize", "dataset": "vali_full", "metric": "AP@0.5"},
            ],
            "policy": "argmax",
            "primary": {"protocol": "whole_resize", "dataset": "vali_full",
                        "metric": "AP@0.5"},
        },
        min_epoch_frac=0.1,
    )


# ---------------------------------------------------------------------------
# Plan resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Binding:
    """A resolved (protocol, dataset) pair the worker can score."""
    fingerprint: str
    protocol: EvalProtocol            # resolved (no Params)
    dataset: DatasetBinding
    dataset_fpath: str
    label: str

    def definition(self) -> Dict[str, Any]:
        return definition_of(self.protocol, self.dataset)


@dataclass
class ResolvedPlan:
    inloop_bindings: List[Binding]
    rerank_bindings: List[Binding]
    buckets: List[BucketSpec]
    anchor_bucket: Optional[BucketSpec]
    anchor_top_m: int
    min_epoch: int
    rerank_axes: List[Axis]
    rerank_policy: str
    rerank_primary: Axis
    derived_inputs: Dict[str, List[Axis]]
    disabled_buckets: List[str] = field(default_factory=list)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "inloop": [
                {"fingerprint": b.fingerprint, "label": b.label,
                 "dataset_fpath": b.dataset_fpath}
                for b in self.inloop_bindings
            ],
            "rerank_bindings": [
                {"fingerprint": b.fingerprint, "label": b.label,
                 "dataset_fpath": b.dataset_fpath}
                for b in self.rerank_bindings
            ],
            "buckets": [
                {"fingerprint": s.fingerprint, "metric": s.metric,
                 "k": s.k, "label": s.label}
                for s in self.buckets
            ],
            "anchor_bucket": (
                {"fingerprint": self.anchor_bucket.fingerprint,
                 "metric": self.anchor_bucket.metric}
                if self.anchor_bucket else None
            ),
            "anchor_top_m": self.anchor_top_m,
            "min_epoch": self.min_epoch,
            "rerank": {
                "axes": [a.axis_id for a in self.rerank_axes],
                "policy": self.rerank_policy,
                "primary": self.rerank_primary.axis_id,
            },
            "disabled_buckets": self.disabled_buckets,
        }

    def definitions(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for b in [*self.inloop_bindings, *self.rerank_bindings]:
            out[b.fingerprint] = b.definition()
        return out


def _is_per_class_metric(metric: str) -> bool:
    return metric.startswith("ap/")


def resolve_plan(
    config: SelectionConfig,
    *,
    train_input_hw: Tuple[int, int],
    source_window_hw: Optional[Tuple[int, int]] = None,
    dataset_fpaths: Mapping[str, str],     # role -> kwcoco path
    dataset_ids: Mapping[str, str],        # role -> content id
    num_epochs: int,
    class_support: Optional[Mapping[str, Mapping[str, int]]] = None,
    support_floor: int = RARE_CLASS_SUPPORT_FLOOR,
    log=print,
) -> ResolvedPlan:
    """Bind the declarative config to one concrete run.

    ``source_window_hw`` is the sliding-window size in SOURCE pixels for the
    true_tiled protocol. Falls back to ``train_input_hw``, which is right for
    whole-frame-trained models; tile-trained runs must pass their actual tile
    size or the model is measured at a different object scale than it saw.
    """
    class_support = class_support or {}
    window_hw = tuple(int(v) for v in (source_window_hw or train_input_hw))

    def _binding(entry: Mapping[str, str]) -> Binding:
        proto = resolve_protocol(
            get_protocol(str(entry["protocol"])),
            train_input_hw=tuple(int(v) for v in train_input_hw),
            source_window_hw=window_hw,
        )
        role = str(entry["dataset"])
        if role not in dataset_fpaths or role not in dataset_ids:
            raise KeyError(
                f"selection config references dataset role {role!r} but the "
                f"run provided roles {sorted(dataset_fpaths)}"
            )
        ds = DatasetBinding(role=role, dataset_id=str(dataset_ids[role]))
        return Binding(
            fingerprint=fingerprint(proto, ds),
            protocol=proto,
            dataset=ds,
            dataset_fpath=str(dataset_fpaths[role]),
            label=f"{proto.name}.{role}",
        )

    inloop_bindings = [_binding(e) for e in config.inloop]
    by_key = {(b.protocol.name, b.dataset.role): b for b in inloop_bindings}

    # ---- buckets (with rare-class auto-disable) ----
    buckets: List[BucketSpec] = []
    disabled: List[str] = []
    for b in config.buckets:
        key = (str(b["protocol"]), str(b["dataset"]))
        binding = by_key.get(key)
        if binding is None:
            binding = _binding({"protocol": key[0], "dataset": key[1]})
            by_key[key] = binding
            inloop_bindings.append(binding)
        metric = str(b["metric"])
        label = f"{binding.label}.{metric}"
        if _is_per_class_metric(metric):
            cls = metric.split("/", 1)[1]
            support = int(class_support.get(binding.dataset.role, {}).get(cls, 0))
            if support < int(support_floor):
                disabled.append(label)
                log(
                    f"[selection] disabling bucket {label}: class {cls!r} "
                    f"support {support} < floor {support_floor}"
                )
                continue
        buckets.append(BucketSpec(
            fingerprint=binding.fingerprint, metric=metric,
            k=int(b["k"]), label=label,
        ))

    # ---- anchors: top-M of the FIRST declared bucket (the primary
    # in-loop board; spec: anchors on the primary board only) ----
    anchor_bucket = buckets[0] if buckets else None
    anchor_top_m = int(config.retention.get("anchor_top_m", 2))

    # ---- rerank ----
    rerank_cfg = config.rerank or {}
    rerank_bindings: List[Binding] = []
    rb_by_key: Dict[Tuple[str, str], Binding] = {}

    def _rerank_binding(entry):
        key = (str(entry["protocol"]), str(entry["dataset"]))
        if key not in rb_by_key:
            rb_by_key[key] = _binding({"protocol": key[0], "dataset": key[1]})
            rerank_bindings.append(rb_by_key[key])
        return rb_by_key[key]

    axes: List[Axis] = []
    derived_inputs: Dict[str, List[Axis]] = {}
    for entry in rerank_cfg.get("axes") or []:
        if "derived" in entry:
            name = str(entry["derived"])
            inputs = [
                Axis(
                    fingerprint=_rerank_binding(ie).fingerprint,
                    metric=str(ie["metric"]),
                    label=f"{_rerank_binding(ie).label}.{ie['metric']}",
                )
                for ie in entry.get("inputs") or []
            ]
            if not inputs:
                # default composition: every primitive rerank axis declared
                # BEFORE this derived entry (explicit beats implicit, but
                # the common config lists the two primitives first)
                inputs = [a for a in axes if not a.derived]
            derived_inputs[name] = inputs
            axes.append(Axis(derived=name, label=name))
        else:
            binding = _rerank_binding(entry)
            axes.append(Axis(
                fingerprint=binding.fingerprint,
                metric=str(entry["metric"]),
                label=f"{binding.label}.{entry['metric']}",
            ))

    primary_cfg = rerank_cfg.get("primary") or {}
    if "derived" in primary_cfg:
        primary = Axis(derived=str(primary_cfg["derived"]))
    elif primary_cfg:
        pb = _rerank_binding(primary_cfg)
        primary = Axis(fingerprint=pb.fingerprint,
                       metric=str(primary_cfg["metric"]),
                       label=f"{pb.label}.{primary_cfg['metric']}")
    elif axes:
        primary = axes[0]
    else:
        raise ValueError("selection config has no rerank axes / primary")

    min_epoch = int(round(float(config.min_epoch_frac) * int(num_epochs)))

    return ResolvedPlan(
        inloop_bindings=inloop_bindings,
        rerank_bindings=rerank_bindings,
        buckets=buckets,
        anchor_bucket=anchor_bucket,
        anchor_top_m=anchor_top_m,
        min_epoch=min_epoch,
        rerank_axes=axes,
        rerank_policy=str(rerank_cfg.get("policy", "argmax")),
        rerank_primary=primary,
        derived_inputs=derived_inputs,
        disabled_buckets=disabled,
    )
