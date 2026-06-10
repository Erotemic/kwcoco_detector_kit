"""Canonical eval protocols and score fingerprints.

Spec: ``docs/planning/checkpoint_selection.md``.

Every score in the selection system is a ``(protocol, dataset, subject)``
triple. This module owns the first two identities and their composition:

- :class:`EvalProtocol` — *which procedure measured* (the lens). Protocols
  are **parameterized families**: registry constants may hold
  :class:`Param` placeholders (e.g. the sliding window tracks the train
  input size) that :func:`resolve_protocol` replaces with concrete values.
  Only fully-resolved protocols can be fingerprinted, so ``tiled@640`` and
  ``tiled@1280`` are distinct comparison spaces by construction.
- :class:`DatasetBinding` — *what it was measured against* (the target).
- :func:`fingerprint` — ``hash(protocol_id ⊕ dataset_id)``. Two
  ScoreRecords are comparable iff their fingerprints match; everything
  else (the subject's epoch/ckpt_hash, git shas, hosts, timestamps) is
  circumstance, stored verbatim but never hashed.

The class-filter holds the *rule* (``exclude_distractors``), never the
*list* — the distractor list belongs to the dataset/scheme side
(``class_schemes.yaml`` ``distractor_classes``); kit code never names a
project's species.

Protocols do **not** embed a metric: one eval pass emits the full
measures dict (``AP@0.5``, ``ap/<class>``, ...), addressed as
``(fingerprint, metric_key)``.

``version`` on a protocol is the curated *eval-protocol version*: bump it
only when something eval-affecting changes (regime semantics, the DEIMv2
patch queue, probe construction). Raw git shas live in circumstances.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

__all__ = [
    "Param", "Resize", "SlidingWindow", "ClassFilter", "EvalProtocol",
    "DatasetBinding", "Subject", "ScoreRecord",
    "resolve_protocol", "protocol_id", "fingerprint",
    "canonical_json", "short_hash", "dataset_id_of_file",
    "TRUE_TILED_V1", "WHOLE_RESIZE_V1", "PROTOCOLS", "get_protocol",
]


# ---------------------------------------------------------------------------
# Canonical hashing helpers
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for every identity hash in this module."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def dataset_id_of_file(fpath: Union[str, Path], n: int = 12) -> str:
    """Content hash of a dataset file (chunked sha256).

    Caveat (see [[feedback-kwcoco-bakes-absolute-paths]]): kwcoco bundles
    bake absolute image paths, so the same logical dataset re-rooted on
    another host hashes differently. That is honest — the bytes differ —
    but cross-host comparability needs the *same* bundle file, not a
    re-written copy.
    """
    h = hashlib.sha256()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


# ---------------------------------------------------------------------------
# Parameterized regimes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Param:
    """Placeholder in a protocol family, resolved per run.

    ``key`` names the resolution-time parameter (e.g. ``train_input_hw``).
    """
    key: str


def _is_unresolved(value: Any) -> bool:
    if isinstance(value, Param):
        return True
    if isinstance(value, (list, tuple)):
        return any(_is_unresolved(v) for v in value)
    return False


def _resolve_value(value: Any, params: Dict[str, Any]) -> Any:
    if isinstance(value, Param):
        if value.key not in params:
            raise KeyError(
                f"protocol param {value.key!r} not provided; have {sorted(params)}"
            )
        return _resolve_value(params[value.key], params)
    if isinstance(value, (list, tuple)):
        return tuple(_resolve_value(v, params) for v in value)
    return value


@dataclass(frozen=True)
class Resize:
    """Whole-image regime: resize each full image to ``size`` (H, W)."""
    size: Any  # (H, W) tuple or Param
    kind: str = field(default="resize", init=False)


@dataclass(frozen=True)
class SlidingWindow:
    """Tiled regime: native-res windows + cross-window NMS merge."""
    window: Any  # (H, W) tuple or Param
    overlap: float = 0.25
    nms_iou: float = 0.5
    kind: str = field(default="sliding_window", init=False)


@dataclass(frozen=True)
class ClassFilter:
    """The *rule* half of class filtering (the list rides with the scheme)."""
    class_agnostic: bool = False
    exclude_distractors: bool = False


@dataclass(frozen=True)
class EvalProtocol:
    """A versioned measurement procedure. See module docstring."""
    name: str
    version: int
    regime: Union[Resize, SlidingWindow]
    class_filter: ClassFilter = ClassFilter()
    score_thresh: float = 0.001

    def is_resolved(self) -> bool:
        return not any(
            _is_unresolved(getattr(self.regime, f.name))
            for f in dataclasses.fields(self.regime)
        )

    def to_jsonable(self) -> Dict[str, Any]:
        if not self.is_resolved():
            raise ValueError(
                f"protocol {self.name!r} has unresolved Params; call "
                "resolve_protocol() before hashing"
            )

        def _enc(v):
            if isinstance(v, tuple):
                return [_enc(x) for x in v]
            return v

        regime = {k: _enc(v) for k, v in dataclasses.asdict(self.regime).items()}
        return {
            "name": self.name,
            "version": self.version,
            "regime": regime,
            "class_filter": dataclasses.asdict(self.class_filter),
            "score_thresh": self.score_thresh,
        }


def resolve_protocol(protocol: EvalProtocol, **params: Any) -> EvalProtocol:
    """Replace every :class:`Param` placeholder with a concrete value."""
    regime = protocol.regime
    kwargs = {}
    for f in dataclasses.fields(regime):
        if f.init:
            kwargs[f.name] = _resolve_value(getattr(regime, f.name), params)
    resolved_regime = type(regime)(**kwargs)
    resolved = dataclasses.replace(protocol, regime=resolved_regime)
    if not resolved.is_resolved():
        raise ValueError(f"protocol {protocol.name!r} still unresolved after params")
    return resolved


def protocol_id(protocol: EvalProtocol) -> str:
    """Human-prefixed stable id of a *resolved* protocol."""
    j = protocol.to_jsonable()
    return f"{protocol.name}.v{protocol.version}.{short_hash(canonical_json(j))}"


# ---------------------------------------------------------------------------
# Dataset / subject / record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetBinding:
    """The measurement target. ``dataset_id`` is content-derived
    (kwcoco file hash or probe manifest hash).

    Identity is **purely content**: only ``dataset_id`` is hashed. The
    ``role`` is informational (drill-down/labels) — two roles bound to the
    same bytes are the same comparison space, so e.g. in-loop scores on
    the full vali file automatically satisfy a re-rank axis bound to the
    same file."""
    role: str          # vali | probe | vali_full | test | ...
    dataset_id: str
    n_images: Optional[int] = None   # informational; not hashed

    def to_jsonable(self) -> Dict[str, Any]:
        return {"dataset_id": self.dataset_id}


@dataclass(frozen=True)
class Subject:
    """Who was measured. Never part of the fingerprint."""
    weights_kind: str   # "ema" | "raw"
    epoch: int
    ckpt_hash: Optional[str] = None


@dataclass
class ScoreRecord:
    """One eval pass: every metric it emitted, bound to its identity."""
    fingerprint: str
    subject: Subject
    measures: Dict[str, float]
    circumstances: Dict[str, Any] = field(default_factory=dict)


def fingerprint(protocol: EvalProtocol, dataset: DatasetBinding) -> str:
    """``hash(protocol_id ⊕ dataset_id)`` — the comparison-space key."""
    payload = {
        "protocol": protocol.to_jsonable(),
        "dataset": dataset.to_jsonable(),
    }
    return short_hash(canonical_json(payload))


def definition_of(protocol: EvalProtocol, dataset: DatasetBinding) -> Dict[str, Any]:
    """The full un-hashed definition stored in the content-addressed
    ``fingerprint -> definition`` registry (drill-down without
    reconstruction)."""
    return {
        "protocol": protocol.to_jsonable(),
        "protocol_id": protocol_id(protocol),
        "dataset": {**dataset.to_jsonable(), "role": dataset.role,
                    "n_images": dataset.n_images},
    }


# ---------------------------------------------------------------------------
# Canonical protocol registry (versioned constants; reviewed code changes)
# ---------------------------------------------------------------------------

TRUE_TILED_V1 = EvalProtocol(
    name="true_tiled",
    version=1,
    regime=SlidingWindow(window=Param("train_input_hw"), overlap=0.25, nms_iou=0.5),
    class_filter=ClassFilter(class_agnostic=True, exclude_distractors=True),
    score_thresh=0.001,
)

WHOLE_RESIZE_V1 = EvalProtocol(
    name="whole_resize",
    version=1,
    regime=Resize(size=Param("train_input_hw")),
    class_filter=ClassFilter(class_agnostic=True, exclude_distractors=True),
    score_thresh=0.001,
)

PROTOCOLS: Dict[str, EvalProtocol] = {
    p.name: p for p in [TRUE_TILED_V1, WHOLE_RESIZE_V1]
}


def get_protocol(name: str) -> EvalProtocol:
    try:
        return PROTOCOLS[name]
    except KeyError:
        raise KeyError(
            f"unknown eval protocol {name!r}; known: {sorted(PROTOCOLS)}"
        ) from None
