"""Shared detector-truth semantics.

Detector datasets often contain more annotation categories than the detector is
asked to predict.  This module keeps three concepts separate:

* ``target``: positive detector supervision;
* ``background``: known non-target / distractor truth that remains background;
* ``ignore``: uncertain truth that must not be silently trained as background.

The abstraction is intentionally backend- and dataset-agnostic.  Dataset-specific
names belong in campaign configuration, not in KDK.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


_VALID_NON_TARGET_POLICIES = {"background", "ignore", "error"}
_VALID_UNCATEGORIZED_POLICIES = {"background", "ignore", "error"}


def _coerce_names(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = value
    return tuple(str(item).strip() for item in values if str(item).strip())


@dataclass(frozen=True)
class TruthSemantics:
    """Policy that maps source annotations to detector supervision roles."""

    target_categories: tuple[str, ...]
    ignore_categories: tuple[str, ...] = ()
    uncategorized_annotation_policy: str = "ignore"
    default_non_target_policy: str = "background"
    unclassified_category_policy: str = "ignore"

    def __post_init__(self):
        overlap = set(self.target_categories) & set(self.ignore_categories)
        if overlap:
            raise ValueError(
                "categories cannot be both target and ignore: "
                f"{sorted(overlap)!r}"
            )
        if self.default_non_target_policy not in _VALID_NON_TARGET_POLICIES:
            raise ValueError(
                "default_non_target_policy must be one of "
                f"{sorted(_VALID_NON_TARGET_POLICIES)!r}"
            )
        if self.unclassified_category_policy not in _VALID_NON_TARGET_POLICIES:
            raise ValueError(
                "unclassified_category_policy must be one of "
                f"{sorted(_VALID_NON_TARGET_POLICIES)!r}"
            )
        if self.uncategorized_annotation_policy not in _VALID_UNCATEGORIZED_POLICIES:
            raise ValueError(
                "uncategorized_annotation_policy must be one of "
                f"{sorted(_VALID_UNCATEGORIZED_POLICIES)!r}"
            )

    @classmethod
    def coerce(
        cls,
        *,
        target_categories: Sequence[str] | str,
        ignore_categories: Sequence[str] | str | None = None,
        uncategorized_annotation_policy: str = "ignore",
        default_non_target_policy: str = "background",
        unclassified_category_policy: str = "ignore",
    ) -> "TruthSemantics":
        return cls(
            target_categories=_coerce_names(target_categories),
            ignore_categories=_coerce_names(ignore_categories),
            uncategorized_annotation_policy=str(uncategorized_annotation_policy),
            default_non_target_policy=str(default_non_target_policy),
            unclassified_category_policy=str(unclassified_category_policy),
        )

    @classmethod
    def from_config(cls, config) -> "TruthSemantics":
        """Build from Tile/Candidate-compatible configuration attributes."""
        return cls.coerce(
            target_categories=getattr(config, "category_names"),
            ignore_categories=getattr(config, "ignore_categories", None),
            uncategorized_annotation_policy=getattr(
                config, "uncategorized_annotation_policy", "ignore"
            ),
            default_non_target_policy=getattr(
                config, "default_non_target_policy", "background"
            ),
            unclassified_category_policy=getattr(
                config, "unclassified_category_policy", "ignore"
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping | None,
        *,
        fallback_targets: Sequence[str] | str | None = None,
    ) -> "TruthSemantics":
        mapping = dict(mapping or {})
        targets = mapping.get("target_categories", fallback_targets)
        if not targets:
            raise ValueError("truth semantics require at least one target category")
        return cls.coerce(
            target_categories=targets,
            ignore_categories=mapping.get("ignore_categories"),
            uncategorized_annotation_policy=mapping.get(
                "uncategorized_annotation_policy", "ignore"
            ),
            default_non_target_policy=mapping.get(
                "default_non_target_policy", "background"
            ),
            unclassified_category_policy=mapping.get(
                "unclassified_category_policy", "ignore"
            ),
        )

    def to_dict(self) -> dict:
        return {
            "target_categories": list(self.target_categories),
            "ignore_categories": list(self.ignore_categories),
            "uncategorized_annotation_policy": self.uncategorized_annotation_policy,
            "default_non_target_policy": self.default_non_target_policy,
            "unclassified_category_policy": self.unclassified_category_policy,
        }

    def category_name(self, dset, ann: Mapping) -> str | None:
        cid = ann.get("category_id")
        if cid is None:
            return None
        cat = dset.cats.get(cid)
        if cat is None:
            return None
        name = cat.get("name")
        return None if name is None else str(name)

    def annotation_role(self, dset, ann: Mapping) -> str:
        """Return ``target``, ``background``, or ``ignore`` for one annotation."""
        cid = ann.get("category_id")
        if cid is None:
            policy = self.uncategorized_annotation_policy
            if policy == "error":
                raise ValueError(
                    f"annotation {ann.get('id')!r} has no category_id under error policy"
                )
            return policy

        cat = dset.cats.get(cid)
        if cat is None or cat.get("name") is None:
            policy = self.unclassified_category_policy
            if policy == "error":
                raise ValueError(
                    f"annotation {ann.get('id')!r} references unknown category {cid!r}"
                )
            return policy

        name = str(cat["name"])
        if name in self.target_categories:
            return "target"
        if name in self.ignore_categories:
            return "ignore"
        policy = self.default_non_target_policy
        if policy == "error":
            raise ValueError(
                f"annotation {ann.get('id')!r} category {name!r} has no explicit truth role"
            )
        return policy

    def ignore_scope(self, dset, ann: Mapping) -> str | None:
        """Return the conservative exclusion scope for an ignored annotation.

        Explicitly named ignore categories are localized to their annotation
        geometry when possible.  Uncategorized or undeclared annotations are
        different: under an ``ignore`` policy their semantics are not known,
        so the entire source image is ineligible for trusted background until
        that truth has been audited.
        """
        if self.annotation_role(dset, ann) != "ignore":
            return None

        cid = ann.get("category_id")
        if cid is None:
            return "image"
        cat = dset.cats.get(cid)
        if cat is None or cat.get("name") is None:
            return "image"
        if not annotation_has_geometry(ann):
            return "image"
        return "region"

    def partition_annotations(
        self, dset, annotations: Iterable[Mapping]
    ) -> dict[str, list[Mapping]]:
        parts = {"target": [], "background": [], "ignore": []}
        for ann in annotations:
            parts[self.annotation_role(dset, ann)].append(ann)
        return parts

    def partition_ignored_annotations(
        self, dset, annotations: Iterable[Mapping]
    ) -> dict[str, list[Mapping]]:
        """Split ignored annotations into region-local and image-global blocks."""
        parts = {"region": [], "image": []}
        for ann in annotations:
            scope = self.ignore_scope(dset, ann)
            if scope is not None:
                parts[scope].append(ann)
        return parts

    def category_role_map(self, dset) -> dict[str, str]:
        """Describe declared source categories without fabricating uncategorized truth."""
        result = {}
        for cat in dset.cats.values():
            name = str(cat.get("name"))
            fake_ann = {"id": None, "category_id": cat.get("id")}
            result[name] = self.annotation_role(dset, fake_ann)
        return result


def annotation_has_geometry(ann: Mapping) -> bool:
    return ann.get("bbox") is not None or ann.get("segmentation") is not None
