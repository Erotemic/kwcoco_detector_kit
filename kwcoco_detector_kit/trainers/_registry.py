"""
Central trainer-plugin registry.

Plugins register at module-import time::

    from kwcoco_detector_kit.trainers._registry import register_trainer

    @register_trainer
    class DEIMv2Trainer:
        name = "deimv2"
        ...

Callers ask for a plugin by name::

    from kwcoco_detector_kit.trainers._registry import get_trainer
    trainer = get_trainer("deimv2")
"""
from __future__ import annotations

from typing import Dict


_REGISTRY: Dict[str, object] = {}


def register_trainer(cls_or_instance):
    """Class or instance decorator. Reads ``.name`` to register.

    Accepts either a class (instantiated lazily) or an already-instantiated
    plugin instance. Re-registering the same name is an error — explicit
    deregister via ``_REGISTRY.pop()`` if you need to swap.
    """
    name = getattr(cls_or_instance, "name", None)
    if not name:
        raise ValueError(
            f"register_trainer: {cls_or_instance!r} has no .name attribute"
        )
    if name in _REGISTRY:
        raise ValueError(
            f"trainer {name!r} already registered ({_REGISTRY[name]!r}); "
            f"deregister explicitly before overriding"
        )
    # If it's a class, instantiate. Plugins that need init args should
    # register a pre-built instance instead.
    instance = cls_or_instance() if isinstance(cls_or_instance, type) else cls_or_instance
    _REGISTRY[name] = instance
    return cls_or_instance


def get_trainer(name: str):
    if name not in _REGISTRY:
        raise KeyError(
            f"trainer {name!r} is not registered; "
            f"available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_trainers() -> list[str]:
    return sorted(_REGISTRY)


def deregister_trainer(name: str):
    _REGISTRY.pop(name, None)
