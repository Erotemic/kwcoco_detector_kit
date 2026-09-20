"""Contract tests shared by all registered trainer plugins."""
from __future__ import annotations

import inspect


def test_registered_generate_config_signatures_cover_protocol():
    """Every built-in trainer must accept every orchestration keyword."""
    from kwcoco_detector_kit.trainers._interface import DetectorTrainer
    from kwcoco_detector_kit.trainers._registry import get_trainer, list_trainers

    protocol_params = inspect.signature(DetectorTrainer.generate_config).parameters
    required = {name for name in protocol_params if name != "self"}

    for trainer_name in list_trainers():
        trainer = get_trainer(trainer_name)
        actual = inspect.signature(trainer.generate_config).parameters
        missing = required - set(actual)
        assert not missing, (
            f"trainer {trainer_name!r} generate_config is missing protocol "
            f"parameters: {sorted(missing)}"
        )


def test_all_registered_trainers_accept_init_checkpoint_keyword():
    """Pareto sweep passes init_checkpoint to every trainer config generator."""
    from kwcoco_detector_kit.trainers._registry import get_trainer, list_trainers

    for trainer_name in list_trainers():
        params = inspect.signature(get_trainer(trainer_name).generate_config).parameters
        assert "init_checkpoint" in params, trainer_name
