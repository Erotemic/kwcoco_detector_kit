from __future__ import annotations

from pathlib import Path


def test_dataset_config_introspects_kwcoco(synthetic_kwcoco):
    from kwcoco_detector_kit.configs import default_dataset_config

    cfg = default_dataset_config(train_kwcoco=str(synthetic_kwcoco))

    assert cfg["kind"] == "kwcoco_detector_kit.dataset"
    assert cfg["dataset"]["train_kwcoco"] == str(Path(synthetic_kwcoco))
    assert cfg["dataset"]["category_name"] == "widget"
    assert cfg["suggestions"]["introspection"]["train"]["n_images"] == 4
    assert cfg["suggestions"]["introspection"]["train"]["n_annotations"] == 4
    assert cfg["suggestions"]["introspection"]["categories"] == ["widget"]


def test_config_init_inspect_edit_roundtrip(tmp_path, synthetic_kwcoco):
    from kwcoco_detector_kit.configs import (
        ConfigEditConfig,
        ConfigInitConfig,
        ConfigInspectConfig,
        edit_configs,
        init_configs,
        inspect_configs,
        read_yaml,
    )

    env_fpath = tmp_path / "env.yaml"
    data_fpath = tmp_path / "dataset.yaml"

    init_cfg = ConfigInitConfig.cli(
        argv=False,
        data={
            "env": str(env_fpath),
            "dataset": str(data_fpath),
            "train_kwcoco": str(synthetic_kwcoco),
            "execution": "slurm-docker",
            "docker_image": "kwcoco-detector-kit:test",
        },
    )
    assert init_configs(init_cfg) == 0

    env_cfg = read_yaml(env_fpath)
    data_cfg = read_yaml(data_fpath)
    assert env_cfg["environment"]["execution"] == "slurm-docker"
    assert env_cfg["environment"]["docker"]["image"] == "kwcoco-detector-kit:test"
    assert data_cfg["dataset"]["category_name"] == "widget"

    inspect_cfg = ConfigInspectConfig.cli(
        argv=False,
        data={"env": str(env_fpath), "dataset": str(data_fpath)},
    )
    assert inspect_configs(inspect_cfg) == 0

    edit_cfg = ConfigEditConfig.cli(
        argv=False,
        data={
            "env": str(env_fpath),
            "dataset": str(data_fpath),
            "set": [
                "environment.slurm.gres=gpu:4",
                "dataset.tiling.tile_size=1024",
            ],
            "non_interactive": True,
        },
    )
    assert edit_configs(edit_cfg) == 0
    assert read_yaml(env_fpath)["environment"]["slurm"]["gres"] == "gpu:4"
    assert read_yaml(data_fpath)["dataset"]["tiling"]["tile_size"] == 1024
