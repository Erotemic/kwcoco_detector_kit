# viame_fish_2026

Systematic host-side runbook for NOAA FishTrack23 VIAME detector training.
Project source lives here; large data, VIAME binaries, generated chips, logs,
checkpoints, and model packages live outside the git checkout.

## Host layout

```text
$HOME/ssd-data/FishTrack23-Latest/       # local data mirror
/data/users/$USER/fish/downloads/        # incoming VIAME archives
/data/users/$USER/fish/software/         # immutable versioned installs
/data/users/$USER/fish/viame-current     # symlink to active install
/data/users/$USER/fish/selected_config.env # active config selection
/data/users/$USER/fish/runs/             # isolated run attempts
```

Defaults are centralized in `scripts/paths.sh` and can be overridden with
`VF_*` environment variables.

## Setup on aiq-gpu

From the repository root:

```bash
cd ~/code/kwcoco_detector_kit
```

### 1. Mirror the data

```bash
bash projects/viame_fish_2026/scripts/setup_data.sh
```

### 2. Download VIAME v0.22.7-rc2

```bash
mkdir -p /data/users/jon.crall/fish/downloads

gdown \
    10tJsWRUJn_FMPwWKkW9S6-H6fOF3DKyB \
    -O /data/users/jon.crall/fish/downloads/VIAME-v0.22.7-rc2-Linux-64Bit.tar.gz
```

### 3. Install the binary

```bash
bash projects/viame_fish_2026/scripts/setup_binaries.sh \
    /data/users/jon.crall/fish/downloads/VIAME-v0.22.7-rc2-Linux-64Bit.tar.gz \
    0.22.7-rc2
```

The install is immutable and versioned. `viame-current` points to the active
installation.

### 4. Select the bundled segmentation config

```bash
bash projects/viame_fish_2026/scripts/setup_config.sh \
    train_detector_rf_detr_l_seg_720.conf
```

This does not overwrite the bundled config. It records the selected basename in:

```text
/data/users/$USER/fish/selected_config.env
```

Because the selection stores a basename rather than an absolute installation
path, it follows `viame-current` after an upgrade. The preflight will fail if the
new installation does not contain that config.

List available configs:

```bash
bash projects/viame_fish_2026/scripts/setup_config.sh --list
```

Show the current selection:

```bash
bash projects/viame_fish_2026/scripts/setup_config.sh --show
```

To use a local or project-owned override, pass its path. It will be copied into
the active VIAME `configs/pipelines` directory and selected:

```bash
bash projects/viame_fish_2026/scripts/setup_config.sh \
    /path/to/custom_training.conf
```

Or select a config from this project's `configs` directory explicitly:

```bash
bash projects/viame_fish_2026/scripts/setup_config.sh \
    --project train_detector_rf_detr_l_720_90gb.conf
```

### 5. Preflight

```bash
bash projects/viame_fish_2026/scripts/check_setup.sh
```

## Launch generation 1

```bash
tmux new -s fish-v0227-rc2-gen001
```

Inside tmux:

```bash
cd ~/code/kwcoco_detector_kit

bash \
    projects/viame_fish_2026/scripts/run_fishtrack23_rfdetr_l_seg720_4gpu_viame0227_gen001.sh
```

Detach with `Ctrl-B`, then `D`. Reattach with:

```bash
tmux attach -t fish-v0227-rc2-gen001
```

Follow the newest attempt from another shell:

```bash
cd ~/code/kwcoco_detector_kit
bash projects/viame_fish_2026/scripts/follow_run.sh
```

## Run provenance

Each launch creates a new timestamped attempt. The attempt contains:

- The exact selected config snapshot and SHA256.
- The config-selection state.
- The versioned run entry point.
- The generated command.
- VIAME archive/install metadata.
- GPU inventory, log, exit code, caches, and model output.

Training uses the config under the active VIAME installation so its relative
`include` paths resolve correctly. The copied snapshot records exactly what was
used.

## Changing configs systematically

`setup_config.sh` chooses the active config, but a versioned `run_*.sh` may also
state an expected config. Generation 1 requires:

```text
train_detector_rf_detr_l_seg_720.conf
```

If another config is the next experiment, copy the run entry point to `gen002`,
change `VF_RUN_NAME` and `VF_EXPECTED_CONFIG_NAME`, and register the new run in
`docs/training_runs.yaml`. This prevents a historical run name from silently
changing meaning.

## Manual diagnosis when a run appears hung

```bash
nvidia-smi
ps aux | grep -E 'viame_train_detector|python|rf_detr'
du -sh /data/users/jon.crall/fish/runs/*/*/augmented_images 2>/dev/null
du -sh /data/users/jon.crall/fish/runs/*/*/deep_training 2>/dev/null
bash projects/viame_fish_2026/scripts/follow_run.sh
```
