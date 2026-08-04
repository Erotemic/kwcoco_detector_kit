# viame_fish_2026

Systematic host-side runbook for the NOAA FishTrack23 VIAME detector training
runs. Project source lives here; large data, VIAME binaries, generated chips,
logs, checkpoints, and model packages live outside the git checkout.

## Layout

```text
projects/viame_fish_2026/
├── configs/
│   └── train_detector_rf_detr_l_720_90gb.conf
├── docs/
│   └── training_runs.yaml
└── scripts/
    ├── paths.sh
    ├── setup_data.sh
    ├── setup_binaries.sh
    ├── setup_config.sh
    ├── check_setup.sh
    ├── _launch_viame_train.sh
    ├── run_fishtrack23_rfdetr_l_seg720_4gpu_viame0227_gen001.sh
    └── follow_run.sh
```

The default host layout is:

```text
$HOME/ssd-data/FishTrack23-Latest/       # local data mirror
/data/users/$USER/fish/downloads/        # incoming VIAME archives
/data/users/$USER/fish/software/         # immutable versioned installs
/data/users/$USER/fish/viame-current      # symlink to active install
/data/users/$USER/fish/runs/              # isolated run attempts
/data/users/$USER/fish/logs/              # spare host logs
```

Override any location by exporting its `VF_*` variable before running a
script. The defaults are centralized in `scripts/paths.sh`.

## First-time setup on aiq-gpu

From the `kwcoco_detector_kit` checkout:

```bash
cd ~/code/kwcoco_detector_kit
```

### 1. Finish the data mirror

This is resumable and safe to run again:

```bash
bash projects/viame_fish_2026/scripts/setup_data.sh
```

The default source and destination are:

```text
numenor:/data/Public/NOAA/FishTrack23-Latest/
$HOME/ssd-data/FishTrack23-Latest/
```

### 2. Copy the VIAME archive to aiq-gpu

Run this on the machine that currently has the archive:

```bash
scp VIAME-v0.22.7-Linux-64Bit.tar.gz \
    aiq-gpu:/data/users/jon.crall/fish/downloads/
```

### 3. Install the binary

Run this on `aiq-gpu`:

```bash
bash projects/viame_fish_2026/scripts/setup_binaries.sh \
    /data/users/jon.crall/fish/downloads/VIAME-v0.22.7-Linux-64Bit.tar.gz
```

The script extracts into a versioned directory, records the archive SHA256,
finds the real directory containing `setup_viame.sh`, and updates:

```text
/data/users/jon.crall/fish/viame-current
```

It refuses to overwrite an existing versioned install. That makes an upgrade
an explicit new installation instead of an in-place mutation.

### 4. Install the project config into VIAME

The checked-in config is the reproducible source copy. VIAME needs a copy under
its own `configs/pipelines` directory so its relative includes resolve:

```bash
bash projects/viame_fish_2026/scripts/setup_config.sh
```

When Matt supplies a newer config, first save it as a new checked-in filename,
copy the current `run_*.sh` to the next generation, and point the new run script
at the new config. Do not silently replace a config used by a completed run.

### 5. Preflight

```bash
bash projects/viame_fish_2026/scripts/check_setup.sh
```

This verifies the data directory, active binary, installed config,
`viame_train_detector`, visible GPUs, and free work-disk capacity.

## Launch generation 1

Use tmux so the process survives an SSH disconnect:

```bash
tmux new -s fish-v0227-gen001
```

Inside tmux:

```bash
cd ~/code/kwcoco_detector_kit
bash projects/viame_fish_2026/scripts/run_fishtrack23_rfdetr_l_seg720_4gpu_viame0227_gen001.sh
```

Detach with `Ctrl-B`, then `D`. Reattach with:

```bash
tmux attach -t fish-v0227-gen001
```

Follow the newest attempt from another shell:

```bash
cd ~/code/kwcoco_detector_kit
bash projects/viame_fish_2026/scripts/follow_run.sh
```

## Run-attempt organization

Each execution creates a new directory such as:

```text
/data/users/jon.crall/fish/runs/
└── fishtrack23_rfdetr_l_seg720_4gpu_viame0227_gen001/
    ├── latest -> attempt_20260804_190000
    └── attempt_20260804_190000/
        ├── command.sh
        ├── run_manifest.txt
        ├── config.sha256
        ├── viame_archive.sha256
        ├── viame_install_info.txt
        ├── nvidia_smi.txt
        ├── train_detector_rf_detr_l_720_90gb.conf
        ├── run_fishtrack23_..._gen001.sh
        ├── train.log
        ├── exit_code.txt
        ├── augmented_images/
        ├── deep_training/
        └── fish_detector.zip
```

The generated `command.sh` can reproduce the exact invocation for that attempt.
The run directory also isolates `deep_training` and `augmented_images`, which is
important for deciding whether v0.22.7 fixes the earlier hang rather than
accidentally reusing stale state from v0.22.6.

## Starting generation 2

Never edit a historical entry point after using it for a meaningful run. Copy
it and make the next change explicit:

```bash
cd ~/code/kwcoco_detector_kit
cp \
    projects/viame_fish_2026/scripts/run_fishtrack23_rfdetr_l_seg720_4gpu_viame0227_gen001.sh \
    projects/viame_fish_2026/scripts/run_fishtrack23_rfdetr_l_seg720_4gpu_viame0227_gen002.sh
```

Then update all of the following in the new file:

- Header comment explaining exactly what changed.
- `VF_RUN_NAME`.
- Config filename when the config changed.
- Binary-version name when the active VIAME version changed.

Add the planned run to `docs/training_runs.yaml`. After it finishes, record its
status, important metrics, output path, and any failure or hang details there.

## Manual diagnosis when a run appears hung

From another shell:

```bash
nvidia-smi
ps aux | grep -E 'viame_train_detector|python|rf_detr'
du -sh /data/users/jon.crall/fish/runs/*/*/augmented_images 2>/dev/null
du -sh /data/users/jon.crall/fish/runs/*/*/deep_training 2>/dev/null
bash projects/viame_fish_2026/scripts/follow_run.sh
```

Record whether it stopped during video extraction, chip generation, trainer
startup, or an epoch. The last log line and growth of the two cache directories
are more useful than simply recording that it "hung."
