# Handoff — running an agent directly on aiq-gpu

Written 2026-08-14 for the case where a second Claude session runs on a VM
co-located with aiq-gpu, with the data directories bind-mounted instead of
rsync'd. Read this together with the project
[README](../README.md), the kit-root [CLAUDE.md](../../../CLAUDE.md), and the
journal entry
[2026-08-14_orientation.md](journals/2026-08-14_orientation.md).

## Which directories to share

**Mount rule: identical absolute paths on both sides.** Every script in this
repo reads paths from a `paths.sh` and never derives them from `$PWD`; VIAME
snapshots absolute config paths into each attempt directory; and kwcoco bundles
bake absolute `file_name` values. A path that means something different on the
two machines silently produces a run you cannot reproduce or re-score. Mount
aiq's `/data/users/jon.crall` at `/data/users/jon.crall`, not at `/mnt/aiq`.

| directory | variable | why | access |
|---|---|---|---|
| `/data/users/jon.crall/fish` | `VF_WORK_DPATH` | **The one that matters.** Contains `runs/<run_name>/attempt_*/` (train.log, run_manifest.txt, config snapshot, exit_code.txt, deep_training/, augmented_images/, the output `.zip`), plus `software/`, `viame-current`, `selected_config.env`, and `inventory/`. Progress inspection is entirely reads under here. | rw for the aiq agent; ro is enough for a remote observer |
| `$HOME/ssd-data/FishTrack23-Latest` | `VF_DATA_DPATH` | The corpus. Needed to inventory it, to build sequence-disjoint `train/ vali/ test/` splits, and to spot-check annotations. | ro for inspection; rw only when writing split directories |
| `/data/users/jon.crall/slurm_logs` | `KCD_SLURM_LOG_DPATH` | Only if fish work goes through slurm. Currently the fish runs launch directly in tmux, so this is empty for fish. | ro |
| `/data/users/jon.crall/pretrained_models` | `KCD_PRETRAINED_ROOT` | Only if we train the DEIMv2 stack on fish and need a seed checkpoint. | ro |
| `/data/Public/VIAME/` | `KCD_DATA_DPATH` | Shared read-only data store. Relevant only if the fish corpus is promoted there alongside the sea-lion bundles. | ro |

### Size warning

`VF_WORK_DPATH` is where the chips land. `deep_training/` and
`augmented_images/` under a single attempt can reach **hundreds of GB** — the
config comment estimates millions of chips for a large video set, PNG by
default. Do not put this mount on a slow link and then expect to `du -sh` it
casually, and do not let a remote agent regenerate the cache over a network
filesystem. Inspecting progress needs `train.log` and a few small text files,
which is cheap; walking the chip tree is not.

### What NOT to share

The kit checkout at `~/code/kwcoco_detector_kit`. Two agents editing one
working tree will collide, and the git state becomes ambiguous. Keep a separate
clone on each side and move code through git. The aiq agent should `git pull`
before acting and commit its own work.

## What the aiq agent should do first

```bash
cd ~/code/kwcoco_detector_kit
git pull
bash projects/viame_fish_2026/scripts/check_setup.sh
bash projects/viame_fish_2026/scripts/collect_data_manifest.sh
```

`collect_data_manifest.sh` writes `$VF_WORK_DPATH/inventory/` with
`inventory.json`, `inventory.md`, a depth-3 tree listing, disk usage, GPU
inventory, and core count. It is stdlib-python only, so it runs under the
system interpreter or VIAME's without installing anything into either. Read
`inventory.md` first — it answers the questions the plan is currently blocked
on (image dimensions, object-size distribution, category frequency, video vs
still composition).

## State of play as of this handoff

**The objective is a complementary DEIMv2 model, box-only, trained through the
kit pipeline.** An RF-DETR model has already been trained through VIAME's
native tooling; this is a second, architecturally different detector, not a
retune of that one. The VIAME-native runbook in `scripts/` (`run_*.sh`,
`_launch_viame_train.sh`, `setup_config.sh`) is therefore **not** the path for
this run — it stays as-is for reference and for retraining the RF-DETR side.

- aiq-gpu is 4× RTX PRO 6000 Blackwell, 96 GB each.
- Kit runs on aiq **always go through slurm**: `KCD_NO_SLURM=0`,
  `KCD_DOCKER_GPU_MODE=gpus` (aiq needs docker's `--gpus all` mode, not
  arisia's `runtime`). `--gres`/`--partition` come from the user's shell rc.
  Follow the sea-lion `submit_train_*_aiq_*.sh` scripts as the template.
- The docker image is the reproducibility unit — prefer a rebuild over a dev
  mount for any non-trivial run.
- No fish training run has completed under *this project's* runbook.
  `docs/training_runs.yaml` lists the VIAME-native `gen001` as deferred; an
  earlier v0.22.6 attempt hung and was abandoned.
- Read the journal entry before doing anything: it reviews the RF-DETR config
  against VIAME's `rf_detr_trainer.py` and turns it into a map of where that
  model is likely weak (small objects, rare classes), which is what makes the
  DEIM model complementary rather than redundant. It also documents a
  contamination problem in any head-to-head comparison that needs deciding
  before results are presented.

## Work queue for the aiq agent

In order — each step is blocked on the one above it:

1. Run the inventory (below). Read `inventory.md`.
2. Write the VIAME-CSV → kwcoco converter. There is **no** existing reader for
   the VIAME alternating class/score format in the kit;
   `projects/viame_sealions_2026/scripts/convert_sealions_csv_to_kwcoco.py`
   handles a different, headered format and says so in its docstring. The
   row-level logic in `scripts/inventory_data.py:parse_viame_csv` is tested and
   is the right starting point.
3. Freeze sequence-disjoint train/vali/test splits — whole sequences on one
   side only, never a frame-level split, because adjacent frames of one track
   are near-duplicates.
4. Build the tile cache at the size the box-size percentiles imply.
5. Submit via slurm. Register the run in `docs/training_runs.yaml` first.

Also open, and worth resolving early because it affects how results can be
reported: locate the existing RF-DETR run's artifacts and determine which
sequences it trained on.

## Conventions this project inherits

- Journals: every meaningful cycle gets a dated entry in
  [docs/journals/](journals/).
- Runs: never edit a historical `run_*.sh`; copy it to the next generation and
  register the new id in [docs/training_runs.yaml](training_runs.yaml).
- Long jobs run in `tmux` with `tee`, never `nohup`, never backgrounded.
- Outputs go to the data drive, never into the kit checkout.
