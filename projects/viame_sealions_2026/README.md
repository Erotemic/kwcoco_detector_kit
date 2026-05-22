# viame_sealions_2026

NOAA Steller sea-lion detector training project, hosted inside
`kwcoco_detector_kit`. This subtree owns scripts, docs, tests, and run
registry for the sea-lion experiments; the underlying imagery and
kwcoco bundles live in a separate data directory pointed to by
`KCD_DATA_DPATH` (see `scripts/paths.sh`).

## Layout

```
projects/viame_sealions_2026/
├── README.md              ← this file
├── AGENT.md               ← project-scoped context for agents
├── docs/
│   ├── research_plan.md   ← phased detector roadmap
│   ├── class_schemes.yaml ← declarative source→target class mappings
│   └── training_runs.yaml ← run registry, synced between hosts
├── scripts/
│   ├── paths.sh           ← single source of truth for all KCD_* paths
│   ├── build_scheme_kwcoco.py
│   ├── fetch_pretrained.sh
│   ├── launch_pup_vs_nonpup_arisia.sh
│   ├── sbatch_pup_vs_nonpup.sh
│   ├── submit_pup_vs_nonpup.sh
│   ├── training_registry.py
│   └── … (data-prep, conversion, spotcheck helpers)
└── tests/
    ├── conftest.py
    ├── unit/              ← fast synthetic tests
    └── expensive/         ← real-data tests, skip if bundles absent
```

## Paths

Every script sources `scripts/paths.sh`. Override variables in your
shell rc instead of editing scripts. Key variables:

| variable | default | notes |
|---|---|---|
| `KCD_REPO_ROOT` | this directory | project tree inside the kit |
| `KCD_DATA_DPATH` | `/data/users/jon.crall/dvc-repos/viame_sealions_2026` | sea-lion data dir |
| `KCD_TRAINING_READY_DIR` | `$KCD_DATA_DPATH/training_ready_v1` | per-scheme kwcoco bundles |
| `KCD_TRAINING_ROOT` | `$KCD_DATA_ROOT/kcd_sealion` | per-experiment workspaces |
| `KCD_PRETRAINED_ROOT` | `$KCD_DATA_ROOT/pretrained_models` | downloaded checkpoints |
| `KCD_KIT_DPATH` | `$HOME/code/kwcoco_detector_kit` | kit checkout for follow_job.py etc. |

### Host-specific overrides

**arisia** (compute host, active training): defaults match
arisia's `/data/users/jon.crall/...` layout — no overrides needed.

**namek** (workstation): the data lives on the raid mount, so:

```bash
# in ~/.bashrc or ~/.zshrc
export KCD_DATA_DPATH=/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026
# KCD_TRAINING_READY_DIR etc. derive from KCD_DATA_DPATH automatically
```

## Workflow

### 1. Build a per-scheme kwcoco bundle

```bash
python3 scripts/build_scheme_kwcoco.py --scheme pup_vs_nonpup
# -> $KCD_SCHEMES_DIR/pup_vs_nonpup/{train,vali,test}.kwcoco.zip
```

Available schemes are declared in [docs/class_schemes.yaml](docs/class_schemes.yaml):
`single_sealion` (P0), `pup_vs_nonpup` (P1, active), `lifestage_6cls` (P2).

### 2. Fetch the pretrained init checkpoint

```bash
bash scripts/fetch_pretrained.sh
# -> $KCD_DEIMV2_DINOV3_S_COCO_PTH
```

### 3. Submit a slurm training run (arisia)

```bash
bash scripts/submit_pup_vs_nonpup.sh
```

This `sbatch`-submits [scripts/sbatch_pup_vs_nonpup.sh](scripts/sbatch_pup_vs_nonpup.sh)
(`--gres=gpu:4`, 72h walltime) and tails the job log via the kit's
`smoketests/dino_v2_4x/slurm/follow_job.py`. The sbatch wrapper runs
[scripts/launch_pup_vs_nonpup_arisia.sh](scripts/launch_pup_vs_nonpup_arisia.sh)
inside the kit's docker image (`kwcoco-detector-kit:ogdino-cu132-arisia`)
for full reproducibility.

Override knobs:

```bash
NUM_EPOCHS=10 KCD_PER_GPU_BATCH=8 bash scripts/submit_pup_vs_nonpup.sh
FOLLOW=0 bash scripts/submit_pup_vs_nonpup.sh   # submit and detach
```

### 4. Record the result

```bash
python3 scripts/training_registry.py update <run-id> \
    --status done \
    --metric vali_map=... --metric vali_map50=... \
    --artifact detect_metrics_json=$KCD_ROOT_PUP_VS_NONPUP/eval/<candidate>/eval/detect_metrics.json
```

The registry file [docs/training_runs.yaml](docs/training_runs.yaml) is
synced between hosts.

## Tests

```bash
python3 -m pytest projects/viame_sealions_2026/tests/unit -q       # fast
python3 -m pytest projects/viame_sealions_2026/tests/expensive -q  # needs real bundles
```

Expensive tests `pytest.skip` when the kwcoco bundles aren't on disk,
so they're safe to run on any host.

## Older examples

[examples/viame_sealions_2026/](../../examples/viame_sealions_2026/) holds
the previous-generation OpenGroundingDINO recipes. They are kept as a
historical reference and are not the active path.
