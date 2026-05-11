# Install

## Standard install

```bash
pip install kwcoco-detector-kit
```

## Editable / dev install

```bash
git clone https://github.com/Erotemic/kwcoco-detector-kit.git
cd kwcoco-detector-kit
git submodule update --init --recursive   # clones tpl/DEIMv2, tpl/Open-GroundingDino
pip install -e ".[dev]"
```

## Third-party trainer codebases (submodules)

The kit drives the real DEIMv2 and OpenGroundingDINO trainers via subprocess. Their source lives as **git submodules** under [`tpl/`](../tpl/):

```text
tpl/
├── DEIMv2/               https://github.com/Erotemic/DEIMv2.git @ 377e10a2 (Phase 1+2)
└── Open-GroundingDino/   https://github.com/Erotemic/Open-GroundingDino.git @ b59dd5e7 (Phase 2)
```

A fresh clone of this repo gets empty `tpl/` directories. Initialize them with:

```bash
git submodule update --init --recursive
```

After that, the kit's trainer plugins find the submodules automatically — no env vars needed. Override the lookup with `$KCD_DEIMV2_REPO_DPATH` / `$KCD_OPENGROUNDINGDINO_REPO_DPATH` if you keep checkouts elsewhere.

To bump a submodule to a newer commit:

```bash
cd tpl/DEIMv2
git fetch && git checkout <sha>
cd ../..
git add tpl/DEIMv2
git commit -m "tpl: bump DEIMv2 to <short-sha>"
```

The kit's `pyproject.toml` doesn't `pip install` these — they're consumed as subprocess targets, not Python packages. (DEIMv2's hidden transitive runtime deps are declared in `[project.optional-dependencies.deimv2]`; install via `pip install -e ".[deimv2]"`.)

## Optional trainer-plugin deps

The base install gets you `mock_tiny`. To run the DEIMv2 or OpenGroundingDINO trainers you also need their plugin-specific deps (failure #11):

```bash
pip install -e ".[deimv2,opengroundingdino]"
```

Then run the env probe to confirm every transitive runtime dep is reachable:

```bash
python -m kwcoco_detector_kit check-env
```

`check-env` probes for every line of `tpl/DEIMv2/requirements.txt` (failure #11), the ONNX-trio (failures #9 + #10), and the OpenGroundingDINO + SAM2 deps. Missing modules are reported; pass `--install` to attempt installation.

## torch / torchvision pin (failure #8)

`torch X.Y` must always be installed alongside `torchvision Z.W` from the same matched PyTorch index. Use:

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
```

Independent installs of torch and torchvision can leave you with `RuntimeError: operator torchvision::nms does not exist`.

## kwcoco subset CLI (failure #19)

`kwcoco subset --select_images "..."` requires the `jq` Python package, which isn't a declared kwcoco dep. The canonical form for the kit is:

```bash
python -m kwcoco subset --gids 1,2,3,4 --src $TRAIN_FPATH --dst $SUBSET_FPATH
```

If you need the richer `--select_images` syntax, install `jq` first:

```bash
pip install jq
```
