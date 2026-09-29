# Install

## Standard install

```bash
pip install kwcoco-detector-kit
```

## Editable / dev install

```bash
git clone https://github.com/Erotemic/kwcoco-detector-kit.git
cd kwcoco-detector-kit
git submodule update --init --recursive   # includes tpl/libreyolo and legacy/reference trainers
pip install -e ".[dev]"
```

## Model-engine and reference codebases (submodules)

KDK pins LibreYOLO as its primary in-process model engine. The older direct DEIMv2 and OpenGroundingDINO integrations remain source-visible reference/specialized backends and are launched via subprocess. These checkouts live as **git submodules** under [`tpl/`](../tpl/):

```text
tpl/
├── libreyolo/            primary general model engine (pinned submodule)
├── DEIMv2/               legacy/direct reference integration
└── Open-GroundingDino/   legacy/direct specialized integration
```

A fresh clone of this repo gets empty `tpl/` directories. Initialize them with:

```bash
git submodule update --init --recursive
```

After that, the kit's trainer plugins find the submodules automatically — no env vars needed. Override the lookup with `$KCD_DEIMV2_REPO_DPATH` / `$KCD_OPENGROUNDINGDINO_REPO_DPATH` / `$KCD_LIBREYOLO_REPO_DPATH` if you keep checkouts elsewhere.

To bump a submodule to a newer commit:

```bash
cd tpl/DEIMv2
git fetch && git checkout <sha>
cd ../..
git add tpl/DEIMv2
git commit -m "tpl: bump DEIMv2 to <short-sha>"
```

KDK does not install the submodule packages themselves. `trainer=libreyolo` imports the pinned `tpl/libreyolo` source tree in-process, while the legacy direct trainers launch their pinned source trees as subprocess targets. Their runtime dependencies are exposed as KDK extras; for example `pip install -e ".[libreyolo]"` or `pip install -e ".[deimv2]"`.

## Optional trainer-plugin deps

The base install gets you `mock_tiny`. Trainer engines have plugin-specific deps. LibreYOLO is the primary general backend; the direct DEIMv2/OpenGroundingDINO integrations remain optional reference/specialized paths:

```bash
pip install -e ".[libreyolo,deimv2]"
```

OpenGroundingDINO currently pins Transformers 4.x while LibreYOLO's modern transformer families use Transformers 5.x. Do **not** install `libreyolo` and `opengroundingdino` extras into the same environment; use a separate legacy/specialized environment if that direct trainer is still needed.

Then run the env probe to confirm every transitive runtime dep is reachable:

```bash
python -m kwcoco_detector_kit check-env
```

`check-env` can probe KDK core/ONNX dependencies and backend-specific groups, including the runtime dependencies mirrored from the pinned LibreYOLO checkout. Missing modules are reported; pass `--install` to attempt installation.

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


For the general LibreYOLO backend, see [`docs/libreyolo_integration.md`](libreyolo_integration.md).
