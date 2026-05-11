# Install

## Standard install

```bash
pip install kwcoco-detector-kit
```

## Editable / dev install

```bash
git clone https://github.com/Erotemic/kwcoco-detector-kit.git
cd kwcoco-detector-kit
pip install -e ".[dev]"
```

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
