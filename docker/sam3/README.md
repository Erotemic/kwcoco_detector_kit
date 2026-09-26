# SAM3 Docker admission runtime

KDK evaluates the pinned `tpl/sam3` fork through a dedicated Docker image so
SAM3's PyTorch/CUDA/Hugging Face dependency stack does not leak into the host
Python environment. The first integration is deliberately inference-only: it
measures pretrained SAM 3.1 zero-shot behavior and RTX 3090 deployment cost
before KDK grows a trainer adapter.

## Build

Initialize the pinned fork, then build the CUDA 12.6 / PyTorch 2.7.1 image:

```bash
cd ~/code/kwcoco_detector_kit
git submodule update --init --recursive tpl/sam3
docker/sam3/kcd-sam3 build
```

The conservative `cu126` profile is intentional for the Ampere RTX 3090. SAM3
requires a modern PyTorch/CUDA stack but does not need a Blackwell-only image
for this admission experiment.

Check the runtime:

```bash
docker/sam3/kcd-sam3 image-info
```

SAM3 and RF-DETR install the same exact released KW runtime stack from
`docker/requirements-kwstack.txt`. `image-info` reports `kwimage_ext` and the
selected CPU-NMS backend; production images require the Rust backend.

The checkpoint is **not** baked into the image. By default the helper mounts
`$HOME` unchanged, so Hugging Face's normal host cache is reused. Authenticate
on the host with `hf auth login`, or export `HF_TOKEN`; the helper forwards the
token when present. A custom `HF_HOME` is forwarded and mounted as well.

## Frozen zero-shot test benchmark

A test-set zero-shot run is useful only if its choices are frozen before
looking at results. Do not use test false positives to tune prompts, thresholds,
stitching, or training data.

```bash
export TEST=/path/to/test.kwcoco.zip
export OUT=$HOME/data/sam3_zero_shot_test
mkdir -p "$OUT"
set -o pipefail

docker/sam3/kcd-sam3 admission \
    --src="$TEST" \
    --out_dpath="$OUT" \
    --target_category=poop \
    --prompt=poop \
    --checkpoint_version=sam3.1 \
    --device=cuda:0 \
    --limit=0 \
    --balanced=false \
    --windowed=true \
    --resolution=1008 \
    --overlap=0.25 \
    --whole_image_pass=false \
    --amp_dtype=bfloat16 \
    --score_thresh=0.05 \
    --evaluate=true \
    --review=false \
    2>&1 | tee "$OUT/run.log"
```

## Validation false-positive / hard-negative review

Validation is the development surface. Run the same frozen zero-shot model but
turn on truth-aware review. Importantly, KDK builds that review against the
**original validation KWCoco**, not the binary metric subset, so existing
nuisance annotations remain visible.

```bash
export VALI=/path/to/vali.kwcoco.zip
export OUT=$HOME/data/sam3_zero_shot_vali
mkdir -p "$OUT"
set -o pipefail

docker/sam3/kcd-sam3 admission \
    --src="$VALI" \
    --out_dpath="$OUT" \
    --target_category=poop \
    --prompt=poop \
    --checkpoint_version=sam3.1 \
    --device=cuda:0 \
    --limit=64 \
    --balanced=true \
    --windowed=true \
    --resolution=1008 \
    --overlap=0.25 \
    --whole_image_pass=false \
    --amp_dtype=bfloat16 \
    --score_thresh=0.05 \
    --evaluate=true \
    --review=true \
    --ignore_categories=ignore,unknown,unkown \
    --review_min_score=0.05 \
    --review_top_n=500 \
    2>&1 | tee "$OUT/run.log"
```

The review directory contains `review_queue.json`, `review_queue.tsv`,
`review.kwcoco.zip`, and `index.html`. Predictions are classified using KDK's
truth semantics as matched target, known distractor, uncertain region,
overlapping target, or unexplained prediction. The latter two review classes
are useful for annotation QA; confident known-distractor/unexplained false
positives are candidates for later hard-negative mining after human review.

## External dataset roots

Absolute `--key=/path` arguments are mounted automatically. If a KWCoco file
references assets under another root, mount it at the identical path:

```bash
export KCD_SAM3_EXTRA_MOUNTS=/data/my_dataset
```

Multiple roots can be newline-separated.

## Debugging / dry-run

```bash
KCD_DOCKER_DRYRUN=1 docker/sam3/kcd-sam3 admission \
    --src=/data/example.kwcoco.zip --out_dpath=/tmp/sam3-test

docker/sam3/kcd-sam3 shell
docker/sam3/kcd-sam3 exec python -c 'import torch; print(torch.cuda.get_device_name(0))'
```
