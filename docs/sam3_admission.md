# SAM 3.1 admission test

KDK's first SAM 3 integration is intentionally an **admission test**, not yet a
full trainer plugin.  The goal is to validate the pretrained model before
investing in training integration:

1. SAM 3.1 checkpoint access works.
2. Eager single-image inference fits the deployment GPU (RTX 3090 / 24 GB).
3. Native instance masks convert cleanly to KWCoco.
4. Zero-shot text prompting is at least plausible on representative validation
   chips.
5. Peak VRAM and latency are recorded before choosing a fine-tuning design.

## Source fork

KDK pins the Erotemic SAM3 fork as the `tpl/sam3` submodule. Initialize
that gitlink (or point `KCD_SAM3_REPO_DPATH` at another checkout) and install
it editable for the admission environment:

```bash
cd ~/code/kwcoco_detector_kit

git submodule update --init --recursive tpl/sam3
uv pip install -e ./tpl/sam3
```

Request access to `facebook/sam3.1` on Hugging Face and authenticate once:

```bash
hf auth login
```

The wrapper explicitly downloads the SAM 3.1 checkpoint when
`--checkpoint_version=sam3.1`.  A local checkpoint can instead be supplied via
`--checkpoint=/path/to/checkpoint.pt`.

## 3090 admission run

Use representative ML-ready/native-resolution chips rather than resizing a
small crop upward merely to satisfy the network.  For example:

```bash
kwcoco-detector-kit sam3-admission \
    --src=/path/to/validation_tiles.kwcoco.zip \
    --out_dpath=$HOME/data/sam3_admission \
    --target_category=poop \
    --prompt=poop \
    --device=cuda:0 \
    --checkpoint_version=sam3.1 \
    --limit=32 \
    --balanced=true \
    --amp_dtype=bfloat16 \
    --score_thresh=0.05 \
    --evaluate=true
```

The command always displays progress and writes:

```text
sam3_admission/
    admission_report.json
    truth_subset.kwcoco.zip
    predictions.kwcoco.zip
    eval/
        detect_metrics.json
```

`admission_report.json` records GPU identity, model-load time, post-warmup
latency, peak allocated/reserved CUDA memory, remaining VRAM headroom, and the
KDK/SAM3 provenance.

## Why the wrapper passes PIL images

At the pinned SAM3 API, `Sam3Processor.set_image()` handles PIL dimensions
correctly.  Its NumPy branch records `image.shape[-2:]`, which is not the `(H,W)`
shape for standard HWC NumPy arrays.  KDK therefore converts its HWC RGB array
to PIL at the backend boundary.  No SAM3 fork patch is required for this
admission test.

## Not yet part of admission

The first pass deliberately does not add a full `sam3` trainer registration,
model packaging, ONNX export, or distributed fine-tuning policy.  Those should
be designed after the pretrained model passes the 3090 and zero-shot tests.

## Native-resolution source-image admission

If the input KWCoco contains original source images rather than already-cut
chips, use KDK's tiled predictor so SAM3 sees native 1008x1008 crops with
overlap instead of shrinking the whole source image:

```bash
kwcoco-detector-kit sam3-admission \
    --src=/path/to/vali.kwcoco.zip \
    --out_dpath=$HOME/data/sam3_admission_native \
    --target_category=poop \
    --prompt=poop \
    --device=cuda:0 \
    --checkpoint_version=sam3.1 \
    --limit=32 \
    --balanced=true \
    --windowed=true \
    --overlap=0.25 \
    --whole_image_pass=false \
    --amp_dtype=bfloat16 \
    --score_thresh=0.05
```

The wrapper uses a 1008x1008 native source crop because that is SAM3's image
processor resolution. KDK translates the surviving boxes and native masks back
to source-image coordinates before writing predictions.
