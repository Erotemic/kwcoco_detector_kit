# New Training Machine Setup

Step-by-step guide for bringing a fresh Linux host into the sea lion
detector training pipeline. Copy-paste each block in order; sections
that produce diagnostic output are marked so you can verify before
continuing.

---

## 1. Hardware introspection

Run these first and share the output — it determines which Docker
CUDA profile to build and how to right-size slurm job requests.

```bash
# GPU inventory
nvidia-smi

# Detailed per-GPU compute capability and VRAM
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap \
    --format=csv,noheader

# Host CUDA toolkit version (may differ from driver's "CUDA Version")
nvcc --version 2>/dev/null || echo "nvcc not on PATH (driver-only install)"

# CPU / memory
lscpu | grep -E '^(Architecture|CPU\(s\)|Thread|Core|Model name)'
free -h

# Storage — identify which filesystems have headroom for data + tile cache
df -h | grep -v tmpfs | grep -v udev

# Check for fast local storage (NVMe/SSD vs HDD)
lsblk -d -o NAME,SIZE,ROTA,TYPE | grep -v loop
# ROTA=0 = SSD/NVMe, ROTA=1 = HDD
```

Key things to record:
- GPU model, VRAM per card, and count
- Driver version + CUDA capability from `nvidia-smi`
- Which filesystem has ≥ 200 GB free (tile cache lives here — SSD preferred)
- Whether `/data/Public` and `/data/users` already exist or need creating

---

## 2. Directory structure

The pipeline assumes two canonical roots at the same paths on every
host. Create them with the correct ownership:

```bash
# Shared read-only corpus tree (kwcoco bundles, imagery)
sudo mkdir -p /data/Public/VIAME/viame_sealions_2026/unpacked

# Per-user work area (training runs, tile cache, slurm logs, pretrained ckpts)
mkdir -p /data/users/$USER/kcd_sealion/runs
mkdir -p /data/users/$USER/kcd_sealion/tile_cache
mkdir -p /data/users/$USER/pretrained_models
mkdir -p /data/users/$USER/slurm_logs

# If your fast storage isn't at /data/users, symlink it there.
# Example: NVMe at /ssd-data -> /data/users/jon.crall/kcd_sealion
# mkdir -p /ssd-data/kcd_sealion/tile_cache
# ln -sfn /ssd-data/kcd_sealion/tile_cache /data/users/$USER/kcd_sealion/tile_cache
```

---

## 3. Clone the kit

```bash
cd ~/code   # or wherever you keep checkouts

# Full clone with both submodules (DEIMv2 + kwcoco_dataloader)
git clone --recurse-submodules \
    https://github.com/Erotemic/kwcoco_detector_kit.git

cd kwcoco_detector_kit

# Verify submodules are populated
ls tpl/DEIMv2/train.py tpl/kwcoco_dataloader/kwcoco_dataloader/__init__.py
```

If the repo already exists but submodules are empty:

```bash
git submodule update --init --recursive
```

---

## 4. Build the Docker image

The kit uses `build_auto.sh` which reads the host NVIDIA driver version
and picks the right CUDA base image automatically.

```bash
# From kit root
bash docker/opengroundingdino/build_auto.sh
```

This tags the result as both `kwcoco-detector-kit:ogdino-auto` and a
profile-specific tag like `kwcoco-detector-kit:ogdino-cu132-arisia`.
The build runs the full pytest suite inside the container as a
regression gate — if it fails, the build is rejected.

**If the auto-detect produces the wrong CUDA version**, override it:

```bash
# Example: force CUDA 12.4 profile
KCD_DOCKER_CUDA_PROFILE=cu124 bash docker/opengroundingdino/build_auto.sh
```

Inspect what profile was chosen:

```bash
docker images | grep kwcoco-detector-kit
```

Record the tag — you will plug it into `KCD_IMAGE` in the next step.

---

## 5. Shell environment

Add these to `~/.bashrc` (or `~/.zshrc`) so every terminal session
picks up the right image tag and user paths without editing scripts:

```bash
# kwcoco_detector_kit — new host overrides
# (adjust the image tag to match what build_auto.sh produced)
export KCD_IMAGE=kwcoco-detector-kit:ogdino-cu132-arisia   # ← change me
export KCD_KIT_DPATH=$HOME/code/kwcoco_detector_kit
export KCD_DATA_ROOT=/data/users/$USER
# If /data/Public/VIAME isn't mounted the same as on arisia/namek,
# override KCD_DATA_DPATH too:
# export KCD_DATA_DPATH=/data/Public/VIAME/viame_sealions_2026
```

Then reload:

```bash
source ~/.bashrc
```

---

## 6. Verify paths

```bash
cd ~/code/kwcoco_detector_kit
bash projects/viame_sealions_2026/scripts/check_paths.sh
```

Expected output: all `[OK]` lines. Any `[MISSING]` tells you exactly
which path to create or rsync (see §7).

---

## 7. Rsync corpus data from arisia or namek

The v2 norm bundles and unpacked imagery live on the host that built
them. Pull them to this machine's `/data/Public/VIAME/viame_sealions_2026/`.

**From arisia** (if this new machine can reach arisia):

```bash
# Dry-run first to see what will transfer
rsync -avhn --info=progress2 \
    arisia:/data/Public/VIAME/viame_sealions_2026/unpacked/ \
    /data/Public/VIAME/viame_sealions_2026/unpacked/

# Actual transfer (omit -n)
rsync -avh --info=progress2 \
    arisia:/data/Public/VIAME/viame_sealions_2026/unpacked/ \
    /data/Public/VIAME/viame_sealions_2026/unpacked/
```

Minimum files needed to run training:

```
unpacked/
  train_norm_v2.kwcoco.zip   (≈3.7 GB)
  vali_norm_v2.kwcoco.zip
  test_norm_v2.kwcoco.zip
  all_norm.kwcoco.zip
  burlynb/Public/Redacted_Imagery/   (raw JPEG imagery — largest transfer)
```

If the source imagery is accessible via the same path (`/data/Public/VIAME/...`
is a NAS/shared filesystem), no rsync is needed — just verify:

```bash
python3 -c "
import kwcoco, sys
d = kwcoco.CocoDataset('/data/Public/VIAME/viame_sealions_2026/unpacked/train_norm_v2.kwcoco.zip')
imgs = list(d.imgs.values())
missing = [i for i in imgs[:50] if not __import__('pathlib').Path(i['file_name']).exists()]
print(f'{len(d.imgs)} images; {len(missing)}/50 sampled missing')
sys.exit(1 if missing else 0)
"
```

---

## 8. Fetch pretrained checkpoints

```bash
cd ~/code/kwcoco_detector_kit

# DINOv3-S COCO checkpoint (foundation backbone, 9.7M params, 50.9 AP)
# — this is the backbone used in gen004/gen005
bash projects/viame_sealions_2026/scripts/fetch_pretrained.sh deimv2_dinov3_s

# (Optional) HGNetv2-N COCO checkpoint (mobile baseline, 3.6M params)
bash projects/viame_sealions_2026/scripts/fetch_pretrained.sh deimv2_hgnetv2_n
```

Both are downloaded from Hugging Face. If the host has no internet
access, rsync them from arisia:

```bash
rsync -avh arisia:/data/users/jon.crall/pretrained_models/ \
    /data/users/$USER/pretrained_models/
```

---

## 9. Build the tile cache

The tile step is CPU-only and runs inside Docker. It is separate from
training so it only needs to happen once per dataset / tile config.

The canonical tile params for gen005+ are set in `paths.sh`:
- `KCD_TILE_SIZE=640`, `KCD_TILE_SOURCE_SCALES=1.0,0.5`
- 9-category universal tile (scheme-agnostic; apply_scheme collapses per-run)

If this host has Slurm:

```bash
cd ~/code/kwcoco_detector_kit

# Dry-run to preview TILE_HASH and job params
bash projects/viame_sealions_2026/scripts/submit_build_tiles.sh -n

# Submit the actual tile job
bash projects/viame_sealions_2026/scripts/submit_build_tiles.sh
```

If this host has no Slurm (standalone GPU box), run directly:

```bash
cd ~/code/kwcoco_detector_kit
SCRIPT_DIR=projects/viame_sealions_2026/scripts
source "$SCRIPT_DIR/paths.sh"

# Compute the same TILE_HASH the training job will look for
TILE_PARAMS_BODY="tile_mode=${KCD_TILE_MODE},tile_size=${KCD_TILE_SIZE},scales=${KCD_TILE_SOURCE_SCALES},stride=${KCD_TILE_STRIDE_FRAC},min_gt_area_frac=${KCD_TILE_MIN_GT_AREA_FRAC},min_keep_fraction=${KCD_TILE_MIN_KEEP_FRACTION},oversize_factor=${KCD_TILE_OVERSIZE_FACTOR},keep_negative=${KCD_TILE_KEEP_NEGATIVE},category_names=${KCD_TILE_CATEGORY_NAMES},writer_passthrough=false"
TILE_HASH=$(echo "$TILE_PARAMS_BODY" | md5sum | cut -c1-8)
TILE_DPATH="$KCD_TILE_CACHE_DPATH/_universal/$TILE_HASH"
echo "TILE_HASH=$TILE_HASH  TILE_DPATH=$TILE_DPATH"

docker run --rm \
    -v "$KCD_DATA_ROOT:$KCD_DATA_ROOT" \
    -v "$KCD_DATA_DPATH:$KCD_DATA_DPATH" \
    -v "$KCD_KIT_DPATH:$KCD_KIT_DPATH" \
    -w "$KCD_KIT_DPATH" \
    "$KCD_IMAGE" \
    python3 kwcoco_detector_kit/data/tile.py \
        --src "$KCD_UNIVERSAL_TRAIN_KWCOCO" \
        --dst_dpath "$TILE_DPATH" \
        --tile_size "$KCD_TILE_SIZE" \
        --mode "$KCD_TILE_MODE" \
        --source_scales "$KCD_TILE_SOURCE_SCALES" \
        --stride_frac "$KCD_TILE_STRIDE_FRAC" \
        --min_gt_area_frac "$KCD_TILE_MIN_GT_AREA_FRAC" \
        --min_keep_fraction "$KCD_TILE_MIN_KEEP_FRACTION" \
        --oversize_factor "$KCD_TILE_OVERSIZE_FACTOR" \
        --keep_negative "$KCD_TILE_KEEP_NEGATIVE" \
        --category_names "$KCD_TILE_CATEGORY_NAMES"
```

Expected output: `≥ 900,000 tiles` across the 9-category universal cache.
On arisia the cache was 114 GB; budget 120-150 GB on the new host.

---

## 10. Submit a training run

### With Slurm

```bash
cd ~/code/kwcoco_detector_kit

# pup_vs_nonpup baseline (dinov3_s, 2-GPU, gen005 recipe)
KCD_DEV_MOUNT_DEIMV2=1 \
    bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_s_2gpu_arisia_gen005_v2splits.sh

# single_sealion (same backbone, 1-class collapse)
KCD_DEV_MOUNT_DEIMV2=1 \
    bash projects/viame_sealions_2026/scripts/submit_train_single_sealion_deimv2_dinov3_s_2gpu_arisia_gen005_v2splits.sh

# Watch the queue
squeue -u $USER
```

`KCD_DEV_MOUNT_DEIMV2=1` mounts `tpl/DEIMv2/` from the host checkout
into the container, picking up the PIL truncated-image patch without a
full image rebuild. Remove it once the image is rebuilt with the patch
baked in.

### Without Slurm (standalone)

Write a new `submit_train_*.sh` or set the resource vars directly and
call `_launch_train.sh` inside Docker. Minimal example:

```bash
cd ~/code/kwcoco_detector_kit
source projects/viame_sealions_2026/scripts/paths.sh

export KCD_RUN_NAME=pup_vs_nonpup_deimv2_dinov3_s_2gpu_newhost_gen005
export KCD_SCHEME=pup_vs_nonpup
export KCD_NUM_GPUS=2
export KCD_PER_GPU_BATCH=8
export KCD_MAX_EPOCHS=30
export KCD_LR=5e-4
export KCD_LR_MIN=2.5e-5
export KCD_IMAGE_SIZE=640
export KCD_DEV_MOUNT_DEIMV2=1

docker run --rm --gpus all \
    -v "$KCD_DATA_ROOT:$KCD_DATA_ROOT" \
    -v "$KCD_DATA_DPATH:$KCD_DATA_DPATH" \
    -v "$KCD_KIT_DPATH:$KCD_KIT_DPATH" \
    $([ -n "${KCD_DEV_MOUNT_DEIMV2:-}" ] && echo "-v $KCD_KIT_DPATH/tpl/DEIMv2:/opt/DEIMv2") \
    -e KCD_RUN_NAME -e KCD_SCHEME -e KCD_NUM_GPUS \
    -e KCD_PER_GPU_BATCH -e KCD_MAX_EPOCHS -e KCD_LR -e KCD_LR_MIN \
    -e KCD_IMAGE_SIZE \
    -w "$KCD_KIT_DPATH" \
    "$KCD_IMAGE" \
    bash projects/viame_sealions_2026/scripts/_launch_train.sh
```

---

## 11. Sanity checks before training

```bash
# Docker can see the GPUs
docker run --rm --gpus all "$KCD_IMAGE" nvidia-smi

# Tile cache exists at the hash the training job will look for
cd ~/code/kwcoco_detector_kit
source projects/viame_sealions_2026/scripts/paths.sh
TILE_PARAMS_BODY="tile_mode=${KCD_TILE_MODE},tile_size=${KCD_TILE_SIZE},scales=${KCD_TILE_SOURCE_SCALES},stride=${KCD_TILE_STRIDE_FRAC},min_gt_area_frac=${KCD_TILE_MIN_GT_AREA_FRAC},min_keep_fraction=${KCD_TILE_MIN_KEEP_FRACTION},oversize_factor=${KCD_TILE_OVERSIZE_FACTOR},keep_negative=${KCD_TILE_KEEP_NEGATIVE},category_names=${KCD_TILE_CATEGORY_NAMES},writer_passthrough=false"
TILE_HASH=$(echo "$TILE_PARAMS_BODY" | md5sum | cut -c1-8)
echo "TILE_HASH=$TILE_HASH"
ls "$KCD_TILE_CACHE_DPATH/_universal/$TILE_HASH/" | wc -l   # expect ≥ 900000

# Check paths.sh resolves correctly
bash projects/viame_sealions_2026/scripts/check_paths.sh

# Zombie janitor — clear any leftover containers before submitting
KCD_KILL_ZOMBIES=1 bash projects/viame_sealions_2026/scripts/kit_zombie_janitor.sh
```

---

## Notes

**CUDA profile mismatch.** If `build_auto.sh` picks the wrong profile
(e.g. host has CUDA 12.4 but script emits a cu130 image), set
`KCD_DOCKER_CUDA_PROFILE=cu124` (or whichever matches your driver) and
rebuild.

**GPU count and memory.** `paths.sh` defaults to `KCD_CPUS_PER_TASK=2`
and `KCD_MEM=24G` per GPU (arisia shared-node budget). For a dedicated
box with no other users you can raise these, but the dinov3_s model
itself needs only ~11 GB VRAM at batch 8 / 640 px.

**First run always slow.** The apply_scheme step runs once per
run-name and caches the result to `scheme_applied/`. Subsequent
submissions of the same run-name skip it.

**Tile cache is transferable.** If the new host can reach arisia and
has the same TILE_HASH, just rsync the cache instead of rebuilding:

```bash
rsync -avhP arisia:/data/users/jon.crall/kcd_sealion/tile_cache/_universal/<TILE_HASH>/ \
    /data/users/$USER/kcd_sealion/tile_cache/_universal/<TILE_HASH>/
```

---

## Appendix A — host profile: aiq-gpu

Recorded 2026-06-06. This is a dedicated (non-shared) box, much more
powerful than arisia.

| Component | aiq-gpu | arisia (for contrast) |
|-----------|---------|-----------------------|
| GPU | 4× RTX PRO 6000 Blackwell Max-Q | 4× (Ampere) |
| VRAM/GPU | **96 GB** (97887 MiB) | ~40 GB |
| Compute cap | **12.0 (sm_120)** | 8.6 |
| Driver | 595.58.03 | 595.58.03 |
| CUDA toolkit | 13.2 (nvcc on PATH) | 13.2 |
| CPU | AMD EPYC 9554, 64C/128T | — |
| RAM | 251 GB | — |
| Fast storage | `nvme0n1` 1.7T (`/`, 1.3T free), `nvme1n1` 3.5T (unmounted SSD) | SSD |
| Bulk storage | `/data` = `md0` 37T HDD RAID (26T free) | — |

### A.1 Build the Docker image

aiq-gpu's CUDA/driver match arisia's cu132 profile exactly, but the
Blackwell GPUs are **sm_120**, not Ampere sm_86. Use the dedicated
Blackwell build script (it sets `TORCH_CUDA_ARCH_LIST=12.0`):

```bash
cd ~/code/kwcoco_detector_kit
bash docker/opengroundingdino/build_aiq_cuda132_blackwell.sh
```

`build_auto.sh` now also auto-detects the arch list from
`nvidia-smi --query-gpu=compute_cap`, so this works too and produces
the same kernels:

```bash
bash docker/opengroundingdino/build_auto.sh   # detects sm_120 automatically
```

Smoke-test that the compiled ops actually have a Blackwell kernel:

```bash
docker run --rm --gpus all kwcoco-detector-kit:ogdino-cu132-aiq python3 -c \
  "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
# expect: (12, 0)

# Exercise the custom MultiScaleDeformableAttention op (the one most
# likely to be missing an sm_120 kernel if the arch list was wrong):
docker run --rm --gpus all kwcoco-detector-kit:ogdino-cu132-aiq \
  python3 -c "import torch; from MultiScaleDeformableAttention import ms_deform_attn_forward; print('msda op OK')" \
  2>/dev/null || echo "NOTE: run a 1-step train smoke instead if import path differs"
```

> **"Too many open files (os error 24)" during the build.** uv compiles
> bytecode in parallel across all 128 cores, exhausting the default
> `nofile` soft limit (1024). The build scripts now pass
> `--ulimit nofile=1048576:1048576` to `docker build` to fix this. If
> your daemon rejects that hard limit, lower it:
> `BUILD_ULIMIT_NOFILE=65536:65536 bash docker/opengroundingdino/build_aiq_cuda132_blackwell.sh`.

### A.2 Storage layout

`/data` is a **37T HDD RAID** — fine for the corpus imagery (read-once,
sequential) but slow for the random-access tile cache. The fast storage
is the NVMe drives. Two options:

- **Simplest:** put everything under the root NVMe. The tile cache is
  ~114 GB and `/` has 1.3 TB free, so it fits with room to spare.

  ```bash
  export KCD_DATA_ROOT=/data/users/$USER          # if /data/users is on /
  # OR keep work area on root NVMe explicitly:
  mkdir -p /opt/kcd/$USER/kcd_sealion/tile_cache
  export KCD_TRAINING_ROOT=/opt/kcd/$USER/kcd_sealion
  ```

- **Most headroom:** mount the spare 3.5T `nvme1n1` SSD and put the
  tile cache + runs there.

  ```bash
  sudo mkfs.ext4 /dev/nvme1n1        # ONLY if blank — check `lsblk -f` first
  sudo mkdir -p /ssd-data
  sudo mount /dev/nvme1n1 /ssd-data
  sudo chown $USER:$USER /ssd-data
  mkdir -p /ssd-data/kcd_sealion/tile_cache
  export KCD_TRAINING_ROOT=/ssd-data/kcd_sealion
  ```

Put the **corpus imagery on `/data`** (HDD RAID, plenty of space):

```bash
export KCD_DATA_DPATH=/data/Public/VIAME/viame_sealions_2026
```

### A.3 Rsync the tile cache + corpus from arisia

Since aiq-gpu's tile params will resolve to the same TILE_HASH
(`paths.sh` defaults are host-independent), copy the prebuilt cache
rather than recomputing:

```bash
# On aiq-gpu — corpus bundles + imagery to the HDD RAID
rsync -avhP arisia:/data/Public/VIAME/viame_sealions_2026/unpacked/ \
    /data/Public/VIAME/viame_sealions_2026/unpacked/

# Tile cache to the fast NVMe (adjust dest to wherever KCD_TRAINING_ROOT points)
rsync -avhP arisia:/data/users/jon.crall/kcd_sealion/tile_cache/_universal/ \
    "$KCD_TRAINING_ROOT/tile_cache/_universal/"

# Pretrained checkpoints
rsync -avhP arisia:/data/users/jon.crall/pretrained_models/ \
    /data/users/$USER/pretrained_models/
```

### A.4 Right-sizing for 96 GB Blackwell GPUs

arisia's gen005 recipe uses `per_gpu_batch=8` at 640 px because its
~40 GB cards OOM higher (dinov3_s peaks ~11 GB at batch 8, but Mosaic
augmentation and EMA push the watermark up). aiq-gpu has **96 GB/GPU
and 4 dedicated GPUs**, so there's large headroom to test:

- **Bigger per-GPU batch** (16 → 32+) for faster epochs / better BN-free
  stability. Scale LR with batch.
- **Higher input resolution** (640 → 896/1024) — pups are the binding
  constraint (median ~46 px) and resolution directly helps small-object
  recall. This is the most promising lever to test here.
- **Larger backbone** (dinov3_s → dinov3_b/l) if S-tier saturates.
- **No shared-node budget caps** — drop the `KCD_CPUS_PER_TASK=2`/
  `KCD_MEM=24G` arisia courtesy limits; the EPYC has 128 threads.

These are candidates to discuss before committing a run — the cluster's
value is testing the resolution/capacity axis that arisia couldn't fit.
