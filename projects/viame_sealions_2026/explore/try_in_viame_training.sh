#!/bin/bash
#ssh aiq-gpu

export GIRDER_API='https://viame.kitware.com/api/v1'

export WORK_ROOT=/data/users/$USER/viame-sealion-rfdetr
export DOWNLOAD_ROOT=$WORK_ROOT/girder-download
export DATA_DIR=$WORK_ROOT/Training-SeaLion-DETR
export VIAME_ROOT=$WORK_ROOT/viame
export VIAME_INSTALL=$VIAME_ROOT/install
export LOCAL_DATA_DIR=/data/users/$USER/viame-sealion-rfdetr/Training-SeaLion-DETR
export REMOTE_DATA_DIR=/data/matt.dawkins/Runs/Training-SeaLion-DETR

mkdir -p "$WORK_ROOT" "$DOWNLOAD_ROOT" "$DATA_DIR" "$VIAME_ROOT"

uv pip install girder-client

girder-client --api-url "$GIRDER_API" --version

### GRAB DATA

# TODO:
# It would be better if we knew how to set this up from cannonical data
# instead.

mkdir -p "$LOCAL_DATA_DIR"

rsync -avhP --info=progress2 \
    --exclude='deep_training/' \
    --exclude='augmented_images/' \
    --exclude='category_models/' \
    --exclude='train_log.txt' \
    --exclude='jobid.txt' \
    --exclude='*.zip' \
    numenor:"$REMOTE_DATA_DIR"/ \
    "$LOCAL_DATA_DIR"/

# Replace the bad symlink trees.
rm -rf "$LOCAL_DATA_DIR/train" "$LOCAL_DATA_DIR/vali"

# Copy symlink targets as real files from numenor.
rsync -avhPL --copy-links --info=progress2 \
    numenor:"$REMOTE_DATA_DIR/train" \
    "$LOCAL_DATA_DIR"/

rsync -avhPL --copy-links --info=progress2 \
    numenor:"$REMOTE_DATA_DIR/vali" \
    "$LOCAL_DATA_DIR"/



### GRAB VIAME BINARY

# TODO: probably better to use girder-cli?
#export GIRDER_ID='6a3e990112d300cfa83b0c8f'
#mkdir -p "$DOWNLOAD_ROOT/$GIRDER_ID"
#girder-client \
#    --api-url "$GIRDER_API" \
#    download "$GIRDER_ID" "$DOWNLOAD_ROOT/$GIRDER_ID"

#find "$DOWNLOAD_ROOT/$GIRDER_ID" -maxdepth 4 -type f | sort | head -n 80
#find "$DOWNLOAD_ROOT/$GIRDER_ID" -maxdepth 4 -type d | sort | head -n 80


# Curl probably works
export VIAME_ITEM_ID=6a3e990112d300cfa83b0c8e
export DOWNLOAD_DIR=/data/users/$USER/viame-sealion-rfdetr/downloads
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"
curl -L \
    -o viame_${VIAME_ITEM_ID}.download \
    "https://data.kitware.com/api/v1/item/${VIAME_ITEM_ID}/download"
file viame_${VIAME_ITEM_ID}.download
ls -lh viame_${VIAME_ITEM_ID}.download




### UNPACK VIAME

export VIAME_ARCHIVE="$DOWNLOAD_DIR/viame_${VIAME_ITEM_ID}.download"
echo "VIAME_ARCHIVE = $VIAME_ARCHIVE"

echo "VIAME_ARCHIVE=$VIAME_ARCHIVE"
echo "VIAME_ROOT=$VIAME_ROOT"

file "$VIAME_ARCHIVE"
ls -lh "$VIAME_ARCHIVE"

rm -rf "$VIAME_ROOT"
mkdir -p "$VIAME_ROOT"

# List first, to verify it is a tar archive.
tar -tf "$VIAME_ARCHIVE" | head -n 40

# Extract.
tar -xf "$VIAME_ARCHIVE" -C "$VIAME_ROOT"

# Find the real VIAME install dir.
export VIAME_INSTALL="$(dirname "$(find "$VIAME_ROOT" -name setup_viame.sh | head -n 1)")"

echo "VIAME_INSTALL=$VIAME_INSTALL"
test -f "$VIAME_INSTALL/setup_viame.sh"

find "$VIAME_ROOT" -maxdepth 3 -type f -name setup_viame.sh -print


source "$VIAME_INSTALL/setup_viame.sh"

which viame_train_detector
viame_train_detector --help | sed -n '1,120p'



##### GET CONFIGS

export VIAME_INSTALL="$(dirname "$(find "$VIAME_ROOT" -name setup_viame.sh | head -n 1)")"
source "$VIAME_INSTALL/setup_viame.sh"

mkdir -p "$VIAME_INSTALL/configs/pipelines"

curl -L \
  -o "$VIAME_INSTALL/configs/pipelines/train_detector_sealion_rf_detr_l_1024_chip.conf" \
  'https://raw.githubusercontent.com/VIAME/VIAME/main/configs/add-ons/sea-lion/train_detector_sealion_rf_detr_l_1024_chip.conf'

curl -L \
  -o "$VIAME_INSTALL/configs/pipelines/train_detector_sealion_rf_detr_l_1344_chip_90gb.conf" \
  'https://raw.githubusercontent.com/VIAME/VIAME/main/configs/add-ons/sea-lion/train_detector_sealion_rf_detr_l_1344_chip_90gb.conf'

test -f "$VIAME_INSTALL/configs/pipelines/train_detector_sealion_rf_detr_l_1024_chip.conf"
test -f "$VIAME_INSTALL/configs/pipelines/common_train_detector.conf"
test -f "$VIAME_INSTALL/configs/pipelines/templates/embedded_default.pipe"


##### VERIFY DATA + LABELS

export DATA_DIR="$LOCAL_DATA_DIR"
export LABELS="$DATA_DIR/labels_adult_pup.txt"
export CONFIG="$VIAME_INSTALL/configs/pipelines/train_detector_sealion_rf_detr_l_1024_chip.conf"

echo "DATA_DIR=$DATA_DIR"
echo "LABELS=$LABELS"
echo "CONFIG=$CONFIG"
echo "VIAME_INSTALL=$VIAME_INSTALL"

test -f "$VIAME_INSTALL/setup_viame.sh"
test -f "$CONFIG"
test -d "$DATA_DIR/train"
test -d "$DATA_DIR/vali"

find "$DATA_DIR" -maxdepth 2 -type d | sort | head -n 80

echo "CSV examples:"
find "$DATA_DIR/train" "$DATA_DIR/vali" -type f -iname '*.csv' | sort | head -n 40

echo "Image examples:"
find "$DATA_DIR/train" "$DATA_DIR/vali" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' \) \
    | sort | head -n 40


# WRite labels

if [ ! -f "$LABELS" ]; then
    cat > "$LABELS" <<'EOF'
# Sea Lion 2-class label map (adult / pup) for viame_train_detector --labels.
# Format: first token = output category; remaining tokens = input categories
# merged into it. Input categories NOT listed here are excluded from training:
#   - negative          (still used as background via hard_negative_categories)
#   - northern_fur_seal (the fur seal category)
#   - unknown           (if present)
adult juvenile bull female subadult_male dead_nonpup
pup dead_pup
EOF
fi

cat "$LABELS"
test -f "$LABELS"




##### WRITE TRAINING LAUNCHER

mkdir -p "$WORK_ROOT/bin" /data/users/$USER/slurm_logs

cat > "$WORK_ROOT/bin/run_sealion_training_aiq.sh" <<'EOF'
#!/usr/bin/env bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --account=noaa
#SBATCH --mem=96000
#SBATCH --partition=priority
#SBATCH --job-name=sealion-rfdetr
#SBATCH --output=/data/users/%u/slurm_logs/%x-%j.out

set -euo pipefail

DATA_DIR="${DATA_DIR:-/data/users/${USER}/viame-sealion-rfdetr/Training-SeaLion-DETR}"
VIAME_INSTALL="${VIAME_INSTALL:-/data/users/${USER}/viame-sealion-rfdetr/viame/viame}"
LABELS="${LABELS:-${DATA_DIR}/labels_adult_pup.txt}"
CONFIG="${CONFIG:-${VIAME_INSTALL}/configs/pipelines/train_detector_sealion_rf_detr_l_1024_chip.conf}"

[ -f "${VIAME_INSTALL}/setup_viame.sh" ] || { echo "ERROR: missing ${VIAME_INSTALL}/setup_viame.sh" >&2; exit 1; }
[ -f "${CONFIG}" ] || { echo "ERROR: missing CONFIG=${CONFIG}" >&2; exit 1; }
[ -d "${DATA_DIR}/train" ] || { echo "ERROR: missing ${DATA_DIR}/train" >&2; exit 1; }
[ -d "${DATA_DIR}/vali" ] || { echo "ERROR: missing ${DATA_DIR}/vali" >&2; exit 1; }
[ -f "${LABELS}" ] || { echo "ERROR: missing LABELS=${LABELS}" >&2; exit 1; }

exec > >(tee -a "${DATA_DIR}/train_log.txt") 2>&1

source "${VIAME_INSTALL}/setup_viame.sh"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KWIVER_DEFAULT_LOG_LEVEL=info

cd "${DATA_DIR}"

[ -n "${SLURM_JOB_ID:-}" ] && echo "${SLURM_JOB_ID}" > "${DATA_DIR}/jobid.txt"

echo "=== TRAIN START $(date) ==="
echo "HOST=$(hostname)"
echo "DATA_DIR=${DATA_DIR}"
echo "VIAME_INSTALL=${VIAME_INSTALL}"
echo "CONFIG=${CONFIG}"
echo "LABELS=${LABELS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

OUT_STEM="$(basename "${CONFIG}" .conf)"
OUT_FILE="${DATA_DIR}/sealion_${OUT_STEM}.zip"

rm -rf "${DATA_DIR}/deep_training" "${DATA_DIR}/augmented_images"

viame_train_detector \
  -i "${DATA_DIR}/train" \
  -v "${DATA_DIR}/vali" \
  -c "${CONFIG}" \
  --labels "${LABELS}" \
  --output-file "${OUT_FILE}" \
  --no-query

echo "=== TRAIN DONE $(date) ==="
echo "OUT_FILE=${OUT_FILE}"
ls -lh "${OUT_FILE}"
EOF

chmod +x "$WORK_ROOT/bin/run_sealion_training_aiq.sh"



### Patch:
python - <<'PY'
from pathlib import Path
fpath = Path('/data/users/jon.crall/viame-sealion-rfdetr/bin/run_sealion_training_aiq.sh')
text = fpath.read_text()
text = text.replace(
    'VIAME_INSTALL="${VIAME_INSTALL:-/data/users/${USER}/viame-sealion-rfdetr/viame/viame}"',
    'VIAME_INSTALL="${VIAME_INSTALL:-/data/users/${USER}/viame-sealion-rfdetr/viame/install}"',
)
fpath.write_text(text)
PY

grep -n 'VIAME_INSTALL=' "$WORK_ROOT/bin/run_sealion_training_aiq.sh"


### PREFLIGH CHECK

export WORK_ROOT=/data/users/$USER/viame-sealion-rfdetr
export DATA_DIR=$WORK_ROOT/Training-SeaLion-DETR
export VIAME_ROOT=$WORK_ROOT/viame
export VIAME_INSTALL="$(dirname "$(find "$VIAME_ROOT" -name setup_viame.sh | head -n 1)")"
export LABELS="$DATA_DIR/labels_adult_pup.txt"
export CONFIG="$VIAME_INSTALL/configs/pipelines/train_detector_sealion_rf_detr_l_1024_chip.conf"

source "$VIAME_INSTALL/setup_viame.sh"

echo "DATA_DIR=$DATA_DIR"
echo "VIAME_INSTALL=$VIAME_INSTALL"
echo "LABELS=$LABELS"
echo "CONFIG=$CONFIG"

test -x "$(command -v viame_train_detector)"
test -f "$VIAME_INSTALL/setup_viame.sh"
test -f "$CONFIG"
test -f "$LABELS"
test -d "$DATA_DIR/train"
test -d "$DATA_DIR/vali"
test -f "$WORK_ROOT/bin/run_sealion_training_aiq.sh"

echo "Image count:"
find "$DATA_DIR/train" "$DATA_DIR/vali" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' \) \
    | wc -l

echo "Groundtruth files:"
find "$DATA_DIR/train" "$DATA_DIR/vali" -type f \
    \( -iname '*.csv' -o -iname '*.json' -o -iname '*.kwcoco.json' \) \
    | sort | head -n 40

echo "Broken symlinks:"
find "$DATA_DIR/train" "$DATA_DIR/vali" -xtype l | head

echo "Labels:"
cat "$LABELS"

echo "Help smoke:"
viame_train_detector --help | head -n 40


### RUN RUN RUN


mkdir -p "/data/users/$USER/slurm_logs"

jid=$(sbatch --parsable \
  --export=ALL,DATA_DIR="$DATA_DIR",VIAME_INSTALL="$VIAME_INSTALL",LABELS="$LABELS",CONFIG="$CONFIG" \
  "$WORK_ROOT/bin/run_sealion_training_aiq.sh")

echo "Submitted $jid"
squeue -j "$jid"
tail -f "/data/users/$USER/slurm_logs/sealion-rfdetr-$jid.out"



###

sed -i '/^#SBATCH --account=/d' "$WORK_ROOT/bin/run_sealion_training_aiq.sh"

grep -n '^#SBATCH' "$WORK_ROOT/bin/run_sealion_training_aiq.sh"

jid=$(sbatch --parsable \
  --partition=priority \
  --export=ALL,DATA_DIR="$DATA_DIR",VIAME_INSTALL="$VIAME_INSTALL",LABELS="$LABELS",CONFIG="$CONFIG" \
  "$WORK_ROOT/bin/run_sealion_training_aiq.sh") || exit $?

echo "Submitted $jid"
squeue -j "$jid"
tail -f "/data/users/$USER/slurm_logs/sealion-rfdetr-$jid.out"

jid=$(sbatch --parsable \
  --export=ALL,DATA_DIR="$DATA_DIR",VIAME_INSTALL="$VIAME_INSTALL",LABELS="$LABELS",CONFIG="$CONFIG" \
  "$WORK_ROOT/bin/run_sealion_training_aiq.sh") || exit $?
echo "jid = $jid"

squeue



### RUn outside of slurm:


export WORK_ROOT=/data/users/$USER/viame-sealion-rfdetr
export DATA_DIR=$WORK_ROOT/Training-SeaLion-DETR
export VIAME_ROOT=$WORK_ROOT/viame
export VIAME_INSTALL="$(dirname "$(find "$VIAME_ROOT" -name setup_viame.sh | head -n 1)")"
export LABELS="$DATA_DIR/labels_adult_pup.txt"
export CONFIG="$VIAME_INSTALL/configs/pipelines/train_detector_sealion_rf_detr_l_1024_chip.conf"

source "$VIAME_INSTALL/setup_viame.sh"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KWIVER_DEFAULT_LOG_LEVEL=info

cd "$DATA_DIR"

test -f "$VIAME_INSTALL/setup_viame.sh"
test -f "$CONFIG"
test -f "$LABELS"
test -d "$DATA_DIR/train"
test -d "$DATA_DIR/vali"

nvidia-smi

viame_train_detector \
  -i "$DATA_DIR/train" \
  -v "$DATA_DIR/vali" \
  -c "$CONFIG" \
  --labels "$LABELS" \
  --output-file "$DATA_DIR/sealion_train_detector_sealion_rf_detr_l_1024_chip.zip" \
  --no-query \
  2>&1 | tee -a "$DATA_DIR/train_log_direct_1024.txt"
