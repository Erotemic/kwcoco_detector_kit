#!/usr/bin/env bash
# Internal launcher. Use a versioned run_*.sh entry point instead.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

if [ -z "$VF_RUN_NAME" ]; then
    echo "ERROR: VF_RUN_NAME is not set"
    exit 1
fi

if [ ! -d "$VF_INPUT_DPATH" ]; then
    echo "ERROR: input directory does not exist: $VF_INPUT_DPATH"
    exit 1
fi

if [ ! -f "$VF_CURRENT_VIAME_LINK/setup_viame.sh" ]; then
    echo "ERROR: active VIAME install is missing: $VF_CURRENT_VIAME_LINK"
    exit 1
fi

if [ ! -f "$VF_CONFIG_FPATH" ]; then
    echo "ERROR: selected VIAME config does not exist: $VF_CONFIG_FPATH"
    echo "Select one with: bash $SCRIPT_DIR/setup_config.sh --list"
    exit 1
fi

mkdir -p "$VF_RUNS_DPATH" "$VF_LOGS_DPATH"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_GROUP_DPATH="$VF_RUNS_DPATH/$VF_RUN_NAME"
RUN_DPATH="$RUN_GROUP_DPATH/attempt_$RUN_STAMP"
mkdir -p "$RUN_DPATH"

rm -f "$RUN_GROUP_DPATH/latest"
ln -s "$RUN_DPATH" "$RUN_GROUP_DPATH/latest"

CONFIG_SOURCE_FPATH="$VF_CONFIG_FPATH"
CONFIG_SNAPSHOT_FPATH="$RUN_DPATH/$VF_CONFIG_NAME"
cp "$CONFIG_SOURCE_FPATH" "$CONFIG_SNAPSHOT_FPATH"

if [ -n "$VF_ENTRYPOINT_FPATH" ] && [ -f "$VF_ENTRYPOINT_FPATH" ]; then
    cp "$VF_ENTRYPOINT_FPATH" "$RUN_DPATH/"
fi

if [ -f "$VF_CONFIG_SELECTION_FPATH" ]; then
    cp "$VF_CONFIG_SELECTION_FPATH" "$RUN_DPATH/selected_config.env"
fi

source "$VF_CURRENT_VIAME_LINK/setup_viame.sh"

export CUDA_VISIBLE_DEVICES="${VF_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export KWIVER_DEFAULT_LOG_LEVEL="${KWIVER_DEFAULT_LOG_LEVEL:-info}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VIAME_INSTALL_REAL="$(readlink -f "$VF_CURRENT_VIAME_LINK")"
VIAME_EXE="$(command -v viame_train_detector)"

{
    echo "run_name=$VF_RUN_NAME"
    echo "run_dpath=$RUN_DPATH"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "user=$USER"
    echo "input_dpath=$VF_INPUT_DPATH"
    echo "viame_install=$VIAME_INSTALL_REAL"
    echo "viame_exe=$VIAME_EXE"
    echo "config_name=$VF_CONFIG_NAME"
    echo "config_origin=${VF_CONFIG_ORIGIN:-unknown}"
    echo "config_source=$CONFIG_SOURCE_FPATH"
    echo "config_snapshot=$CONFIG_SNAPSHOT_FPATH"
    echo "config_selection_state=$VF_CONFIG_SELECTION_FPATH"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "kwiver_default_log_level=$KWIVER_DEFAULT_LOG_LEVEL"
    echo "pytorch_cuda_alloc_conf=$PYTORCH_CUDA_ALLOC_CONF"
    echo "project_git_commit=$(git -C "$VF_PROJECT_DPATH" rev-parse HEAD 2>/dev/null)"
} > "$RUN_DPATH/run_manifest.txt"

sha256sum "$CONFIG_SOURCE_FPATH" > "$RUN_DPATH/config.sha256"

if [ -f "$VIAME_INSTALL_REAL/.viame_archive.sha256" ]; then
    cp "$VIAME_INSTALL_REAL/.viame_archive.sha256" "$RUN_DPATH/viame_archive.sha256"
fi
if [ -f "$VIAME_INSTALL_REAL/.viame_install_info.txt" ]; then
    cp "$VIAME_INSTALL_REAL/.viame_install_info.txt" "$RUN_DPATH/viame_install_info.txt"
fi

nvidia-smi > "$RUN_DPATH/nvidia_smi.txt"

cat > "$RUN_DPATH/command.sh" <<COMMAND
#!/usr/bin/env bash
source "$VIAME_INSTALL_REAL/setup_viame.sh"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export KWIVER_DEFAULT_LOG_LEVEL="$KWIVER_DEFAULT_LOG_LEVEL"
export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
cd "$RUN_DPATH"
viame_train_detector \\
    -i "$VF_INPUT_DPATH" \\
    -c "$CONFIG_SOURCE_FPATH" \\
    --threshold 0.0 \\
    --output-file "$RUN_DPATH/fish_detector.zip"
COMMAND
chmod +x "$RUN_DPATH/command.sh"

cd "$RUN_DPATH"

echo "Run directory: $RUN_DPATH"
echo "Log: $RUN_DPATH/train.log"
echo "Selected config: $CONFIG_SOURCE_FPATH"
echo "Command snapshot: $RUN_DPATH/command.sh"
echo "Starting at $(date)"

viame_train_detector \
    -i "$VF_INPUT_DPATH" \
    -c "$CONFIG_SOURCE_FPATH" \
    --threshold 0.0 \
    --output-file "$RUN_DPATH/fish_detector.zip" \
    2>&1 | tee "$RUN_DPATH/train.log"

STATUS=${PIPESTATUS[0]}
echo "$STATUS" > "$RUN_DPATH/exit_code.txt"
echo "finished_at=$(date --iso-8601=seconds)" >> "$RUN_DPATH/run_manifest.txt"
echo "exit_code=$STATUS" >> "$RUN_DPATH/run_manifest.txt"

echo "Finished at $(date) with exit code $STATUS"
exit "$STATUS"
