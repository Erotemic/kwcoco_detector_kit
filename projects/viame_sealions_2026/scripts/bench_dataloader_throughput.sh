#!/usr/bin/env bash
# Throughput benchmark: kwcoco baseline vs. kwcoco_dataloader's
# WebDataset detection reader, on real sealion tiles. Designed for
# namek so we can profile / compare without burning arisia's GPUs.
#
# Runs three variants in sequence, all against the same source tile
# bundle and the same shard set:
#
#   1. baseline (kwcoco CocoDetection-style, PIL.open per __getitem__)
#   2. webdataset (kwcoco_dataloader.readers.detection pipeline)
#   3. webdataset + LINE_PROFILE=1 (line_profiler timings per hot fn)
#
# Each run writes a log to $KCD_BENCH_ROOT/<variant>.log; the third
# also drops profile_output.lprof + profile_output.txt in there. A
# final summary contrasts samples/sec across the variants.
#
# Defaults (override via env):
#   KCD_BENCH_N_SAMPLES   500
#   KCD_BENCH_WORKERS     4
#   KCD_BENCH_BATCH_SIZE  16
#   KCD_BENCH_ROOT        $KCD_TRAINING_ROOT/bench/<timestamp>
#   PYTHON_BIN            python (your active venv)
#   KCD_BENCH_TILES       auto-detect: most recent
#                         $KCD_TILE_CACHE_DPATH/_universal/*/tiles.kwcoco.zip
#   KCD_KWCOCO_DATALOADER_DPATH   path to a local kwcoco_dataloader
#                         checkout to pip install -e from
#
# Usage (from kit root):
#   bash projects/viame_sealions_2026/scripts/bench_dataloader_throughput.sh
#
# Or with overrides:
#   KCD_BENCH_N_SAMPLES=2000 KCD_BENCH_WORKERS=8 \
#       bash projects/viame_sealions_2026/scripts/bench_dataloader_throughput.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
N_SAMPLES="${KCD_BENCH_N_SAMPLES:-500}"
WORKERS="${KCD_BENCH_WORKERS:-4}"
BATCH_SIZE="${KCD_BENCH_BATCH_SIZE:-16}"

# ----------------------------------------------------------------
# Step 1: dependency check
# ----------------------------------------------------------------
echo "=== Dependency check ==="
required_deps=(kwcoco_dataloader webdataset line_profiler)
optional_deps=(torch)

missing_required=()
for dep in "${required_deps[@]}"; do
    if ! "$PYTHON_BIN" -c "import $dep" >/dev/null 2>&1; then
        missing_required+=("$dep")
    fi
done

if [ ${#missing_required[@]} -gt 0 ]; then
    echo "ERROR: missing required python deps in $PYTHON_BIN: ${missing_required[*]}" >&2
    echo >&2
    echo "Suggested fix:" >&2
    if [ -n "${KCD_KWCOCO_DATALOADER_DPATH:-}" ]; then
        echo "  uv pip install -e \"$KCD_KWCOCO_DATALOADER_DPATH\"" >&2
    else
        echo "  uv pip install webdataset line_profiler" >&2
        echo "  uv pip install -e \$HOME/code/kwcoco_dataloader  # for the local checkout" >&2
    fi
    exit 1
fi

# torch is optional — without it we use the bench's --no_loader mode
# (no DataLoader worker fanout; bypass batching). Still meaningful as
# a single-process IO comparison; not directly comparable to a
# multi-worker production setup.
LOADER_FLAG=()
HAS_TORCH=1
if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
    HAS_TORCH=0
    LOADER_FLAG=(--no_loader)
fi

echo "  python:  $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
echo "  required OK: ${required_deps[*]}"
if [ "$HAS_TORCH" = "1" ]; then
    echo "  optional OK: torch (using torch.DataLoader for worker fanout)"
else
    echo "  optional missing: torch — falling back to --no_loader mode"
    echo "    (single-process IO comparison; numbers won't reflect multi-worker prod)"
fi

# ----------------------------------------------------------------
# Step 2: resolve source tile bundle
# ----------------------------------------------------------------
if [ -n "${KCD_BENCH_TILES:-}" ]; then
    TILES_KWCOCO="$KCD_BENCH_TILES"
else
    # Pick the most-recently-modified universal tile bundle.
    candidate="$(ls -dt "$KCD_TILE_CACHE_DPATH"/_universal/*/tiles.kwcoco.zip 2>/dev/null | head -n1 || true)"
    if [ -z "$candidate" ]; then
        echo "ERROR: no universal tile bundle under $KCD_TILE_CACHE_DPATH/_universal/" >&2
        echo "  Set KCD_BENCH_TILES to point at a tiles.kwcoco.zip, or run the" >&2
        echo "  tile step of any submit_train_* wrapper to populate the cache." >&2
        exit 1
    fi
    TILES_KWCOCO="$candidate"
fi
kcd_require_path "tile bundle" "$TILES_KWCOCO" || exit 1

# ----------------------------------------------------------------
# Step 3: prep bench output dir
# ----------------------------------------------------------------
TS="$(date +%Y%m%dT%H%M%S)"
BENCH_ROOT="${KCD_BENCH_ROOT:-$KCD_TRAINING_ROOT/bench/$TS}"
mkdir -p "$BENCH_ROOT"

# Shards live next to the source tile bundle's hash to allow reuse
# across benchmark runs.
SHARDS_PARENT="$(dirname "$TILES_KWCOCO")"
SHARDS_DPATH="${KCD_BENCH_SHARDS:-$SHARDS_PARENT/wds_shards}"

echo
echo "=== Config ==="
echo "  tiles_kwcoco:    $TILES_KWCOCO"
echo "  shards_dpath:    $SHARDS_DPATH"
echo "  bench_root:      $BENCH_ROOT"
echo "  n_samples:       $N_SAMPLES"
echo "  workers:         $WORKERS"
echo "  batch_size:      $BATCH_SIZE"

# Stash the config alongside the results for later reference.
{
    echo "tiles_kwcoco=$TILES_KWCOCO"
    echo "shards_dpath=$SHARDS_DPATH"
    echo "n_samples=$N_SAMPLES"
    echo "workers=$WORKERS"
    echo "batch_size=$BATCH_SIZE"
    echo "python=$($PYTHON_BIN --version 2>&1)"
    echo "host=$(hostname)"
    echo "timestamp=$TS"
} > "$BENCH_ROOT/config.txt"

# ----------------------------------------------------------------
# Step 4: build WebDataset shards (skip if already built)
# ----------------------------------------------------------------
echo
echo "=== Build / verify WebDataset shards ==="
shards_exist=0
if [ -d "$SHARDS_DPATH" ]; then
    tar_count=$(find "$SHARDS_DPATH" -name '*.tar' 2>/dev/null | wc -l)
    if [ "$tar_count" -gt 0 ]; then
        shards_exist=1
        echo "  reusing $tar_count existing tar shards under $SHARDS_DPATH"
    fi
fi
if [ "$shards_exist" = "0" ]; then
    echo "  building shards (this is the one-time pack cost) ..."
    "$PYTHON_BIN" -m kwcoco_dataloader.cli.build_detection_webdataset \
        --in_fpath "$TILES_KWCOCO" \
        --out_dpath "$SHARDS_DPATH" \
        --maxcount 5000 \
        --maxsize_mb 1024 \
        2>&1 | tee "$BENCH_ROOT/build_shards.log"
fi

# ----------------------------------------------------------------
# Step 5: run bench variants
# ----------------------------------------------------------------
COMMON_BENCH_ARGS=(
    --kwcoco "$TILES_KWCOCO"
    --shards "$SHARDS_DPATH"
    --n_samples "$N_SAMPLES"
    --workers "$WORKERS"
    --batch_size "$BATCH_SIZE"
    "${LOADER_FLAG[@]}"
)

# Two-pass warmup: run a small bench first to populate the page cache
# uniformly across both paths, so first-real-pass isn't penalized.
echo
echo "=== Warmup pass (n=64, throwaway) ==="
"$PYTHON_BIN" -m kwcoco_dataloader.benchmarks.bench_detection_throughput \
    --kwcoco "$TILES_KWCOCO" \
    --shards "$SHARDS_DPATH" \
    --n_samples 64 \
    --workers 0 \
    --batch_size 4 \
    "${LOADER_FLAG[@]}" \
    > "$BENCH_ROOT/warmup.log" 2>&1 || \
    echo "  (warmup failed — not fatal)"

# (a) baseline alone
echo
echo "=== [a] baseline (no profiler) ==="
"$PYTHON_BIN" -m kwcoco_dataloader.benchmarks.bench_detection_throughput \
    "${COMMON_BENCH_ARGS[@]}" --skip webdataset \
    2>&1 | tee "$BENCH_ROOT/a_baseline.log"

# (b) webdataset alone
echo
echo "=== [b] webdataset (no profiler) ==="
"$PYTHON_BIN" -m kwcoco_dataloader.benchmarks.bench_detection_throughput \
    "${COMMON_BENCH_ARGS[@]}" --skip baseline \
    2>&1 | tee "$BENCH_ROOT/b_webdataset.log"

# (c) webdataset with LINE_PROFILE=1
echo
echo "=== [c] webdataset (LINE_PROFILE=1) ==="
# Run from BENCH_ROOT so line_profiler's output files land there.
(
    cd "$BENCH_ROOT"
    LINE_PROFILE=1 "$PYTHON_BIN" \
        -m kwcoco_dataloader.benchmarks.bench_detection_throughput \
        "${COMMON_BENCH_ARGS[@]}" --skip baseline \
        2>&1 | tee "$BENCH_ROOT/c_webdataset_profile.log"
)

# ----------------------------------------------------------------
# Step 6: summary
# ----------------------------------------------------------------
echo
echo "=== Summary ==="

parse_samples_per_sec() {
    # Pull the "<NN.N> samples/s" number from a bench log.
    local fp="$1"
    grep -oE '[0-9]+\.[0-9]+ samples/s' "$fp" | head -n1 \
        | awk '{print $1}'
}

baseline_sps=$(parse_samples_per_sec "$BENCH_ROOT/a_baseline.log" || echo "n/a")
wds_sps=$(parse_samples_per_sec "$BENCH_ROOT/b_webdataset.log" || echo "n/a")
wds_prof_sps=$(parse_samples_per_sec "$BENCH_ROOT/c_webdataset_profile.log" || echo "n/a")

printf '  %-32s %s samples/s\n' "baseline (kwcoco)"           "${baseline_sps:-n/a}"
printf '  %-32s %s samples/s\n' "webdataset"                   "${wds_sps:-n/a}"
printf '  %-32s %s samples/s\n' "webdataset + LINE_PROFILE=1"  "${wds_prof_sps:-n/a}"

if [ -n "${baseline_sps:-}" ] && [ -n "${wds_sps:-}" ] && \
   [ "$baseline_sps" != "n/a" ] && [ "$wds_sps" != "n/a" ]; then
    speedup=$(awk -v a="$baseline_sps" -v b="$wds_sps" \
              'BEGIN { if (a > 0) printf "%.2f", b/a; else print "n/a"; }')
    printf '\n  webdataset speedup vs baseline: %sx\n' "$speedup"
fi

if [ -n "${wds_sps:-}" ] && [ -n "${wds_prof_sps:-}" ] && \
   [ "$wds_sps" != "n/a" ] && [ "$wds_prof_sps" != "n/a" ]; then
    overhead=$(awk -v n="$wds_sps" -v p="$wds_prof_sps" \
              'BEGIN { if (n > 0) printf "%.1f", 100 * (1 - p/n); else print "n/a"; }')
    printf '  LINE_PROFILE=1 overhead: %s%%\n' "$overhead"
fi

echo
echo "Artifacts:"
echo "  $BENCH_ROOT/"
ls -1 "$BENCH_ROOT" | sed 's/^/    /'

if [ -f "$BENCH_ROOT/profile_output.lprof" ]; then
    echo
    echo "Inspect the profile with:"
    echo "  $PYTHON_BIN -m line_profiler -rtmz $BENCH_ROOT/profile_output.lprof"
fi
