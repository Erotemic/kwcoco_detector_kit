#!/usr/bin/env bash
# Batch WHOLE-IMAGE + TILED rescore over EVERY trained run we have on disk,
# so the comparison table has proper numbers for all models (not just the
# DEIMv2-internal whole-image@640 eval, which understates small objects).
#
# Each model is scored on its OWN scheme_applied/test.kwcoco.zip, twice:
#   <run>/tiled_compare/eval/wholeimage/eval/detect_metrics.json   (scaled full-image)
#   <run>/tiled_compare/eval/tiled/eval/detect_metrics.json        (windowed)
# Per-scheme category_names + NFS distractor come from docs/class_schemes.yaml.
#
# Runs on a non-slurm host (namek) — loops on the HOST and shells out to
# rescore_tiled_compare.sh (one `docker run` per model). Long: ~tens of min
# per model x ~17 models. Run in tmux + tee (NOT nohup):
#
#   tmux new -s rescore
#   bash projects/viame_sealions_2026/scripts/rescore_all.sh \
#       2>&1 | tee /data/users/jon.crall/kcd_sealion/rescore_all.log
#
# Restrict to specific runs by passing their names as args. Re-run is cheap:
# models that already have BOTH metrics are skipped (KCD_FORCE_RESCORE=1 to
# recompute). The whole-image pass is reused if present; the tiled pass is
# always (re)computed by the underlying tool.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./paths.sh
source "$SCRIPT_DIR/paths.sh"

# scheme -> "category_names_csv|distractors_csv" (from docs/class_schemes.yaml)
scheme_args() {
    case "$1" in
        pup_vs_nonpup) echo "pup,nonpup_sealion|" ;;
        single_sealion) echo "sealion|" ;;
        lifestage_6cls) echo "bull,subadult_male,female,juvenile,pup,northern_fur_seal|northern_fur_seal" ;;
        *) echo "" ;;
    esac
}

scheme_of() {
    case "$1" in
        pup_vs_nonpup_*) echo pup_vs_nonpup ;;
        single_sealion_*) echo single_sealion ;;
        lifestage_6cls_*) echo lifestage_6cls ;;
        *) echo "" ;;
    esac
}

# ---- build the work list ----------------------------------------------------
if [ "$#" -gt 0 ]; then
    RUNS=("$@")
else
    RUNS=()
    while IFS= read -r d; do
        RUNS+=("$(basename "$d")")
    done < <(find "$KCD_RUNS_DPATH" -mindepth 4 -maxdepth 4 -name best_stg2.pth \
                 -printf '%h\n' 2>/dev/null | sed 's#/runs/[^/]*$##' | sort -u)
fi

echo "=================================================================="
echo "rescore plan  (image: ${KCD_IMAGE:-kwcoco-detector-kit:ogdino-auto}  device: ${KCD_EVAL_DEVICE:-cuda})"
echo "=================================================================="
declare -a TODO
for run in "${RUNS[@]}"; do
    scheme="$(scheme_of "$run")"
    if [ -z "$scheme" ]; then
        printf '  SKIP  %-60s (unknown scheme)\n' "$run"; continue
    fi
    root="$KCD_RUNS_DPATH/$run"
    w="$root/tiled_compare/eval/wholeimage/eval/detect_metrics.json"
    t="$root/tiled_compare/eval/tiled/eval/detect_metrics.json"
    if [ -z "${KCD_FORCE_RESCORE:-}" ] && [ -f "$w" ] && [ -f "$t" ]; then
        printf '  done  %-60s (whole+tiled present)\n' "$run"; continue
    fi
    printf '  RUN   %-60s [%s]\n' "$run" "$scheme"
    TODO+=("$run")
done
echo "------------------------------------------------------------------"
echo "${#TODO[@]} model(s) to score; $(( ${#RUNS[@]} - ${#TODO[@]} )) skipped."
echo

# ---- score ------------------------------------------------------------------
fails=()
for run in "${TODO[@]}"; do
    scheme="$(scheme_of "$run")"
    IFS='|' read -r CATS DISTRACTORS <<< "$(scheme_args "$scheme")"
    echo "################################################################"
    echo "# $run  [$scheme]  cats=$CATS  distractors=${DISTRACTORS:-<none>}"
    echo "################################################################"
    if bash "$SCRIPT_DIR/rescore_tiled_compare.sh" "$run" "$CATS" "$DISTRACTORS"; then
        echo "[rescore_all] OK: $run"
    else
        echo "[rescore_all] FAILED: $run (continuing)"
        fails+=("$run")
    fi
    echo
done

# ---- summary table ----------------------------------------------------------
echo "=================================================================="
echo "COMPARISON TABLE"
echo "=================================================================="
python3 "$SCRIPT_DIR/rescore_collect_table.py" \
    --runs_dpath "$KCD_RUNS_DPATH" \
    --tsv "$KCD_RUNS_DPATH/../rescore_comparison.tsv" || true

if [ "${#fails[@]}" -gt 0 ]; then
    echo
    echo "WARNING: ${#fails[@]} run(s) failed to score:"
    printf '  - %s\n' "${fails[@]}"
    exit 1
fi
