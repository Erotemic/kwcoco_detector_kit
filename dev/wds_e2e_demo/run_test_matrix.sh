#!/usr/bin/env bash
# Run dev/wds_e2e_demo/run_demo.sh across the compute × data-path matrix
# and produce a concise summary so we can verify the kit works in every
# deploy target we care about.
#
# Scenarios (auto-skipped if the runtime isn't available):
#
#   host-cpu-jpeg          host venv, CPU,        kwcoco JPEG path
#   host-cpu-wds           host venv, CPU,        WebDataset path
#   host-gpu1-jpeg         host venv, 1 GPU,      kwcoco JPEG
#   host-gpu1-wds          host venv, 1 GPU,      WebDataset
#   host-gpu2-jpeg         host venv, 2 GPUs,     kwcoco JPEG
#   host-gpu2-wds          host venv, 2 GPUs,     WebDataset
#   docker-gpu1-jpeg       docker,    1 GPU,      kwcoco JPEG
#   docker-gpu1-wds        docker,    1 GPU,      WebDataset
#   docker-gpu2-jpeg       docker,    2 GPUs,     kwcoco JPEG
#   docker-gpu2-wds        docker,    2 GPUs,     WebDataset
#   slurm-host-gpu1-wds    sbatch -> host venv,   WebDataset
#   slurm-docker-gpu1-wds  sbatch -> docker,      WebDataset
#
# Each scenario produces:
#   <out>/<name>/log.txt        full stdout/stderr
#   <out>/<name>/summary.json   {status, duration_s, train_total_time_s,
#                                ap, bench_ms, error?}
#
# After all scenarios run we print a TSV table and write it to
# <out>/matrix.tsv.
#
# Usage:
#   bash dev/wds_e2e_demo/run_test_matrix.sh
#
# Common overrides:
#   MATRIX_OUT=/path/to/dir              # default /tmp/wds_e2e_matrix
#   MATRIX_ONLY="host-cpu-wds,..."       # comma list to filter
#   MATRIX_SKIP="docker-gpu2-jpeg,..."   # comma list to skip
#   MATRIX_EPOCHS=2                      # per-scenario epochs (default 2)
#   MATRIX_DOCKER_IMAGE=kwcoco-detector-kit:ogdino-auto
#   MATRIX_SLURM_PARTITION=<name>        # required for slurm scenarios
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MATRIX_OUT="${MATRIX_OUT:-/tmp/wds_e2e_matrix}"
MATRIX_ONLY="${MATRIX_ONLY:-}"
MATRIX_SKIP="${MATRIX_SKIP:-}"
MATRIX_EPOCHS="${MATRIX_EPOCHS:-2}"
MATRIX_DOCKER_IMAGE="${MATRIX_DOCKER_IMAGE:-kwcoco-detector-kit:ogdino-auto}"
MATRIX_SLURM_PARTITION="${MATRIX_SLURM_PARTITION:-}"
# Per-scenario wall-clock cap. Heterogeneous DDP (e.g. yardrat's
# RTX 8000 + RTX 5000) can hang in NCCL init_process_group; without
# a timeout the runner blocks indefinitely. 600s is comfortably
# enough for a clean GPU/docker run of the demo (~30s on a 3090).
MATRIX_SCENARIO_TIMEOUT="${MATRIX_SCENARIO_TIMEOUT:-600}"
# Heartbeat log every N seconds so a "hung" scenario is visible
# (otherwise stdout is silent between print_freq=500 iter prints).
MATRIX_HEARTBEAT_INTERVAL="${MATRIX_HEARTBEAT_INTERVAL:-30}"
PYTHON_BIN="${PYTHON_BIN:-}"

rm -rf "$MATRIX_OUT"
mkdir -p "$MATRIX_OUT"

# ----- capability detection -------------------------------------------
HAVE_DOCKER=0
HAVE_SLURM=0
HAVE_NVIDIA=0
N_GPUS=0
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    HAVE_DOCKER=1
fi
if command -v sbatch >/dev/null 2>&1; then
    HAVE_SLURM=1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    HAVE_NVIDIA=1
    N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
fi
# Does the docker image exist locally?
HAVE_DOCKER_IMAGE=0
if [ "$HAVE_DOCKER" = "1" ] && docker image inspect "$MATRIX_DOCKER_IMAGE" >/dev/null 2>&1; then
    HAVE_DOCKER_IMAGE=1
fi

echo "=== test matrix for $REPO_ROOT ==="
echo "  output:   $MATRIX_OUT"
echo "  host:     $(hostname)"
echo "  GPUs:     $N_GPUS detected"
[ "$N_GPUS" -gt 0 ] && nvidia-smi --query-gpu=index,name --format=csv,noheader | sed 's/^/    /'
echo "  docker:   $([ "$HAVE_DOCKER" = "1" ] && echo "yes" || echo "no")"
echo "  image:    $MATRIX_DOCKER_IMAGE $([ "$HAVE_DOCKER_IMAGE" = "1" ] && echo "(present)" || echo "(MISSING)")"
echo "  slurm:    $([ "$HAVE_SLURM" = "1" ] && echo "yes" || echo "no")"
echo "  epochs:   $MATRIX_EPOCHS"
echo

# ----- scenario table -------------------------------------------------
# Naming convention: `gpu<N>x` means "N GPUs total" (a count), not "GPU
# index N". Earlier `gpu1` / `gpu2` was confusable with CUDA device IDs.
#
# name|compute|container|data_path|gpus|requires
declare -a SCENARIOS=(
    "host-cpu-jpeg|host|none|jpeg|0|"
    "host-cpu-wds|host|none|wds|0|"
    "host-gpu1x-jpeg|host|none|jpeg|1|cuda"
    "host-gpu1x-wds|host|none|wds|1|cuda"
    "host-gpu2x-jpeg|host|none|jpeg|2|cuda2"
    "host-gpu2x-wds|host|none|wds|2|cuda2"
    "docker-gpu1x-jpeg|host|docker|jpeg|1|cuda,docker"
    "docker-gpu1x-wds|host|docker|wds|1|cuda,docker"
    "docker-gpu2x-jpeg|host|docker|jpeg|2|cuda2,docker"
    "docker-gpu2x-wds|host|docker|wds|2|cuda2,docker"
    "slurm-host-gpu1x-wds|slurm|none|wds|1|slurm"
    "slurm-docker-gpu1x-wds|slurm|docker|wds|1|slurm,docker"
)

# ----- runner ---------------------------------------------------------
should_run() {
    local name="$1" reqs="$2"
    if [ -n "$MATRIX_ONLY" ] && ! echo ",$MATRIX_ONLY," | grep -q ",$name,"; then
        echo "SKIP_FILTER"; return
    fi
    if [ -n "$MATRIX_SKIP" ] && echo ",$MATRIX_SKIP," | grep -q ",$name,"; then
        echo "SKIP_EXPLICIT"; return
    fi
    for req in ${reqs//,/ }; do
        case "$req" in
            cuda)   [ "$HAVE_NVIDIA" = "1" ] || { echo "SKIP_NO_CUDA"; return; } ;;
            cuda2)  [ "$N_GPUS" -ge 2 ]      || { echo "SKIP_NEED_2GPU"; return; } ;;
            docker) [ "$HAVE_DOCKER_IMAGE" = "1" ] || { echo "SKIP_NO_DOCKER_IMAGE"; return; } ;;
            slurm)  [ "$HAVE_SLURM" = "1" ]  || { echo "SKIP_NO_SLURM"; return; } ;;
            "")     ;;
        esac
    done
    echo "RUN"
}

run_scenario() {
    local name="$1" compute="$2" container="$3" data_path="$4" gpus="$5"

    local scen_dir="$MATRIX_OUT/$name"
    mkdir -p "$scen_dir"
    local log="$scen_dir/log.txt"
    local summary="$scen_dir/summary.json"
    local demo_out="$scen_dir/demo"

    echo
    echo "--- running $name ---"
    local t_start
    t_start=$("$PYTHON_BIN" -c 'import time; print(time.monotonic())' 2>/dev/null \
              || python3 -c 'import time; print(time.monotonic())')

    # gpus=0 means "CPU only" — torch.distributed.run still needs
    # --nproc_per_node >= 1 (one driver process), so map 0→1 and rely
    # on the absence of CUDA to keep tensors on CPU.
    local nproc="$gpus"
    [ "$nproc" -lt 1 ] && nproc=1

    local env_prefix=(
        env
        "DEMO_OUT=$demo_out"
        "DEMO_DATA_PATH=$data_path"
        "DEMO_EPOCHS=$MATRIX_EPOCHS"
        "DEMO_NUM_GPUS=$nproc"
    )
    # For multi-GPU scenarios, enable NCCL + torch.distributed verbose
    # logging into the scenario log so we can see WHERE a hang happens
    # (rendezvous, topology negotiation, first all_reduce, etc.).
    # Also pin DEMO_WDS_EPOCH_LENGTH so every rank iterates the same
    # number of samples per epoch — without it, uneven shard splits
    # cause DDP collective mismatches between ranks (host-gpu2x-wds
    # symptom 2026-05-30: "Rank 0 BROADCAST vs Rank 1 REDUCE"). 16
    # matches the demo corpus size; smaller values are fine, the
    # adapter cycles to fill the count.
    if [ "$gpus" -gt 1 ]; then
        env_prefix+=(
            "NCCL_DEBUG=INFO"
            "TORCH_DISTRIBUTED_DEBUG=DETAIL"
            "TORCH_NCCL_BLOCKING_WAIT=1"
            "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
            "DEMO_WDS_EPOCH_LENGTH=${DEMO_WDS_EPOCH_LENGTH:-16}"
        )
    fi
    # For multi-GPU on the host with potentially heterogeneous cards
    # (yardrat: RTX 8000 + RTX 5000), let NCCL try anyway — the failure
    # mode tells us whether DDP can handle the heterogeneity, which is
    # itself useful info.
    [ -n "$PYTHON_BIN" ] && env_prefix+=("PYTHON_BIN=$PYTHON_BIN")

    local status="ok"
    local cmd=()
    case "$container" in
        none)
            cmd=("${env_prefix[@]}" bash dev/wds_e2e_demo/run_demo.sh)
            ;;
        docker)
            cmd=(
                docker run --rm
                --gpus all
                --ipc=host --shm-size=16g
                -v "$REPO_ROOT:$REPO_ROOT" -w "$REPO_ROOT"
                -e "DEMO_OUT=$demo_out"
                -e "DEMO_DATA_PATH=$data_path"
                -e "DEMO_EPOCHS=$MATRIX_EPOCHS"
                -e "DEMO_NUM_GPUS=$nproc"
                -e "PYTHON_BIN=/opt/venv/bin/python"
                "$MATRIX_DOCKER_IMAGE"
                bash dev/wds_e2e_demo/run_demo.sh
            )
            ;;
    esac
    case "$compute" in
        slurm)
            # Run via sbatch and wait. The slurm scenarios need a
            # partition; if MATRIX_SLURM_PARTITION isn't set, fail.
            if [ -z "$MATRIX_SLURM_PARTITION" ]; then
                echo "  SKIP: slurm scenarios require MATRIX_SLURM_PARTITION" >&2
                status="skip_no_partition"
                cmd=(true)
            else
                local sbatch_script="$scen_dir/sbatch.sh"
                {
                    echo '#!/usr/bin/env bash'
                    echo "set -e"
                    echo "${cmd[@]}"
                } > "$sbatch_script"
                chmod +x "$sbatch_script"
                cmd=(
                    sbatch
                    --partition="$MATRIX_SLURM_PARTITION"
                    --gres="gpu:$gpus"
                    --wait
                    --output="$scen_dir/slurm.out"
                    "$sbatch_script"
                )
            fi
            ;;
    esac

    # Run with a wall-clock timeout + a heartbeat loop that prints the
    # last log line every $MATRIX_HEARTBEAT_INTERVAL seconds so a hung
    # scenario is at least visible.
    set +e
    timeout --kill-after=30 --signal=TERM "$MATRIX_SCENARIO_TIMEOUT" \
        "${cmd[@]}" >"$log" 2>&1 &
    local cmd_pid=$!
    while kill -0 "$cmd_pid" 2>/dev/null; do
        sleep "$MATRIX_HEARTBEAT_INTERVAL"
        kill -0 "$cmd_pid" 2>/dev/null || break
        local last
        last=$(tail -n 1 "$log" 2>/dev/null | tr -d '\r' | cut -c1-100)
        echo "    [heartbeat $(date +%H:%M:%S) pid=$cmd_pid] ${last:-(no output yet)}"
    done
    wait "$cmd_pid"
    local rc=$?
    set -e
    if [ "$rc" = "124" ] || [ "$rc" = "137" ]; then
        status="timeout"
        # On timeout, dump py-spy stacks of any surviving python
        # processes (best-effort — needs --privileged or root for cross-
        # process). The dumps land alongside the log so debugging a
        # hang doesn't require re-running the scenario.
        if command -v py-spy >/dev/null 2>&1; then
            local pyspy_log="$scen_dir/py-spy.txt"
            : > "$pyspy_log"
            for pid in $(pgrep -f "tpl/DEIMv2/train.py" 2>/dev/null); do
                echo "=== py-spy dump pid=$pid ===" >> "$pyspy_log"
                py-spy dump --pid "$pid" >> "$pyspy_log" 2>&1 || true
                echo >> "$pyspy_log"
            done
            for pid in $(pgrep -f "torch.distributed.run" 2>/dev/null); do
                echo "=== py-spy dump pid=$pid (launcher) ===" >> "$pyspy_log"
                py-spy dump --pid "$pid" >> "$pyspy_log" 2>&1 || true
                echo >> "$pyspy_log"
            done
            echo "    [timeout] dumped py-spy stacks -> $pyspy_log"
        fi
    fi

    local t_end
    t_end=$("$PYTHON_BIN" -c 'import time; print(time.monotonic())' 2>/dev/null \
            || python3 -c 'import time; print(time.monotonic())')
    local duration_s
    duration_s=$(awk -v a="$t_end" -v b="$t_start" 'BEGIN{printf "%.2f", a-b}')

    if [ "$status" = "ok" ] && [ "$rc" -ne 0 ]; then
        status="failed"
    fi

    # Extract metrics from the log.
    local ap="" bench_ms="" train_time=""
    if [ -f "$log" ]; then
        ap=$(grep -oE 'test AP[[:space:]]+[0-9.]+' "$log" | tail -1 | awk '{print $NF}')
        bench_ms=$(grep -oE 'mean=[0-9.]+ms' "$log" | tail -1 | sed 's/mean=//;s/ms//')
        train_time=$(grep -oE 'Training time [0-9:]+' "$log" | tail -1 | awk '{print $NF}')
    fi

    # Capture first error line if not ok.
    local error=""
    if [ "$status" != "ok" ] && [ -f "$log" ]; then
        error=$(grep -oE '^[A-Za-z][^:]+Error:.*|^FAILED.*|^fatal: .*' "$log" | head -1 | tr -d '\n' | cut -c1-200)
    fi

    python3 - <<PYEOF
import json
data = {
    "name": "$name",
    "compute": "$compute",
    "container": "$container",
    "data_path": "$data_path",
    "gpus": $gpus,
    "status": "$status",
    "rc": $rc,
    "duration_s": float("$duration_s"),
    "ap": "$ap" or None,
    "bench_ms": float("$bench_ms") if "$bench_ms" else None,
    "train_time": "$train_time" or None,
    "error": "$error" or None,
    "log": "$log",
}
with open("$summary", "w") as f:
    json.dump(data, f, indent=2)
PYEOF

    # Result line. Visually distinct from the heartbeat noise so the
    # outcome is obvious even on a long-running matrix.
    local tag
    case "$status" in
        ok)      tag="[OK]      " ;;
        failed)  tag="[FAIL]    " ;;
        timeout) tag="[TIMEOUT] " ;;
        *)       tag="[$status]" ;;
    esac
    echo
    printf "    %s %-25s  duration=%-6s  ap=%-6s  bench=%-6s\n" \
        "$tag" "$name" "${duration_s}s" "${ap:-—}" "${bench_ms:-—}"
    if [ "$status" != "ok" ]; then
        # Surface the actual error so the user doesn't have to dig
        # into the log file to know what broke. Lead with the
        # first python-style traceback if present, else the last
        # 15 non-blank lines.
        echo "    --- error context (last 15 lines of $log) ---"
        tail -n 15 "$log" 2>/dev/null | sed 's/^/        /'
        echo "    --- full log: $log ---"
    fi
}

# Resolve PYTHON_BIN if user didn't set it; same auto-detection as the demo.
if [ -z "$PYTHON_BIN" ]; then
    for cand in "$REPO_ROOT/.venv/bin/python" /opt/venv/bin/python "$(command -v python3 || true)"; do
        if [ -n "$cand" ] && [ -x "$cand" ]; then
            PYTHON_BIN="$cand"
            break
        fi
    done
fi
export PYTHON_BIN

# ----- main loop ------------------------------------------------------
echo "=== scenarios ==="
declare -a RAN=() SKIPPED=()
for row in "${SCENARIOS[@]}"; do
    IFS='|' read -r name compute container data_path gpus reqs <<< "$row"
    decision=$(should_run "$name" "$reqs")
    if [ "$decision" = "RUN" ]; then
        run_scenario "$name" "$compute" "$container" "$data_path" "$gpus"
        RAN+=("$name")
    else
        printf "    %-25s  %s\n" "$name" "$decision"
        SKIPPED+=("$name:$decision")
    fi
done

# ----- summary table --------------------------------------------------
echo
echo "=== summary ==="
{
    printf "name\tstatus\tcompute\tcontainer\tdata_path\tgpus\tduration_s\ttrain_time\tap\tbench_ms\terror\n"
    for n in "${RAN[@]}"; do
        python3 - <<PYEOF
import json
with open("$MATRIX_OUT/$n/summary.json") as f:
    d = json.load(f)
print("\t".join(str(d.get(k) or "") for k in (
    "name","status","compute","container","data_path","gpus",
    "duration_s","train_time","ap","bench_ms","error"
)))
PYEOF
    done
} | tee "$MATRIX_OUT/matrix.tsv"

echo
echo "Artifacts:"
echo "  matrix.tsv:   $MATRIX_OUT/matrix.tsv"
echo "  per-scenario: $MATRIX_OUT/<name>/{log.txt,summary.json,demo/}"
echo
echo "Ran:     ${#RAN[@]} scenario(s)"
echo "Skipped: ${#SKIPPED[@]} scenario(s)"

# Exit non-zero if any ran scenario failed. List the failures explicitly
# at the end so the matrix verdict is impossible to miss.
declare -a FAILED=()
for n in "${RAN[@]}"; do
    s=$(python3 -c "import json; print(json.load(open('$MATRIX_OUT/$n/summary.json'))['status'])")
    [ "$s" != "ok" ] && FAILED+=("$n:$s")
done
echo
echo "============================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  MATRIX PASSED — ${#RAN[@]} scenarios ok, ${#SKIPPED[@]} skipped"
    echo "============================================================"
    exit 0
else
    echo "  MATRIX FAILED — ${#FAILED[@]} of ${#RAN[@]} scenarios failed"
    for entry in "${FAILED[@]}"; do
        echo "    - $entry"
    done
    echo "  Logs under: $MATRIX_OUT/<name>/log.txt"
    echo "============================================================"
    exit 1
fi
