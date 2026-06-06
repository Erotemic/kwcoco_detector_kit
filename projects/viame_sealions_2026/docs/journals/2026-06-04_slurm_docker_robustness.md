# 2026-06-04 — slurm + docker robustness lessons

## Context

The gen004 submission cycle ([[2026-06-04_gen004_forensic_and_resume]])
revealed several non-obvious failure modes in the slurm + docker
launch path. None of them were our pipeline; all were environmental
gotchas that future agents need to know about so they don't
re-discover the same set.

## 1. Docker `--gpus` value parser is broken for multi-GPU on arisia

**Symptom**: every form of `--gpus device=0,1` produced one of:

```
docker: Error response from daemon: cannot set both Count and
  DeviceIDs on device request

invalid argument "device=GPU-uuid1,GPU-uuid2" for "--gpus" flag:
  invalid count (GPU-uuid2): value must be either "all" or an integer
```

**Root cause**: docker's value parser splits the `device=...` value on
commas and tries each piece. A numeric piece looks like a "count"
spec; a UUID piece is invalid syntax. We tried:

* `--gpus=device=0,1` (equals form) → reads "1" as Count → error
* `--gpus device=0,1` (space form) → same parser, same bug (job 2574)
* `--gpus device=GPU-uuid1,GPU-uuid2` → reads UUID2 as count → "invalid syntax"

All three route through the same broken value parser.

**Fix** (commit `c9183c5`): bypass `--gpus` entirely. The pre-19.03
nvidia-docker2 path uses `--runtime=nvidia` plus
`NVIDIA_VISIBLE_DEVICES`. The nvidia runtime hook reads the env var
before container start and exposes exactly the listed GPUs.
`NVIDIA_VISIBLE_DEVICES` accepts comma-separated UUIDs or indices and
does NOT go through the broken `--gpus` parser.

```bash
GPU_UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader \
              -i "$CUDA_VISIBLE_DEVICES" | tr '\n' ',' | sed 's/,$//')
GPU_FLAG=(--runtime=nvidia \
          -e "NVIDIA_VISIBLE_DEVICES=${GPU_UUIDS}" \
          -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility")
```

Single-GPU jobs work fine through `--gpus device=0` (no comma, no
parser ambiguity). The fix unifies both paths through
`--runtime=nvidia`.

## 2. Slurm SIGKILL beats `docker stop -t 30` → orphaned containers

**Symptom**: after gen003 single_sealion 2565 and gen004
hgnetv2_n 2570 walltime-hit, `docker ps` showed both containers
still running 2 days later with no associated slurm job. Both
were sitting on GPUs with ~3-4 GB VRAM held.

**Root cause**: slurm sends `SIGTERM` to the wrapper bash, waits
`KillWait=30s` (default), then sends `SIGKILL`. Our trap ran
`docker stop -t 30` — also 30s. The bash wrapper gets SIGKILL'd
before docker stop returns. The `docker run` client process dies
but the **container in the daemon keeps running** — `--rm` only
triggers when the CONTAINER exits, not when the client exits.

**Fix** (commit `3d3cda7`): swap `docker stop -t 30` for
`docker kill` in the trap (immediate SIGKILL to the container).
Saves up to 30s in the cleanup path. The container is being
terminated either way; giving it time to flush is nice but not
worth losing the race.

The same commit also added `kit_zombie_janitor.sh` as a
report-only sweep for any zombies that still slip through —
e.g. SSH disconnect during scancel, or future failure modes
we haven't anticipated. Killing zombies is opt-in via
`KCD_KILL_ZOMBIES=1` (commit `2ed13b3`): if the janitor finds
something, that's a signal worth investigating before reflexively
nuking.

## 3. Trap defense layers in `_sbatch_train.sh`

Current scheme (committed across `4206970`, `3d3cda7`, `2ed13b3`):

```
docker run with:
  --name kcd-<jobid>-<run>              ← deterministic cleanup target
  --cidfile /tmp/kcd-cid.<rand>         ← survives if wrapper PID dies
  --label kcd.run_name=...              ← forensics
  --label kcd.slurm_job_id=...
  --label kcd.user=...
  --label kcd.created_at=...
  --runtime=nvidia                       ← per (1)
  -e NVIDIA_VISIBLE_DEVICES=<UUIDs>

trap _kcd_cleanup EXIT INT TERM HUP
  → trap - EXIT INT TERM HUP            ← prevent recursion
  → docker kill <cid>                    ← immediate; no 30s wait
  → docker rm -f <cid>                   ← belt + suspenders
  → docker kill <container_name>         ← if cidfile race
  → docker rm -f <container_name>
  → GPU leak report                      ← always-on diagnostic
     (KCD_KILL_GPU_LEAKS=1 to escalate to SIGKILL on same-user PIDs)
  → exit $exit_code
```

What this catches:
* Normal exit (training completed) → EXIT fires, container
  already self-removed by `--rm`, cleanup is no-op
* Bash error (mid-trap, set -e) → EXIT fires
* Slurm scancel (SIGTERM grace) → TERM fires, ~immediate kill
* User Ctrl-C → INT fires
* SSH disconnect → HUP fires

What this still misses:
* slurm `scancel --signal=SIGKILL` (skips grace period)
* OOM-killer on the wrapper bash itself
* Host crash
For those, the janitor is the catch-all.

## 4. Dev mount vs image rebuild — when to use which

The image bakes kit code via `pip install -e` from
`/opt/kwcoco_detector_kit/` at build time. The host's
`$KCD_REPO_ROOT` is bind-mounted at the same host path inside
the container BUT that's not where the install lives.

So for any kit Python change to take effect at runtime:

| Path | Method | When to use |
|---|---|---|
| `kwcoco_detector_kit/*.py` (kit Python) | `KCD_DEV_MOUNT_KIT=1` OR rebuild | Most iteration |
| `tpl/DEIMv2/*.py` (submodule) | `KCD_DEV_MOUNT_DEIMV2=1` OR rebuild | DEIMv2 changes |
| `tpl/kwcoco_dataloader/*.py` | `KCD_DEV_MOUNT_DATALOADER=1` OR rebuild | reader changes |
| `projects/.../scripts/*.sh` | always live via `$KCD_REPO_ROOT` mount | no action needed |
| `_sbatch_train.sh` | host-side, never enters container | no action needed |

**Workflow:**
* Quick iteration / experimental fixes → use the dev mounts.
* Stable run that should be reproducible → rebuild the image
  via `docker/opengroundingdino/build_arisia_cuda132.sh` (the
  build script tests pytest inside the image so regressions
  don't ship).
* Long training runs → strongly prefer rebuild. Live code can
  shift under the container if the host's git state changes;
  rebuild guarantees a frozen artifact.

The user explicitly asked for the rebuild workflow on
2026-06-03 to avoid "live code changing from under us mid-train"
— that preference is now in [[feedback-image-is-reproducibility-unit]].

## 5. The "stale image" failure mode (job 2566)

When the docker image is OLDER than a kit Python change you
need, the container will simply not have the new module. We
saw this once:

```
job 2566: ModuleNotFoundError: No module named
kwcoco_detector_kit.data.balance_mscoco
```

`balance_mscoco.py` was on the host but the image hadn't been
rebuilt since before that file landed.

Diagnostic command (on arisia):
```bash
docker run --rm kwcoco-detector-kit:ogdino-cu132-arisia \
  python -c "from kwcoco_detector_kit.data.balance_mscoco \
  import BalanceMSCOCOConfig; print('OK')"
```

If that fails with ImportError, you need either `KCD_DEV_MOUNT_KIT=1`
or a rebuild. If it fails with anything else, the kit Python is
broken — not a stale-image issue.

## 6. Resource budgets per arisia (shared node)

Per [[feedback-arisia-resource-budgets]] (revised this session):

| model + scale | KCD_CPUS_PER_TASK | KCD_MEM | KCD_TRAIN_NUM_WORKERS |
|---|---|---|---|
| hgnetv2_n @ 320, 1 GPU | 2 | 12 GB | 2 |
| dinov3_s @ 640, 2 GPU | 4 | 24-32 GB | 2 |
| 4-GPU runs | scale linearly | scale linearly | scale linearly |

These are right-sized for actual observed peaks (typically half
to a third of the kit's pre-revision defaults). The motivation
is arisia is a shared 12-CPU / 126 GB / 4-GPU node; oversized
reservations pend other users on (Resources) without actually
using the headroom.

Override via env at submit time if a profile shows real pressure:
```bash
KCD_MEM=48G bash submit_train_...
```

## Open

* arisia's docker version: should we upgrade so `--gpus
  device=0,1` works natively? Probably yes, but it requires
  cluster admin time. The `--runtime=nvidia` workaround is
  stable and adds no overhead, so this isn't urgent.

* Job-level cleanup hook: slurm has Prolog/Epilog scripts that
  fire at job start/end at the cluster level. A cluster-wide
  janitor in Epilog would catch every zombie regardless of our
  in-trap cleanup. Requires cluster admin coordination.

* Container TTL based on slurm walltime: pass the walltime to
  the container and have an internal watchdog self-terminate at
  walltime - grace. Slightly redundant with slurm's own
  enforcement but defensive.
