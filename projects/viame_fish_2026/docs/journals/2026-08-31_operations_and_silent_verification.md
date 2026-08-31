# 2026-08-31 — running jobs on aiq-gpu, and five ways a check said "fine"

Companion to [the falsification entry](2026-08-31_tiling_hypothesis_falsified.md).
That one records what the experiments showed; this one records how to run
anything here, and the failure mode that cost the most time.

## Part 1 — this host has no slurm

Every `submit_train_*.sh` in this project called `sbatch` unconditionally.
There is no `sbatch` binary on aiq-gpu and `slurmd`/`slurmctld` are inactive,
so every wrapper would pass its preflight checks, print its banner, and die on
`sbatch: command not found`. gen007 set `KCD_NO_SLURM=1` and nothing read it.

The sea-lion project's `_submit_train.sh` has had a no-slurm branch since
yardrat/namek; the fish one never got it. Fixed in `6e69a98`:

- fish `_submit_train.sh` execs the shared `_run_standalone.sh` when
  `KCD_NO_SLURM=1`, and raises an explicit error when `sbatch` is absent and it
  is not — so the next host without slurm gets a diagnosis, not a shell error.
- `_run_standalone.sh` sourced **its own directory's** `paths.sh`, which would
  have handed a fish run the sea-lion bundles. It now honours `KCD_REPO_ROOT`
  the way `_sbatch_train.sh` always has. Every other path in it — the mounts,
  the workdir, the launch script — already resolved that way; this was the one
  line that did not.
- `KCD_DOCKER_CMD` lets only the `docker run` elevate. The `docker` group is
  empty here, and running the whole submit under `sudo` would resolve every
  `VF_*`/`KCD_*` default against root's home instead of the user's.

### Host facts worth not rediscovering

- **Docker needs `sudo`** (empty `docker` group), and Docker 29 ships **no
  builder** — `docker-buildx` had to be installed. The Dockerfile is
  `# syntax=docker/dockerfile:1.7` with `RUN --mount=type=cache` throughout, so
  the legacy builder is not a substitute.
- `$HOME/ssd-data` and `/mnt/aivm-persistent/hostcode-ssd-data-023888cb` are
  the **same device and inode**. No `VF_SSD_ROOT` override is needed.
- Training runs in the **foreground** under `docker run`. There is no job id and
  no `squeue`; a tmux pane owns the job and detaching is what backgrounds it.
- The GPUs are **not visible from every session on this box**. Agent sessions
  saw no `/dev/nvidia*` and `--gpus all` failed, while training ran fine from
  the user's. Do not conclude "no GPUs" from one shell.

## Part 2 — the end-to-end drivers

`run_gen007_end_to_end.sh` and `run_test_scores_end_to_end.sh` exist so a long
job is one command from a tmux pane. Both share a shape worth reusing:

- **Preflight fails in seconds, not 40 minutes into a build.** Bundles present
  *and different files*, pretrained checkpoint, tile size resolved from cache
  metadata, disk headroom, and GPUs reachable **through docker** — which is the
  question that matters and is not the same as whether the host has them.
- **Phases are independently skippable** (`KCD_SKIP_*`) so one failure does not
  force redoing the others, and each tees to its own log with the *phase's*
  exit status propagated rather than `tee`'s.
- **Image verification always runs, even when the build is skipped.** It
  compares the image's baked submodule shas against local HEAD and greps the
  baked solver for the line that makes `KCD_BALANCE_REPLACEMENT=False` mean
  anything. A stale image is the failure that looks like success.
- **Training is deliberately not auto-resumed.** gen007 pins
  `KCD_RESUME_CKPT=fresh`; resuming into a half-finished schedule would blend
  two configurations.

`score_epochs.py` now **merges** into an existing vali summary instead of
overwriting it. Scoring gen006 alone would otherwise have dropped the
gen001/gen003/gen007 rows that had just cost five hours, and the next reader
would have concluded those runs were never measured. Rows for runs scored in
the current pass replace their prior entries; the rest carry forward. Safe
because the filename pins window, overlap and stride, so a merge can only join
same-protocol rows.

`score_test_once.py` refuses to score a run that has staged epochs and no vali
entry. Picking a checkpoint there would be choosing on test, which is the one
thing that would make the numbers meaningless.

## Part 3 — five checks that said "fine" while checking nothing

This is the part worth carrying forward. Every one of these was green, and
every one was green for a reason unrelated to the code being correct.

| # | the check | why it was green |
|---|---|---|
| 1 | `dev/check_undefined_names.py` | `Path(root).rglob("*.py")` yields **nothing** for a file argument. Every explicit-path invocation checked **zero files** and printed "0 finding(s)". It missed a `SyntaxError` pytest then caught. |
| 2 | `kcd_sample_replacement` | Emitted into the config, parsed by the CLI, recorded in the sidecar, announced in the banner — and never **read** by the solver. Every test passed: the factory worked, the flag parsed, the key was present. Nothing checked that anything read it. |
| 3 | the same flag again | The fix was in the working tree but **uncommitted inside the submodule**, so the tracked pointer lacked it. `git add -A` in the parent cannot advance a pointer for a dirty submodule with no commit. The image bakes the pointer, so a built image would have carried the bug regardless of the checkout. |
| 4 | `test_kwcoco_sampler.py` | 8 tests **skipped** locally because `kwcoco_dataloader` is absent from a plain checkout. They only execute inside the image, where they failed on kwcoco 0.9. The suite was green because the subject was missing. |
| 5 | `test_noreplacement_sampler.py` | 14 tests silently **skipped** for want of `torch`, so the session's core new mechanism was unverified while the suite reported clean. |

And a sixth, different in kind: gen007 shipped without the block that points
`KCD_TRAIN_KWCOCO` at the **tiled** bundle. In a clean shell it aborted on its
own `KCD_TILE_SOURCE_KWCOCO == KCD_TRAIN_KWCOCO` guard — fail-closed and
correct — but that guard exists to catch a *misconfiguration* and was catching
an *omission*. With a stale variable exported it would have trained 13 h on
whole frames while the sampler weights were computed against tiles.

### The rule these suggest

**A passing check is evidence only if you know it ran.** Before trusting one,
confirm it executed: count the tests that ran rather than the ones that failed,
make skips loud when they cover the thing under test, and prefer a check that
can be shown to fail on the broken state. Every fix above was verified by
reverting the defect and watching the test go red — that step is what turned
these from assertions into evidence.

Two structural guards now exist for the class:

- `test_solver_consumes_sample_config.py` requires every `kcd_sample_*` key the
  trainer emits to be **read** by `_solver.py`, and `sampler_from_weights_file`
  to be called with `replacement=`. It generalises to the next write-only key.
- `test_launcher_pins_tiled_splits.py` requires any launcher enabling
  `KCD_BALANCE_SEQUENCE` to pin both splits from their `KCD_TILE_*`
  counterparts, before anything reads them.

## Commits

`ef750d2` checkpoint pinning · `fd96e32` sequence balancing · `cf0705c` alpha
metric + tail + no-replacement · `cdd7dd6`/`dcf8cec` solver flag + submodule
pointer · `13b9fe8` epoch shuffle + off-by-one · `bd76c62` kwcoco 0.9 migration
· `a7ef134` tiled split pin · `6e69a98` no-slurm path · `936fd07` gen007 driver
· `719382f` test scoring · `d9a8a64` falsification record
