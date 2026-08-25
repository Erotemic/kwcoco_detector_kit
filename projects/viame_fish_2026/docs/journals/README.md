# Project Journals — viame_fish_2026

Dated, narrative entries capturing what we tried, what we learned, and
what we'd do differently. Same convention as
[viame_sealions_2026](../../../viame_sealions_2026/docs/journals/README.md),
scoped to the FishTrack23 detector work.

## When to add an entry

- After a meaningful training cycle finishes (success or failure).
- After a multi-day debug session that ended with a fix worth remembering.
- After landing a refactor or new sub-system that changes how the pipeline
  behaves.
- After a hung or abandoned run, recording the stage it died at and the last
  log line.

If a future agent or collaborator could reasonably ask "why did we end up doing
it this way?", and the answer isn't in code comments, it belongs here.

## What goes in an entry

Each file is named `YYYY-MM-DD_<slug>.md`. Suggested sections, in order:
context, what we did, what worked / what broke, root cause, fixes, lessons.
Keep code references precise (`file.py:line` plus commit SHAs).

## Existing entries

- [2026-08-14 — orientation: what it would take to train the best fish
  detector](2026-08-14_orientation.md)
- [2026-08-14 — aiq: RF-DETR baseline audit + DEIMv2 prep
  runbook](2026-08-14_aiq_baseline_audit_and_deim_prep.md)
- [2026-08-17 — DEIM gen001: 13 good epochs, then a DDP
  deadlock](2026-08-17_deim_gen001_deadlock_at_epoch13.md)
- [2026-08-23 — the stop_epoch collision, and the first completed
  schedule](2026-08-23_stop_epoch_collision_and_gen003.md)
- [2026-08-25 — data audit, and the state of the RF-DETR
  comparison](2026-08-25_data_audit_and_the_rfdetr_comparison.md)
