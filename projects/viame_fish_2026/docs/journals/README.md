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
