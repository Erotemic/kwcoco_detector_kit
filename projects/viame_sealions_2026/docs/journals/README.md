# Project Journals — viame_sealions_2026

Dated, narrative entries capturing what we tried, what we learned,
and what we'd do differently. Mirrors the convention of
`kwcoco_detector_kit/dev/journals/` and `shitspotter/dev/journals/`,
but scoped to the sealion project — pipeline changes specific to the
sealion data, training runs (successful and failed), bug post-mortems,
operational gotchas.

## When to add an entry

- After a meaningful training cycle finishes (success or failure).
- After a multi-day debug session that ended with a fix worth
  remembering.
- After landing a refactor or new sub-system that changes how the
  pipeline behaves.
- After a `scancel`-then-resubmit cycle that took non-trivial time.

If a future agent or collaborator could reasonably ask "why did we
end up doing it this way?", and the answer isn't in code comments,
it belongs here.

## What goes in an entry

Each file is named `YYYY-MM-DD_<slug>.md`. Suggested sections, in
order:

1. **Context** — what kicked this off, what was the goal.
2. **What we did** — pipeline changes, runs submitted, datasets
   touched.
3. **What worked / what broke** — direct observations + diagnosis.
4. **Root cause** when applicable — *why* something broke, not just
   what symptom showed.
5. **Fix(es)** — links to commits + the line of reasoning behind
   them.
6. **Lessons** — durable rules / "next time don't do X" / "the cheap
   way to catch this is Y."

Keep code references precise: `file.py:line` plus commit SHAs so the
entry stays useful as the codebase evolves. Don't repeat what's
already in the code or research_plan — link to it.

## Existing entries

- [2026-05-26 — first 1-GPU baseline cycle: 48h spent training empty
  targets](2026-05-26_first_baseline_attempt.md)
- [2026-05-26 — passthrough whitelist wasn't enough: tile writer
  needed to stamp source_category](2026-05-26_passthrough_was_not_enough.md)
- [2026-05-29 — NFS must count as a negative; pup is the binding
  constraint](2026-05-29_nfs_must_be_negative_and_first_per_class_ap.md)
- [2026-05-29 — per-checkpoint vali rescoring: in-train selection
  agrees, last.pth ≡ epoch-0.pth](2026-05-29_per_checkpoint_rescoring_results.md)
