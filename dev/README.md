# Developer notes — `dev/`

Long-running engineering memory for the `kwcoco_detector_kit` project. Anything that **isn't** the package itself (`kwcoco_detector_kit/`), examples (`examples/`), or end-user documentation (`docs/`) lives here. The contents are deliberately agent-readable: an agent arriving cold should be able to read this folder, understand the project's pattern of past mistakes, and take fewer of them.

The two subtrees today:

```text
dev/
  benchmark-candidates/   # Distilled hard questions from real engineering mistakes
  journals/               # >1hr-debug-time bug postmortems
```

`dev/` is **not** a TODO list, **not** a feature log, **not** a code documentation tree (use `docs/`), and **not** a place to dump WIP scratch.

The layout was ported from `/home/joncrall/code/ambition/dev/` and `/home/joncrall/code/shitspotter/dev/` — two prior projects where the same discipline produced agent-readable memory that materially shortened debugging cycles on later work. Read those original `dev/` trees if you want to see mature instances of the same pattern.

---

## `dev/benchmark-candidates/`

**What it is.** A growing corpus of self-contained kwcoco / pytorch / detector-training questions distilled from real maintenance mistakes made while building this kit. Each question captures a *pre-error* setup — the context an agent had at the moment of the mistake — so a different model facing the same setup can be tested for the same failure mode. The corpus is intended to be benchmark-track quality if the project ever ships one; for now its first job is making future `kwcoco_detector_kit` agents better.

**Why an agent should care.**

- **Read** these before tackling a refactor that resembles one catalogued here. The "Why this was easy to miss" section on each question is the single most useful one — it names the cognitive trap so you can recognise it in your own reasoning.
- **Write** here when you cause (or watch the user cause, or resolve) a mistake whose root cause is a transferable invariant. The bar is "another model in the same situation could plausibly make the same mistake without this question written down."

**Layout.**

```text
benchmark-candidates/
  README.md                          # Workflow, quality bar, prompt levels (Levels A/B/C)
  pipeline-bootstrap-questions.md    # Setup / env / install / YAML-gen invariants
  compositions.md                    # Multi-invariant questions that compose single-issue Qs
```

Topic-scoped files spin off only when parallel agents are touching the same section concurrently. When in doubt, add to `pipeline-bootstrap-questions.md`.

`compositions.md` is special — it catalogues *combinations* of single-issue questions that test capabilities (enumeration, synthesis, error-attribution, interference detection) which the component questions can't measure on their own.

[`benchmark-candidates/README.md`](benchmark-candidates/README.md) spells out the workflow and is required reading before adding a new question.

---

## `dev/journals/`

**What it is.** Postmortem journal of bugs that took >1 hour to diagnose. Newest-first, written in the moment so the symptom language matches what a future debugger would search for.

**Why an agent should care.**

- **Read** here first when you encounter a confusing symptom in the same area as a past entry. The grep-target is the symptom description (e.g. "OOM in deformable attention", "pos_embed shape mismatch", "ModuleNotFoundError: onnxscript"). The entries are deliberately written so the symptom keywords match what you'd search for from inside the bug, not the technically correct vocabulary that comes after diagnosis.
- **Write** here after a fix that took >1 hour to diagnose. The goal is for the next person to recognise the bug 10× faster. Skip the narrative — the format at the end of the file shows the canonical shape: Symptom, Root cause, Fix, Takeaway.

**Layout.**

```text
journals/
  lessons_learned.md   # All entries, newest first
```

(Plus eventually per-area journals if the file grows past ~50 entries.)

---

## How `dev/` relates to the rest of the repo

```text
PLAN.md                  -> the original handoff plan from the prior agent
AGENT_PROMPT.md          -> the first-prompt for the current agent
kwcoco_detector_kit/     -> the package itself
examples/                -> runnable example projects
tests/                   -> pytest suite
docs/                    -> end-user documentation
dev/                     -> long-running engineering memory (you are here)
  benchmark-candidates/  -> distilled "hard question" corpus from real mistakes
  journals/              -> >1hr-debug-time bug postmortems
```

When in doubt about *where* to write something, ask: would a brand-new agent landing in this repo benefit from reading it cold, without a conversation context? If yes → `dev/`.

---

## Quality bar (one paragraph)

Don't add entries that record trivia. Both subtrees are deliberately curated; an over-long file is a worse signal than a shorter one, because future readers won't believe the "important" entries hidden between filler. If you're unsure whether something belongs, write it in your scratch notes first; if a week later you still think the lesson is durable, move it in.
