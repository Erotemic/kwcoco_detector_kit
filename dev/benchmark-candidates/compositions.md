# Compositional benchmark candidates

This file catalogues benchmark questions whose **interest comes from
multi-invariant interference** — not just from any single invariant in
`pipeline-bootstrap-questions.md` or its sibling files.

A composition belongs here when an agent's score on it is **not predictable**
from their scores on the components. If solving the composition is just
"solve the worst component," it's a stand-in for that component and should not
be catalogued here.

## When to add an entry

Add when:

- A real maintenance task surfaces **two or more** invariants from the
  single-issue files at the same time.
- The agent's mistake was *not* on any individual invariant but on the
  **interaction** between them (e.g. a single-fix attempt that resolves
  invariant A re-introduces invariant B; or the agent enumerates A and B
  correctly but doesn't realise they constrain a shared field).
- The shape of the resulting question requires the agent to **enumerate**,
  **prioritise**, and **interleave** fixes across the invariants.

Do **not** add when:

- The composition is just "list of bugs to fix" — that's a single-issue
  walkthrough question, not a composition.
- The interference is purely temporal ("first do A, then do B"). Real
  compositions involve shared state, shared fields, or shared resources.

## Entry template

```markdown
## C{N} — {short title}

Status: draft
Level: A | B
Component questions: Q{a}, Q{b}, ... (cross-reference the single-issue files)
Source commit(s): {git SHAs that motivated this composition}

### Source context

What was the agent doing? What were the two-plus invariants in play?

### The hard question

A single, compact question whose correct answer requires preserving ALL
component invariants simultaneously.

### Why this composition is worth catalogueing

What does the composition test that the components don't? Examples:

- **Enumeration** — can the agent identify all invariants from the task shape?
- **Synthesis** — can they hold several in working memory while drafting?
- **Error attribution** — given a failure, can they tell which invariant fired
  vs. which is just downstream?
- **Interference detection** — can they spot when a fix for A breaks B?

### Expected answer

What does the correct minimal-patch look like? Include both the structural
shape and the rationale.

### Acceptance criteria

What pytest fixtures / static checks would catch the composed bug?
```

---

## Catalogued compositions

*(none yet — first one lands when an end-to-end Phase 1 task surfaces two
invariants interfering with each other.)*

### Candidates to write up when they materialise

These are flagged in `pipeline-bootstrap-questions.md`'s composition note and
would naturally become catalogued compositions if observed in a live run:

- **Q1 + Q3 cross-product** — agent generates a HGNetv2 config with multi-scale
  enabled (Q3 violation) AND a YAML indent bug (Q1 violation). The resulting
  `with_pos_embed` error looks like a Q3 root cause but is actually Q1 — the
  collate that should have suppressed multi-scale never landed in the right
  spot in the YAML. Tests **error-attribution** under invariant interference.
- **Q2 + Q4** — agent's pre-flight passes (all imports resolve), but didn't
  probe FD limits. First multi-worker run hits Q4 at training time. Tests
  **enumeration completeness** — did the pre-flight check enumerate all O(N)
  failure categories, including the resource-limit one?

Write these up as actual `C1`, `C2`, ... entries when a maintenance task
reproduces them and the resulting fix touches both invariants in one patch.
