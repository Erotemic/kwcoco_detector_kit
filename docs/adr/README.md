# Architecture Decision Records (ADRs)

This directory captures load-bearing architecture decisions for
`kwcoco-detector-kit` and the projects that consume it. Each ADR is a
short, immutable record of *why* we chose what we chose, so future
contributors don't have to re-derive the constraints from scratch.

## Format

One file per decision, numbered sequentially:

```
docs/adr/NNNN-kebab-case-title.md
```

Each ADR follows a fixed shape:

- **Status** — `proposed` / `accepted` / `superseded by ADR-XXXX` / `deprecated`
- **Context** — what problem are we solving, what constraints are in play
- **Decision** — what we're committing to
- **Consequences** — what this enables, what this rules out, what we
  accept as the cost
- **Alternatives considered** — short paragraph each; why each was rejected

Keep them short. ADRs are not design docs — they're the *commitments*
extracted from design discussions. The companion design discussion
lives in `docs/` (the user-facing docs) or `dev/` (engineering memory).

## Conventions

- ADRs are **immutable once accepted**. If we change our mind, write a
  new ADR that supersedes the old one (referenced in the new ADR's
  Status line and back-pointed in the old ADR's Status line).
- Cross-link related ADRs with relative paths: `[ADR-0001](0001-...md)`.
- Cross-link to design docs / journal entries when relevant.
- Aim for ~1 ADR per quarter on average. ADRs that capture a single
  bug-fix or refactor are too granular.

## Index

| # | Title | Status |
|---|---|---|
| [0001](0001-dual-tile-store-backends.md) | KwcocoJpegStore + WebdatasetStore both stay first-class | accepted |
