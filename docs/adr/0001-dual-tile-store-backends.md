# ADR-0001: KwcocoJpegStore and WebdatasetStore both stay first-class

- **Status**: accepted (2026-05-29)
- **Authors**: Jon Crall + Claude
- **Related**:
  - [`docs/storage.md`](../storage.md) — the TileStore abstraction
  - [`docs/data_pipeline.md`](../data_pipeline.md) — the broader data-serving landscape
  - [`dev/fast_image_backend_design.md`](../../dev/fast_image_backend_design.md)
    — Phase 3 design notes

## Context

Phase 3 of the kit introduced the `TileStore` abstraction (`docs/storage.md`)
with two concrete backends:

- `KwcocoJpegStore` — one `.jpg` per tile + a kwcoco manifest.
  Random-access friendly, no build step, works on any filesystem.
- `WebdatasetStore` — `shard-NNNNNN.tar` archives + `_bundle_meta.json`.
  Sequential reads, much faster on rotational disks (HDDs) and on
  network filesystems, but requires a one-time "build the shards"
  conversion step.

Empirical results from the 2026-05 sealion dataloader benchmark on
namek's HDD-backed storage confirmed that webdataset shards deliver
substantial throughput improvements over per-JPEG kwcoco reads when
the underlying disk is rotational. See the bench commits on
`kwcoco_dataloader` `main` (e.g.
`f42b755 journals: workers=8 follow-up on namek HDD`) for the
measurements.

The natural follow-up question: **should webdataset become the new
default, or should kwcoco-jpeg stay as a first-class fast path?**

## Decision

**Both backends remain first-class and supported indefinitely.**
Neither is deprecated in favor of the other.

- `KwcocoJpegStore` stays the default for new tile bundles.
  Zero-build-step iteration on a fresh kwcoco bundle is a load-bearing
  experience for prototyping, smoke tests, examples, CI, and the
  "I want to look at the actual JPEGs on disk" debugging path.
- `WebdatasetStore` is opt-in via a recipe-level `tile_store: webdataset`
  setting (or the `convert-store` CLI). It is the recommended path
  when the user's training data lives on rotational disk, NFS, or any
  storage tier where per-tile random access is expensive relative to
  sequential bulk reads.
- The `TileStore` Protocol stays the abstraction boundary. Trainer
  plugins, data loaders, hard-neg mining, and eval all consume
  `TileStore` rather than the concrete backend.

## Consequences

**Enables:**

- Fresh-clone-to-first-training in <90 s on any filesystem via
  `KwcocoJpegStore` (the `run_smoke.sh` contract).
- Real production wall-clock improvements on HDD/NFS via
  `WebdatasetStore` without forcing every consumer to convert.
- A migration path: users can profile their actual IO bottleneck and
  switch backends based on evidence, not by upgrading the kit version.

**Rules out:**

- Removing kwcoco-jpeg as a backend in any future version. We accept
  that the kit carries two storage backends forever.
- Webdataset-only trainer plugins. Any new trainer must consume the
  `TileStore` Protocol, not a webdataset-specific API.
- Implicit format conversion at training start (users explicitly opt
  in via `convert-store` or by pointing the recipe at a pre-built WDS
  bundle).

**Accepted costs:**

- ~2x the test surface (every backend-touching test needs both flavors).
- A second on-disk format in the install footprint.
- Two `convert-store`-style code paths (one for kwcoco→wds, none yet
  for wds→kwcoco; round-tripping is not required and not committed to).

## Alternatives considered

- **Webdataset-only.** Rejected: the "I want to look at the JPEGs on
  disk" debugging affordance is too valuable to lose, and the
  always-build-shards step adds friction at the prototyping stage
  that hurts new-user onboarding.
- **Auto-promote to webdataset above N tiles.** Rejected as too
  magical: the right cutoff depends on filesystem characteristics
  (HDD vs SSD vs NVMe vs network) that the kit can't reliably probe.
  Explicit opt-in is clearer.
- **Defer the decision.** Rejected: users (shitspotter v10, sealion
  v4+) are already making per-recipe choices and need a stable
  commitment to point at.

## Operational notes

- The `tile_store:` field belongs on the `data:` block of `recipe.v1`
  (see [`docs/configs.md`](../configs.md) for the schema). Allowed
  values: `kwcoco_jpeg` (default), `webdataset`.
- The `convert-store` CLI builds a WDS bundle from any
  `TileStore`-readable source; idempotent.
- The kit's `kwcoco_dataloader` extras pin a recent enough
  `kwcoco-dataloader` release for the WDS path. If the install layer
  omits that extra, the kit falls back to `KwcocoJpegStore` with a
  one-line warning.
