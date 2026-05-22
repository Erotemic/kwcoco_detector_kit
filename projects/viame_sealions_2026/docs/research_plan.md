# Sea Lion Detector Research Plan

Last updated: 2026-05-21

This plan tracks which detectors we want to train, what class-collapse scheme
each one uses, and how we promote runs from baseline → operational.

The companion artifacts:
- [class_schemes.yaml](class_schemes.yaml) — declarative source→target class mappings
- [training_runs.yaml](training_runs.yaml) — the run registry (synced between hosts)
- [../scripts/training_registry.py](../scripts/training_registry.py) — CLI to query/update the registry

## Source class distribution (89,955 annotations across all years)

| code | full name           | count  | positive | notes                          |
|------|---------------------|-------:|:--------:|--------------------------------|
| F    | female              | 20,500 | yes      | most abundant adult            |
| J    | juvenile            | 19,478 | yes      |                                |
| P    | pup                 | 18,634 | yes      | small object class             |
| NFS  | northern fur seal   | 15,965 | yes      | different species — distractor |
| B    | bull                |  5,947 | yes      | adult male                     |
| O    | background          |  5,909 | no       | hard negatives                 |
| S    | subadult male       |  3,202 | yes      | thin class                     |
| DP   | dead pup            |    301 | yes      | very rare                      |
| DN   | dead non-pup        |     19 | yes      | extremely rare — drop          |

## Class-collapse schemes (P0 → P2)

### P0: `single_sealion` (1 class)

Goal: prove the pipeline end-to-end and establish a localization-only ceiling.

- Target classes: `sealion`
- Source mapping: B, S, F, J, P, NFS, DP → `sealion`
- Drop: O (background), DN (only 19 samples)
- Already built: [training_ready_v1/all_collapsed.kwcoco.zip](../training_ready_v1/all_collapsed.kwcoco.zip)
- Why first: gives us a single-class detection mAP to compare against later
  multi-class runs, and it isolates localization quality from
  classification difficulty.

### P1: `pup_vs_nonpup` (2 class)

Goal: smallest useful operational split — pup counts are the highest-value
biological signal, and pups are small and visually distinct.

- Target classes: `pup`, `nonpup_sealion`
- Source mapping:
  - `pup`            ← P
  - `nonpup_sealion` ← B, S, F, J
- Drop: NFS, O, DP (too few alive pups dead anyway → consider folding into `pup`
  in a later ablation), DN
- Why second: gives a binary classifier that we can validate against pup
  counts in the metadata; small-object recall on pups is the most
  scientifically valuable single metric here.

### P2: `lifestage_6cls` (6 class)

Goal: full operational age-sex classifier including the species distractor.

- Target classes: `bull`, `subadult_male`, `female`, `juvenile`, `pup`, `northern_fur_seal`
- Source mapping: 1-to-1 from B, S, F, J, P, NFS
- Drop: O, DP, DN (revisit DP if a per-class fine-tune is worthwhile)
- Why last: subadult_male is thin (3.2k) and will likely be the weakest
  class. We want a strong localization backbone (from P0) and a known-good
  small-object regime (from P1) before tackling this.

## Model variants — phased rollout

We use [kwcoco_detector_kit](../../../../../code/kwcoco_detector_kit)'s `deimv2`
trainer with HGNetv2 (mobile, fixed input) and DINOv3 (server, dynamic input)
backbones. Each (scheme × variant) combination is one registry row.

Phase ordering — train in this order and gate each phase on the previous:

1. **Smoke**: `mock_tiny` on `single_sealion`, CPU/single GPU. Confirms pipeline on every new host before burning real GPU time.
2. **Baseline mobile**: `deimv2_hgnetv2_s` on `single_sealion`, multiscale 512–768, ~30 epochs. Establishes the mobile-deployable detection AP floor.
3. **Baseline server**: `deimv2_dinov3_s` on `single_sealion`. Establishes the server AP ceiling on the same data — diff is the "deployability tax".
4. **Operational pup**: `deimv2_dinov3_s` on `pup_vs_nonpup`. Track per-class AP and small-object AP — pup AP_small is the headline metric.
5. **Operational age-sex**: `deimv2_dinov3_m` (or `_l` if compute available) on `lifestage_6cls`. Track per-class AP, focus on subadult_male and NFS separability.
6. **Multi-scale ablation**: pick the best operational config and repeat with multiscale 384–1024 to confirm note from [../notes.md](../notes.md) that altitude varies.

For each phase: a successful run is `vali AP ≥ previous_phase × 0.95` (or
absolute target, whichever is higher) before moving on. If a phase regresses
the previous baseline, fix that before progressing.

## Open data questions

These should be resolved before phase 5, ideally before phase 4:

- **2025 RAW imagery** ([burlynb Girder folder](https://viame.kitware.com/girder#user/5f04993c9e3d20c2b572cbce/folder/68b9bbfab7b90ae9f3804986)) — does it have usable detections? If yes, it's a free test-set extension. Tracked in [../notes.md](../notes.md).
- **Pre-2021 redacted imagery** — [training_ready_v1/prepare_report.json](../training_ready_v1/prepare_report.json) currently only contains 2021–2024 (1,642 images). `splits_v1.json` references 15 sources / 8,218 assignments — the 2007–2019 norm.kwcoco.zip files likely don't exist on namek yet. Verify on arisia with [../scripts/check_split_status.py](../scripts/check_split_status.py).
- **Annotation conversion spotcheck** — flagged file in [../notes.md](../notes.md): `_viz__sealions_2021_2024_sample40.kwcoco_f98c785e/loose-images/_anns/null/32_null_20240701_UNGA_ACHEREDIN POINT_SLP01717_KLS.jpg.view_ann.jpg`. Until resolved, any per-class metric is suspect.
- **Multi-scale necessity** — altitude varies; we should record per-image GSD/altitude metadata in the registry once available.

## Workflow

When starting a new training run:

1. Decide (scheme, variant, host). Pick the next phase from above unless we're filling a gap.
2. Generate the per-scheme kwcoco if not already present (see [class_schemes.yaml](class_schemes.yaml) for the `include_cats` / target-name mapping to pass to `prepare_training_kwcoco.py`).
3. Add a registry row: `python3 scripts/training_registry.py add --id ...`. The CLI assigns a deterministic ID and pre-fills the date and host.
4. Launch the run (kit's `sweep` command). Record the `kcd_root` path in the registry.
5. After the run, update with `python3 scripts/training_registry.py update <id> --status done --metric map=...`.
6. Rsync `docs/training_runs.yaml` between hosts so the registry stays in sync. [class_schemes.yaml](class_schemes.yaml) is also synced but changes rarely.

## Result-tracking conventions

Always record at least:
- `vali_map` (COCO mAP @ 0.5:0.95)
- `vali_map50` (mAP @ 0.5 — closer to operational counting performance)
- `vali_ap_small` when pups are in the scheme
- `test_map` / `test_map50` once the run is final
- Pointer to the `eval/detect_metrics.json` produced by the kit
- Pointer to the ONNX export if one was produced

Per-class AP goes in `metrics.per_class` as a dict — the CLI doesn't validate
key names so we can match whatever the kit emits.
