# kwcoco_detector_kit

A domain-agnostic Python package for training object detectors on **kwcoco**
datasets. Scales from a single 12 GB GTX 1080 Ti up through 4× 96 GB Blackwell /
A100 / H100 clusters, and from small mobile detectors (DEIMv2 HGNetv2 Atto,
~0.5 M params) to large DINOv2/DINOv3-backed transformer detectors
(OpenGroundingDINO, ~200 M params). Targets land photos, aerial mosaics,
underwater, and (eventually) multispectral satellite imagery — anywhere a kwcoco
file with bounding-box annotations is a reasonable input.

## Status — pre-Phase-1 handoff

This repository contains the handoff plan + engineering-memory seed for an
incoming agent. **No package code has been written yet.**

```text
kwcoco_detector_kit/
├── README.md           ← you are here (orientation)
├── AGENT_PROMPT.md     ← FIRST PROMPT for the next agent — start here
├── PLAN.md             ← 740-line handoff plan (the authoritative spec)
├── .gitignore
└── dev/                ← engineering memory (read before writing code)
    ├── README.md
    ├── benchmark-candidates/
    │   ├── README.md                       ← workflow + quality bar
    │   ├── pipeline-bootstrap-questions.md ← 4 seeded hard-problem invariants
    │   └── compositions.md                 ← multi-invariant questions (placeholder)
    └── journals/
        └── lessons_learned.md              ← 19 seeded >1hr-debug postmortems
```

## For agents starting work here

1. Read **`AGENT_PROMPT.md`** first. It tells you what to read, what to confirm
   with the user, and what Phase 1 scope is.
2. Read **`PLAN.md`** cover-to-cover. It is the spec.
3. Read **`dev/README.md`** so you understand what the engineering memory tree
   is for, then skim the seeded entries in `dev/benchmark-candidates/` and
   `dev/journals/lessons_learned.md` — they encode hard-won invariants from the
   prior project that this kit must preserve.

## For humans browsing this repo

The two prior prototypes whose patterns are being lifted (read-only sources):

- `/home/joncrall/code/shitspotter/experiments/mobile_app_training_v4/` — small-detector lineage (DEIMv2 Atto/Femto/Pico/N)
- `/home/joncrall/code/shitspotter/experiments/mobile_app_training_v5/` — multi-scale tiles + hard-negative mining
- `/home/joncrall/code/shitspotter/experiments/foundation_detseg_v3/` — big-DINO lineage (OpenGroundingDINO + SAM2)

The kit's job is to absorb the engineering invariants from those prototypes
without absorbing the project-specific names ("poop", "shitspotter", "Pixel 5",
"v9 baseline", etc.).

## Three phases (per `PLAN.md`)

- **Phase 1** — package scaffold + DEIMv2 small-detector lineage + kwcoco-demo
  example + pytest suite (≥ 80 tests). Single-GPU + CPU smoke. Target: `pip
  install -e . && pytest -q` passes; `run_smoke.sh` produces a `.onnx` in <90s.
- **Phase 2** — OpenGroundingDINO big-detector lineage, multi-GPU DDP, NOAA
  Steller sealion example.
- **Phase 3** — webdataset shard pipeline (via `kwcoco_dataloader`),
  multispectral support, cloud-cluster target.

Stop at each phase boundary and confirm with the user.
