# Audit scripts (2026-08-25)

Standalone, stdlib-only. They read kwcoco JSON directly so they run on any host
with python3 — no kit install, no docker, no numpy. That is the point: they were
written to audit data on a VM that has none of those.

- `audit_bundle_geometry.py <bundle.kwcoco.json> <label>` — degenerate boxes,
  inverted boxes, NaN/inf coords, out-of-bounds, area/bbox disagreement,
  normalized coords outside [0,1], box-size percentiles, anns-per-image.
- `score_headtohead_stdlib.py` — AP@0.5 over the saved RF-DETR and DEIMv2
  prediction sets. Greedy IoU>=0.5 matching, monotonic precision envelope,
  all-point interpolation, applied identically to both.

`score_headtohead_stdlib.py` is a CROSS-CHECK, not the authority.
`../score_headtohead.sh` runs `kwcoco eval`, which is the protocol every other
number in this project uses; prefer it when docker is available. The stdlib
version exists because it can run anywhere and because a second implementation
disagreeing would itself be informative.

Findings are written up in
`../../docs/journals/2026-08-25_data_audit_and_the_rfdetr_comparison.md`.
