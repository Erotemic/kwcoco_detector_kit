# kwcoco-detector-kit Smoketests

Small scenario ladders for proving a training environment before spending real
GPU-hours.

Each scenario directory is ordered from cheap plumbing checks to the real run.
The intent is to stop at the first failure, fix that layer, and rerun without
rediscovering the same setup issue in a full training job.

Current scenarios:

| directory | target |
|---|---|
| `dino_v2_4x/` | OpenGroundingDINO / DINOv2-style detector on 1x then 4x GPUs |

