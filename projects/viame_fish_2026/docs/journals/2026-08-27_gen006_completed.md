# gen006 finished: the machinery worked, the model peaked at epoch 4 (2026-08-27)

The first fish run configured from DEIMv2's actual recipe completed all 14
epochs, 27.9 h of training over a 30.4 h span, exit clean. Every mechanism this
project spent a week repairing behaved correctly. The model still peaked at
**epoch 4 of 14** and every epoch after that was worse.

## Result

| epoch | loss | tile-vali AP | | epoch | loss | tile-vali AP |
|---|---|---|---|---|---|---|
| 0 | 35.28 | 0.526 | | 7 | 28.19 | 0.526 |
| 1 | 35.29 | 0.531 | | 8 | 27.66 | 0.527 |
| 2 | 34.16 | 0.532 | | 9 | 27.25 | 0.524 |
| 3 | 33.65 | 0.532 | | 10 | 26.82 | 0.522 |
| **4** | **33.20** | **0.53285** | | 11 | 26.80 | 0.522 |
| 5 | 32.93 | 0.531 | | 12 | 31.02 | 0.531 |
| 6 | 32.70 | 0.530 | | 13 | 31.02 | 0.529 |

`best_stg2.pth` was never written, which is correct: it is only produced when an
epoch at or past `stop_epoch` beats the global best, and none did. DEIMv2 kept
epoch 4.

## Correction: epoch 12 was not a recovery

I reported the 0.522 -> 0.531 jump at epoch 12 as the clean-NoAug phase
recovering. It was not. It is a **restore**.

`stop_epoch = 12` triggers two things in `det_solver`, and both fired:

```
23:01:40  Load optimizer/scaler/lr_warmup ... Refresh EMA at epoch 12 (0.9999)
01:10:36  Load optimizer/scaler/lr_warmup ... Refresh EMA at epoch 12 (0.9998)
```

The first is the top-of-loop stage transition; the second is the bottom-of-loop
restore-the-best after epoch 12's eval failed to beat epoch 4. Start-of-epoch
loss makes it unambiguous:

```
epoch 11 starts at 25.69
epoch 12 starts at 31.85   <- reset to best_stg1.pth (epoch 4 weights)
epoch 13 starts at 33.03   <- reset again
```

So epochs 12 and 13 are not the continuation of epochs 0-11. They are each
**one clean NoAug epoch starting from epoch-4 weights**, which is why their
end-of-epoch loss is identical (31.02) and their scores are close (0.531,
0.529).

That makes them a cleaner experiment than a recovery would have been, and the
answer is negative: **epoch-4 weights plus one epoch of clean low-LR NoAug
fine-tuning scores 0.529-0.531, no better than epoch 4's own 0.53285.**

This is the stage-2 mechanism working exactly as upstream designed -- restore
the best, fine-tune without augmentation -- reaching a fish run for the first
time. It simply did not find anything.

## What the shape actually says

The pre-registered call was "under 0.530 at epoch 11 means 14 epochs was too
short". It was 0.522, so the criterion fired. But the completed curve does not
support "too short" as the explanation:

- AP peaked at **epoch 4**, in the middle of the augmented phase, and declined
  monotonically from epoch 5 to epoch 11.
- Training loss fell the whole way (33.20 -> 26.80) while vali AP fell with it.
  That is fitting the training distribution at the expense of generalisation,
  not a model still climbing.
- The clean fine-tune from the best weights did not improve them.

More epochs of the same schedule would very likely have produced more of the
same decline. The candidate explanations now are that the model saturates near
0.533 on this data, or that the augmentation stack is too aggressive for tiles
that are already crops -- not that the run ended early.

## What is NOT concluded

**gen006's 0.53285 is not comparable to gen003's 0.5406.** gen006's vali is
TILE-level (69,284 tiles); gen003's was whole-image. Different denominators,
different object scales, different protocols. Reading the two side by side is
the same category error as comparing either to a test number.

The comparison that would settle it is the one the staged checkpoints exist
for: score gen001, gen003 and every gen006 epoch under one frozen protocol --
true-tiled, 1229 px source window, full vali. Until then gen006 is neither
better nor worse than gen003; it is unmeasured.

Epochs 12 and 13 are the specific candidates worth scoring, despite losing on
tile-level AP. They are the only epochs trained in the clean NoAug regime that
matches inference geometry, so they may rank differently under whole-frame
windowed evaluation than under tile-level validation.

## Everything mechanical worked

For the record, since these were the failures of gen001-gen005:

| | gen006 |
|---|---|
| NaN aborts | **0**, across 13 augmented epochs |
| destructive checkpoint reloads | **0** -- the only two reloads were the designed stage-2 restore at epoch 12 |
| schedule source | read from `deimv2_dinov3_x_coco.yml`, resolved to `[1,7,12]` / stop 12 / matcher 11 / flat 7 / wd 1.25e-4 |
| input normalization | ImageNet mean/std applied, DINOv3-gated |
| GradScaler | disabled under bf16 |
| holdout | untouched -- `do_eval/export/bench` all False |
| per-epoch staging | 14 checkpoints, 15 journal events, 11 GB, first live run of that wiring |
| completion | `.train_complete` written, 27.9 h train / 30.4 h span, 56 h limit |

Three fp16 runs died inside augmented epochs; bf16 went 13 for 13 here. gen004
took four destructive reloads in 19 epochs; gen006 took none in 14.

## Next

1. **Score the staged checkpoints.** `submit_baseline_vali.sh` already
   implements the frozen protocol for gen001/gen003; pointing it at gen006's 14
   staged epochs answers "what is B" and "which gen006 epoch is best" in one
   pass, without retraining.
2. Only then is there a real comparison, and only then should test be touched --
   once, on the single selected checkpoint.

The next TRAINING experiment should not be another schedule length. Two lengths
(24 and 14) and two batch sizes have now landed within noise of each other on
their own validation sets, and gen006's curve says the model degrades under
continued augmented training rather than plateauing. The open levers are the
augmentation stack on tiled input -- `RandomIoUCrop` and `RandomZoomOut` are
crop-on-crop when every sample is already a crop -- and input resolution
(retiling at `oversize_factor=1.0` for true 1:1 native pixels).
