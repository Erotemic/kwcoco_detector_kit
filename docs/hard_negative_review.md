# Hard-negative truth review

Hard-negative mining is not proof that an example is truly negative. A high-score
prediction on a nominally negative tile has two important interpretations:

1. the detector produced a genuine false positive, so the tile is useful hard
   background for the next training round; or
2. the detector found a real object missing from the truth annotations, so the
   dataset needs correction and the tile must **not** be trained as negative.

KDK keeps enough provenance in the virtual candidate index and mining ledgers to
make that distinction reviewable after scoring, without rescoring the model.

## Build a review queue

After all mining shards have completed and `mine-finalize` has written its
`*.selected_candidates.json` sidecar:

```bash
python -m kwcoco_detector_kit mine-review \
    --candidate_index /path/to/train_negative_candidates \
    --ledgers /path/to/rank0.mine_ledger.json /path/to/rank1.mine_ledger.json \
              /path/to/rank2.mine_ledger.json /path/to/rank3.mine_ledger.json \
    --selected_candidates /path/to/hard_negatives.kwcoco.selected_candidates.json \
    --dst_dpath /path/to/mining/review \
    --top_n 200 \
    --per_source 3
```

Omit `--selected_candidates` to inspect the highest-scoring records directly from
completed ledgers instead of restricting review to the finalizer's admitted set.

## Artifacts

`review_queue.json` and `review_queue.tsv` contain the ranked queue. Each row
includes stable tile identity, score, source KWCoco, source gid/path, tile extent,
top predicted box in both tile and source coordinates, current target-truth
counts, and blank `review_status` / `review_note` fields.

`review.kwcoco.zip` is a diagnostic source-image subset. Existing source truth is
preserved and mined predictions are added under the category
`__hard_negative_review__`. It exists to make existing KWCoco visualization tools
usable; it is not a new source of truth.

With previews enabled, `index.html` and `previews/*.jpg` provide a static browser
view sorted hardest-first:

- red: mined detector prediction;
- blue: mined tile extent in the source image;
- green: existing target-truth bounding boxes.

The queue also records an adjacent `.json` sidecar path when present. KDK does not
assume that sidecar is canonical truth; domain wrappers may give it stronger
meaning (for example a LabelMe workflow).

## Truth correction invalidates negative provenance

If review discovers missing truth, update the canonical annotation source first.
Then regenerate the source KWCoco and every truth-dependent candidate/pool artifact
before training on mined negatives. A candidate index only proves that a window was
safe under the truth fingerprint used when that index was built.
