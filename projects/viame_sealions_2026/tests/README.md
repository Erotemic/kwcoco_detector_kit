# Tests

- `unit/` — fast unit tests (run on every change). Each file is self-contained
  and uses small synthetic fixtures.
- `expensive/` — tests that need real data, large files, or non-trivial time
  (e.g. open the actual `training_ready_v1/*.kwcoco.zip` bundles). Run these
  manually before doing something risky, not on every change.

Run unit tests:

    python3 -m pytest tests/unit -q

Run everything (slow):

    python3 -m pytest tests -q

Run only expensive:

    python3 -m pytest tests/expensive -q
