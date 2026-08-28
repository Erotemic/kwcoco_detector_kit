"""A launcher must pin the split it trains on, not inherit it.

`paths.sh` defaults `KCD_TRAIN_KWCOCO` / `KCD_VALI_KWCOCO` to the UNTILED
bundles. A tiled-recipe launcher that forgets to reassign them either

  * trains at 1920x1080 whole-frame scale rather than the 1229px tile scale the
    recipe, the eval window and the sampler weights are all built around -- and
    the weights are positionally meaningless besides, being one-per-tile over a
    495,514-tile corpus while the dataset would hold 251,143 frames; or
  * picks up whatever the calling shell happens to have exported, which makes
    the dataset of a 13-hour run depend on ambient state.

gen007 shipped without that block. In a clean shell it aborted on its own
`KCD_TILE_SOURCE_KWCOCO == KCD_TRAIN_KWCOCO` guard -- fail-closed, but the
guard was catching an omission the script should never have had. With a stale
`KCD_TRAIN_KWCOCO` exported it would have run, on the wrong data, silently.
"""
import re
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _submit_scripts():
    return sorted(SCRIPTS.glob("submit_train_*.sh"))


def _enables_sequence_balance(text):
    return re.search(r'^\s*export\s+KCD_BALANCE_SEQUENCE=True', text, re.M)


def test_there_is_something_to_check():
    assert _submit_scripts(), f"no submit_train_*.sh under {SCRIPTS}"


@pytest.mark.parametrize("fpath", _submit_scripts(), ids=lambda p: p.stem)
def test_tiled_launchers_pin_both_splits(fpath):
    """Any launcher that balances sequences trains on tiles by definition."""
    text = fpath.read_text()
    if not _enables_sequence_balance(text):
        pytest.skip("not a sequence-balanced tiled launcher")
    for var, src in (("KCD_TRAIN_KWCOCO", "KCD_TILE_TRAIN_KWCOCO"),
                     ("KCD_VALI_KWCOCO", "KCD_TILE_VALI_KWCOCO")):
        assert re.search(rf'^\s*export\s+{var}="\$\{{?{src}\}}?"', text, re.M), (
            f"{fpath.name} never assigns {var} from {src}; it would inherit "
            f"the UNTILED bundle from paths.sh")


@pytest.mark.parametrize("fpath", _submit_scripts(), ids=lambda p: p.stem)
def test_the_split_pin_precedes_everything_that_reads_it(fpath):
    """Order matters: the balance step and its guard both read KCD_TRAIN_KWCOCO."""
    text = fpath.read_text()
    if not _enables_sequence_balance(text):
        pytest.skip("not a sequence-balanced tiled launcher")
    pin = re.search(r'^\s*export\s+KCD_TRAIN_KWCOCO="\$\{?KCD_TILE_TRAIN_KWCOCO\}?"',
                    text, re.M)
    assert pin, "no tiled train pin at all"
    for reader in (r'^\s*export\s+KCD_BALANCE_SEQUENCE=True',
                   r'KCD_TILE_SOURCE_KWCOCO"\s*=\s*"\$KCD_TRAIN_KWCOCO'):
        m = re.search(reader, text, re.M)
        if m:
            assert pin.start() < m.start(), (
                f"{fpath.name}: KCD_TRAIN_KWCOCO is pinned at offset "
                f"{pin.start()} but read at {m.start()}")


@pytest.mark.parametrize("fpath", _submit_scripts(), ids=lambda p: p.stem)
def test_sequence_source_is_the_untiled_bundle(fpath):
    """Sequence identity exists only in the source; equal paths mean no balance.

    The tiler stamps tile_source_gid but not video_id, so grouping must join
    tile -> source frame -> sequence. If both point at the same file every tile
    is its own sequence and the weighting silently does nothing.
    """
    text = fpath.read_text()
    if not _enables_sequence_balance(text):
        pytest.skip("not a sequence-balanced tiled launcher")
    assert 'KCD_TILE_SOURCE_KWCOCO' in text
    assert re.search(r'KCD_TILE_SOURCE_KWCOCO"\s*=\s*"\$KCD_TRAIN_KWCOCO', text), (
        f"{fpath.name} does not guard against the source and train bundles "
        "being the same file")
