"""In-container launchers may only reference forwarded (KCD_*) paths.

Regression for slurm job 493. `_launch_tiles.sh` read `$VF_TRAIN_KWCOCO`, but
`_submit_train.sh` forwards only variables matching `^KCD_` into docker
(_submit_train.sh:72). Inside the container `$HOME` is /root, so paths.sh
re-derived every VF_* path from it and the job died looking for
/root/ssd-data/fish_kcd/bundle/train.kwcoco.json.

The rule: anything the container must see is resolved on the host and exported
under a KCD_ name. These tests are static greps, so they need neither slurm nor
docker.

Note: the kit's own pytest has testpaths=["tests"] and does not collect project
tests, so this does NOT run in the Docker build gate. Run it from inside the
project subtree.
"""
import re
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

def _in_container_launchers():
    """Launchers dispatched via KCD_LAUNCH_SCRIPT, i.e. run inside the image.

    Derived from the assignments rather than globbed, so a launcher added
    through that mechanism is checked automatically -- and host-side scripts
    are excluded by construction. `_launch_viame_train.sh` is the reason this
    is not a glob: it runs on the HOST against the VIAME install (no docker at
    all), so its VF_* reads are correct.
    """
    names = {"_launch_train.sh"}          # _submit_train.sh's default
    for sh in SCRIPTS.glob("*.sh"):
        for m in re.finditer(r"KCD_LAUNCH_SCRIPT=[\"\']?\$?\{?[A-Za-z_]*:?-?"
                             r"(_launch_[a-z_]+\.sh)", sh.read_text()):
            names.add(m.group(1))
    return sorted(p for p in (SCRIPTS / n for n in names) if p.exists())


IN_CONTAINER = _in_container_launchers()

#: `$VF_FOO` or `${VF_FOO}` — a VF_ variable being READ.
VF_READ = re.compile(r"\$\{?VF_[A-Z0-9_]+")


def _strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(line.split(" #", 1)[0])
    return "\n".join(out)


def test_the_launcher_set_is_discovered():
    """Guard against the discovery silently matching nothing."""
    names = {p.name for p in IN_CONTAINER}
    assert {"_launch_train.sh", "_launch_tiles.sh",
            "_launch_export_score.sh"} <= names, names
    assert "_launch_viame_train.sh" not in names, (
        "the host-side VIAME launcher must not be checked; it never enters "
        "the container and its VF_* reads are correct")


@pytest.mark.parametrize("script", IN_CONTAINER, ids=lambda p: p.name)
def test_in_container_launcher_reads_no_vf_paths(script):
    hits = VF_READ.findall(_strip_comments(script.read_text()))
    assert not hits, (
        f"{script.name} reads {sorted(set(hits))}, which _submit_train.sh does "
        f"not forward into the container. Resolve on the host and export under "
        f"a KCD_ name instead (see paths.sh)."
    )


@pytest.mark.parametrize("name", ["KCD_TILE_TRAIN_KWCOCO",
                                  "KCD_TILE_VALI_KWCOCO",
                                  "KCD_TILE_DPATH"])
def test_tile_paths_are_bridged_to_forwarded_names(name):
    """paths.sh must export the KCD_ bridge the launchers rely on."""
    text = (SCRIPTS / "paths.sh").read_text()
    assert re.search(rf"^export {name}=", text, re.M), (
        f"paths.sh does not export {name}; _launch_tiles.sh would see an "
        f"empty value in-container.")
