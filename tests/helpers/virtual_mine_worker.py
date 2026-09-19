"""Subprocess entrypoint for virtual-mining interruption tests."""
import json
import sys

from kwcoco_detector_kit.data.mine import MineConfig, run


data = json.loads(open(sys.argv[1]).read())
run(MineConfig.cli(argv=False, data=data))
