"""Subprocess helper for cache publication interaction tests."""
import json
import sys

from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache


cache_root, material_path, payload_path = sys.argv[1:]
material = json.loads(open(material_path).read())
payload = open(payload_path, "rb").read()
path, created = TileMaterializationCache(cache_root).publish_bytes(
    material, payload, suffix="jpg",
)
print(json.dumps({"path": str(path), "created": created}), flush=True)
