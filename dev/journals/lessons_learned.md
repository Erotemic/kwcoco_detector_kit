# Lessons learned

Postmortems of bugs that took >1 hour to diagnose, newest-first. Format: Symptom / Root cause / Fix / Takeaway.

The grep target on this file is the **symptom**, not the technically correct vocabulary that comes after diagnosis. Write entries in the language a future debugger would use *from inside the bug*.

The bar is "took >1 hour" or "would have been ≥10× faster with this entry on file."

---

## SEED ENTRIES — from `/home/joncrall/code/shitspotter/dev/journals/lessons_learned.md`

The following entries are scrubbed and ported from the prior project. They capture the 19 failure modes that cost real time in the v4/v5 prototype. They are pre-seeded here so the kit doesn't have to re-encounter them.

When you (the next agent) encounter a NEW failure mode worth >1 hour of someone's time, add it above the SEED ENTRIES block. Newest-first.

---

### Seed #19 — `kwcoco subset --select_images` requires the `jq` Python package

**Symptom:** `ModuleNotFoundError: No module named 'jq'` when running `python -m kwcoco subset --select_images ".id <= 8"`.

**Root cause:** kwcoco's `--select_images` flag uses jq syntax under the hood and requires the `jq` Python binding, which isn't a declared kwcoco dependency.

**Fix:** Use the deprecated `--gids 1,2,3,4` form (which doesn't need jq) OR `pip install jq` first.

**Takeaway:** Document the `--gids` form as the kit's canonical subset CLI. If anyone reaches for `--select_images`, the kit's check-env should remind them to install `jq`.

---

### Seed #18 — DEIMv2 HGNetv2 fixed-input invariant requires FIVE sizes to match

**Symptom:** `RuntimeError: The size of tensor a (400) must match the size of tensor b (100) at non-singleton dimension 1` deep inside `with_pos_embed` during eval, even though training appeared to work.

**Root cause:** A DEIMv2 HGNetv2 encoder pre-bakes its positional embedding at `eval_spatial_size`. For a generated config with `eval_spatial_size=[320,320]` but inherited upstream `Resize: [640, 640]` in train + val transforms (from `tpl/DEIMv2/configs/base/dataloader.yml`), the eval batch tensor was 20×20=400 feature elements but pos_embed was 10×10=100. Training had worked because the encoder dynamically resizes pos_embed during training, but eval uses the pre-baked one.

**Fix:** Override ALL FIVE of these together in the generated YAML:
- `eval_spatial_size`
- `train_dataloader.dataset.transforms[Resize].size`
- `val_dataloader.dataset.transforms[Resize].size`
- `train_dataloader.collate_fn.base_size`
- `train_dataloader.dataset.transforms[Mosaic].output_size` (conventionally INPUT/2)

**Takeaway:** When the kit's config generator changes one size knob, it MUST also propagate to all five. The benchmark-candidate Q3 enforces this. Write a unit test that loops over (variant, input_hw) pairs and asserts the five values match.

---

### Seed #17 — Multi-GPU all-reduce bottlenecked by slowest PCIe peer

**Symptom:** A 2× RTX 3090 host trained *slower* in DDP mode than single-GPU on GPU 0.

**Root cause:** GPU 1 was on a 2× PCIe link (vs. 16× for GPU 0). Gradient all-reduce runs at the slowest peer's bandwidth.

**Fix:** Default `CUDA_VISIBLE_DEVICES=0` for single-host non-cluster configs. Probe PCIe link width via `nvidia-smi --query-gpu=pcie.link.width.current` at setup; warn when `num_gpus > 1` and any active GPU is below 8× PCIe.

**Takeaway:** Multi-GPU is not unconditionally a speedup. The kit's `--tier` flag and PCIe probe must opt into multi-GPU only when it actually helps.

---

### Seed #16 — DEIMv2 OOMs on 24 GB GPU with naïve upstream batch sizes

**Symptom:** `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 MiB. GPU 0 has a total capacity of 23.54 GiB of which 117.62 MiB is free.` during deformable-attention `sampling_locations` allocation.

**Root cause:** Upstream's `tpl/DEIMv2/configs/base/dataloader.yml` sets `total_batch_size: 32` assuming 8-GPU training (per-GPU 4). On a single 24 GB GPU at batch=128 per the prototype's initial default, deformable-attention intermediate tensors pushed the model past 24 GB even at 320×320 input.

**Fix:** Per-`(variant, input_hw, tier)` memory table. For deimv2_n: batch 32 at input ≤ 320, 24 at ≤ 512, 16 at > 512. Scale auto-shrinks with input area.

**Takeaway:** A flat per-variant batch default fails at larger input sizes. The lookup must be two-dimensional. Document the empirical anchor point: `deimv2_n @ 320×320 batch=32 → 8.1 GB training footprint` on a 3090.

---

### Seed #15 — Non-root users can't raise RLIMIT_NOFILE above `ulimit -Hn`

**Symptom:** Wrapper script tried to raise the soft FD limit to 65536 via `ulimit -n 65536`, got "Operation not permitted", and printed "WARNING: failed to raise" without actually changing anything safer.

**Root cause:** Non-root users can only raise the soft limit up to the hard limit (`ulimit -Hn`); going above requires CAP_SYS_RESOURCE. On hosts where the kernel/systemd cap is < 65536, the wrapper would fail without a useful fallback.

**Fix:** Clamp the requested limit to `ulimit -Hn` before raising. Below 16384, direct the user to set `*_TORCH_MP_SHARING=file_system` as a kernel-cap-independent fallback (switches torch IPC from FD-per-tensor to filesystem-backed shared memory).

**Takeaway:** Don't assume your wrapper can raise resource limits arbitrarily. Always clamp to the hard cap; always have an alternative path.

---

### Seed #14 — HGNetv2 hybrid encoder doesn't support multi-scale (pos_embed mismatch)

**Symptom:** Same as Seed #18 but on the first training cell, not eval.

**Root cause:** A v4 sweep cell enabled `multiscale` collate for a HGNetv2 (DEIMv2-N) variant. The encoder pre-bakes pos_embed at `eval_spatial_size` and doesn't dynamically interpolate during training either. Upstream DEIMv2 explicitly sets `base_size_repeat: ~` for every HGNetv2 variant in its configs; only DINOv3-backed variants support multi-scale.

**Fix:** Per-variant `supports_dynamic_input()` method on the trainer plugin. The round-loop driver coerces `train_policy=multiscale → fixed` when False.

**Takeaway:** Architectural constraints leak through config. The kit's trainer plugin must expose them, and the orchestration layer must respect them. Benchmark candidate Q3 codifies this.

---

### Seed #13 — YAML `collate_fn` indent-leak (config generator bug)

**Symptom:** `TypeError: CocoDetection.__init__() got an unexpected keyword argument 'collate_fn'` deep inside DEIMv2's `workspace.create()`.

**Root cause:** The bash heredoc generator emitted `collate_fn:` with 4 spaces of indent, intending to land it as a sibling of `dataset` under `train_dataloader`. With the surrounding heredoc's indentation, it instead landed as a CHILD of `dataset`. `workspace.create()` then forwarded `collate_fn` as a kwarg to `CocoDetection.__init__`, which has no such parameter.

The trainer SURVIVED `bash -n`, SURVIVED `yaml.safe_load` (the document is *valid* YAML, just wrong), and only failed deep inside the framework at model-build time.

**Fix:** Dedent to 2 spaces. **Larger fix:** port YAML generation to Python with a `yaml.safe_load` round-trip + structural-invariant assertion before launching the trainer.

**Takeaway:** Bash-heredoc-templated YAML is fragile. Any structured-config generation by string templating needs a parse-validate step. The kit ports YAML-gen to Python explicitly to avoid this whole bug class. Benchmark candidate Q1 codifies this.

---

### Seed #12 — Sweep records `status=ok` after failed stage

**Symptom:** A sweep cell's training stage crashed but the sweep summary printed `status=ok` and proceeded to export against a missing checkpoint.

**Root cause:** The sweep's `run_cell` function was called inside `if run_cell ...; then` — bash's `errexit` behavior is surprising in `if` conditions, and a `printf ... status=ok` at the end of the function could still run after an inner pipeline failed (the pipeline's exit was hidden by `tee`).

**Fix:** Each stage gets its own explicit exit-code check; status is `fail_<stage>` on the first failure and never `ok` by default. Use `set +e` around the call, then read `$?` explicitly, so the parent script doesn't abort under `set -e`.

**Takeaway:** Never trust a default `ok` in shell error handling. Every stage that can fail must record its own status; the aggregate is the worst of the parts.

---

### Seed #11 — DEIMv2 has hidden runtime deps (faster_coco_eval, calflops, etc.)

**Symptom:** `ModuleNotFoundError: No module named 'faster_coco_eval'` 30 seconds into the first training cell, AFTER model construction.

**Root cause:** DEIMv2's `tpl/DEIMv2/engine/data/dataset/coco_dataset.py` imports `faster_coco_eval` at module top-level. The `tpl/DEIMv2/requirements.txt` declares it (and `calflops`, `transformers`, `tensorboard`, `scipy`, `PyYAML`), but no parent package installs them transitively.

**Fix:** Setup-time probe-and-install pulls all of `tpl/DEIMv2/requirements.txt`. Loop over each line, probe via `importlib.util.find_spec`, batch the missing ones into a single `pip install`.

**Takeaway:** Hidden transitive deps in submoduled trainers must be discovered at setup time, not at first run. Add a `--check-env` CLI that does this for every trainer plugin.

---

### Seed #10 — DEIMv2 ONNX export needs `onnxsim` and opset 18 on torch ≥ 2.5

**Symptom:** `ModuleNotFoundError: No module named 'onnxsim'` at end of export. Plus a `RuntimeError: No Adapter To Version $17 for Pad` warning during version conversion.

**Root cause:** DEIMv2's `tools/deployment/export_onnx.py --simplify` imports `onnxsim` after the ONNX is already written (the .onnx file is on disk; only the optional simplify step failed). The opset 17 → 18 conversion warning is because torch ≥ 2.5's dynamo exporter targets opset 18 internally and the Pad op has no opset-17 adapter.

**Fix:**
- Install `onnxsim` at setup time (added to ONNX trio).
- Default opset 18, not 17.
- The export wrapper detects `onnxsim` availability before passing `--simplify`; if missing, skip the flag and use the unsimplified ONNX.
- Recover the .onnx from the staging dir if the subprocess crashed during `--simplify` — the file is valid even without simplification.

**Takeaway:** Optional post-export steps can crash a subprocess that *already produced its main output*. Detect availability before passing optional flags; recover the artifact if it exists despite the crash.

---

### Seed #9 — `torch.onnx.export` on torch ≥ 2.5 imports `onnxscript` at function-call time

**Symptom:** `ModuleNotFoundError: No module named 'onnxscript'` from inside `torch.onnx.__init__` when calling `torch.onnx.export(...)`, even with `dynamo=False`.

**Root cause:** Modern torch's `torch.onnx.export` imports `_compat` → `_core` → `onnxscript` at function-call time, not module-import time. Passing `dynamo=False` doesn't avoid the import.

**Fix:** Declare `onnxscript` as `install_requires`. Setup-time probe adds it to the missing-deps list.

**Takeaway:** Top-level function-call-time imports are real and not preventable from the caller's side. The kit must declare every dep that *any code path* needs.

---

### Seed #8 — torch / torchvision ABI mismatch

**Symptom:** `RuntimeError: operator torchvision::nms does not exist` when importing torchvision.

**Root cause:** Independently-installed torch + torchvision with mismatched compiled-against versions. The torchvision C++ NMS op isn't registered for the installed torch's dispatcher.

**Fix:** Pin a matching pair in `pyproject.toml`. For torch 2.11, torchvision 0.26.

**Takeaway:** torchvision must always be installed alongside torch from the same matched pair. Use `pip install torch==X.Y torchvision==Z.W` from PyTorch's index, not separate installs.

---

### Seed #7 — pip 25+ rejects girder index due to missing PEP 700 metadata

**Symptom:** `ERROR: Index https://girder.github.io/large_image_wheels/ does not provide upload-time metadata.`

**Root cause:** pip 25.0+ enforces PEP 700 (upload-time metadata). Static-HTML wheel indexes (like the girder one) don't provide it.

**Fix:** Install from a direct wheel URL instead of via `--find-links`. Document the pattern in `docs/install.md`.

**Takeaway:** pip's index-discovery rules tighten over time. Direct wheel URLs always work; relying on a third-party index doesn't.

---

### Seed #6 — `geowatch.__init__` hard-imports `osgeo` at module load

**Symptom:** `ModuleNotFoundError: No module named 'osgeo'` from `geowatch/__init__.py` line 178 when importing any geowatch utility.

**Root cause:** geowatch has a `_execute_ordered_preimports()` that hard-imports `osgeo.gdal` at package init time. GDAL Python bindings need `libgdal` system libs and can't be installed cleanly via pip alone.

**Fix:** Make geowatch an OPTIONAL dep; soft-import via `try/except ImportError`. Document the manual GDAL install path (direct wheel from girder index) for users who need geowatch features.

**Takeaway:** Hard-importing optional system-lib-dependent packages at `__init__` time forces a dep onto every downstream user. The kit must NOT do this for any optional component.

---

### Seed #5 — `kwimage.imresize(interp='area')` fails on skimage backend without cv2

**Symptom:** `NotImplementedError: area` from `kwimage.im_transform._coerce_skimage_interpolation_order`.

**Root cause:** `kwimage.imresize` falls back to skimage when cv2 isn't installed. The skimage codepath doesn't recognise `'area'` interpolation.

**Fix:** Either install `opencv-python-headless` (preferred for headless servers) or catch `NotImplementedError` and retry with `'linear'`.

**Takeaway:** Library API "consistent" calls aren't really consistent when backends differ. Wrap such calls with try/except + a fallback that's known to work in all backends.

---

### Seed #4 — gdown silently writes HTML error pages

**Symptom:** Subsequent run sees a 4 KB `.pth` file at the expected path, treats it as cached, and downstream PyTorch crashes with a pickle error.

**Root cause:** When Google Drive throttles or requires confirmation, gdown writes an HTML error page to the destination path with exit code 0. The next run's `[ -f "$dst" ]` cache check accepts it.

**Fix:** Post-download size guard. Require ≥ 1 MiB for any checkpoint (well below the smallest legitimate one, well above any HTML stub). Re-download if too small. Hard error if still bogus after re-download.

**Takeaway:** Never trust `[ -f $dst ]` as "successfully downloaded." Always verify with a size threshold or a hash.

---

### Seed #3 — `kwimage.imwrite(..., imwrite_params=...)` is not a valid cv2 kwarg

**Symptom:** `cv2.error: imwrite() got an unexpected keyword argument 'imwrite_params'`.

**Root cause:** `kwimage.imwrite` forwards `**kwargs` straight to `cv2.imwrite`. cv2's JPEG-quality knob is a flat `params=[FLAG_INT, VALUE_INT, ...]` list, not a name=value kwarg.

**Fix:** Pass `params=[int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]`.

**Takeaway:** When a wrapper forwards `**kwargs`, the wrapped library's API is what matters, not the wrapper's friendlier-looking surface. Read the wrapped library's docs, not the wrapper's.

---

### Seed #2 — `kwimage.imresize(image, (W, H))` 2nd positional is `scale=`, not `dsize=`

**Symptom:** `cv2.error: Failed to allocate 44947419955200 bytes` (45 TB!) on the first image processed.

**Root cause:** Called `kwimage.imresize(image, (1280, 960), interpolation='area')` thinking the tuple was the target size. The 2nd positional argument is `scale`, so kwimage interpreted it as "scale by (1280, 960)" — a 4032×3024 phone image was asked to become ~5 megapixels wide.

**Fix:** Always pass `dsize=` keyword argument explicitly: `kwimage.imresize(image, dsize=(W, H), interpolation='area')`.

**Takeaway:** Library function positional arguments are not stable across versions or across "what the call would naturally read as." Pin every wrapper call to keyword arguments. This is a benchmark-worthy lesson: enforcing `dsize=` everywhere prevents this whole class.

---

### Seed #1 — gdown 6.x dropped `--fuzzy`

**Symptom:** `__main__.py: error: unrecognized arguments: --fuzzy`.

**Root cause:** gdown 6.0.0 removed the `--fuzzy` flag; the URL parser now handles every form natively, but the breaking CLI change wasn't surfaced clearly.

**Fix:** Pass the bare file ID as the first positional argument. Works in every gdown version.

**Takeaway:** Third-party CLI tools change incompatibly. Prefer the most stable form of the call (bare positional ID) over flags that may not exist.

---

## Empty section — first NEW lesson lands above this divider

When the kit's first >1hr-debug bug surfaces, write the entry above this line in the same Symptom / Root cause / Fix / Takeaway format. The seed entries below are reference material; new entries should reflect the kit's own history.
