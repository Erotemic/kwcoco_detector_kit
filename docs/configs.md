# Environment and Dataset Configs

`kwcoco-detector-kit` separates the pieces that are hard to infer from the
pieces that usually can be inferred.

Training config can usually be derived from the trainer, model variant, image
scale, GPU count, and a scale tier. Environment and dataset config are different:
they depend on the host, scheduler, container image, filesystem, and local
kwcoco paths. Those live in editable YAML files.

## Initialize

```bash
kwcoco-detector-kit config-init \
    --env kcd.environment.yaml \
    --dataset kcd.dataset.yaml \
    --execution slurm-docker \
    --docker_image kwcoco-detector-kit:ogdino-cu132-arisia \
    --train_kwcoco /path/to/train.kwcoco.zip \
    --vali_kwcoco /path/to/vali.kwcoco.zip
```

The generated files contain a `suggestions.introspection` section with facts
observed from the host and kwcoco files. The operative values are under
`environment:` and `dataset:`.

## Inspect

```bash
kwcoco-detector-kit config-inspect \
    --env kcd.environment.yaml \
    --dataset kcd.dataset.yaml \
    --refresh
```

`--refresh` recomputes suggestions from the current host and dataset paths.

## Edit

For an interactive text prompt:

```bash
kwcoco-detector-kit config-edit \
    --env kcd.environment.yaml \
    --dataset kcd.dataset.yaml
```

For scripted changes:

```bash
kwcoco-detector-kit config-edit \
    --env kcd.environment.yaml \
    --dataset kcd.dataset.yaml \
    --non_interactive \
    --set environment.slurm.gres=gpu:4 \
    --set dataset.tiling.tile_size=1024
```

The files are plain YAML, so editing them directly is equally valid.
