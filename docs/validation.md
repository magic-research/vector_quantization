# Validation

The validation command follows a similar pattern as the training command:

```bash
auto_torchrun -m vq.val ${EXPERIMENT_NAME} ${CONFIG_NAME} \
    --config-options ... \
    --override ... \
    --visual ... \
    --autocast \
    --load-model-from ... \
    --load-from ...
```

`--config-options`, `--override`, `--autocast`, and `--load-model-from` have the same meanings as in the training command.

If the `--visual` argument is present, visualizations will be saved under `work_dirs/${EXPERIMENT_NAME}/unbatched_visuals`. The value of `--visual` specifies a regex to filter the images to visualize. In most cases, `--visual pred_image` is sufficient to save the reconstructed images.

In `vq.train`, `--load-from` is used to resume training. However, in `vq.val`, `--load-from` is used to specify the checkpoints to validate. By default, `vq.val` automatically validates all checkpoints found under `work_dirs/${EXPERIMENT_NAME}/checkpoints`. If `--load-from iter_{15..26}0000` is provided, only `work_dirs/${EXPERIMENT_NAME}/iter_{15..26}0000` are validated.

To construct a validation command from the training command, simply replace `vq.train` with `vq.val`. For example, the validation command for VQ-KD decoderis:

```bash
# auto_torchrun -m vq.train \
#     decoder/llamagen/vqkd_clip_8192_imagenet_ddp/llamagen_8192_dd2_aglwg075_imagenet_ddp \
#     configs/decoder/llamagen.py \
#     --config-options it_config::configs/vqkd/clip_8192_imagenet_ddp.py \
#     --load-model-from work_dirs/vqkd/clip_8192_imagenet_ddp/checkpoints/iter_250000/model.pth

auto_torchrun -m vq.val \
    decoder/llamagen/vqkd_clip_8192_imagenet_ddp/llamagen_8192_dd2_aglwg075_imagenet_ddp \
    configs/decoder/llamagen.py \
    --config-options it_config::configs/vqkd/clip_8192_imagenet_ddp.py \
    --load-model-from work_dirs/vqkd/clip_8192_imagenet_ddp/checkpoints/iter_250000/model.pth
```

To test a single checkpoint, we also provie the `vq.test` command:

```bash
auto_torchrun -m vq.test ${EXPERIMENT_NAME} ${CONFIG_NAME} \
    --config-options ... \
    --override ... \
    --visual ... \
    --autocast \
    --load-model-from ...
```

Compared to `va.val`, the `--load-from` argument is removed. The `vq.test` command validates a single checkpoint specified by `--load-model-from`.
