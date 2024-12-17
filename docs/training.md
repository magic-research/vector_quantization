# Tokenizers

The training command is as follows:

```bash
auto_torchrun -m vq.train ${EXPERIMENT_NAME} ${CONFIG_NAME} \
    --config-options ... \
    --override ... \
    --autocast \
    --load-model-from ... \
    --load-from ... \
    --auto-resume
```

The `auto_torchrun` command is installed by the `todd_ai` package and is equivalent to `torchrun --nproc-per-node=${GPUS} --master-port=${PORT}`. You can always use `torchrun` as a workaround should `auto_torchrun` fail.

Checkout `work_dirs/${EXPERIMENT_NAME}` for the training products. Specifically, the checkpoints are stored under `work_dirs/${EXPERIMENT_NAME}/checkpoints`.

The `${CONFIG_NAME}` argument follows the format:

```text
configs/{model}/{codebook size}_{architecture detail}_{dataset}_{strategy}.py
```

- `model` is the name of the tokenizer. For example, `vqgan`, `cvqvae`, `fsq`, `vqkd`, or `cluster`.
- `codebook size` is the number of tokens in that can be used by the tokenizer. By default, VQ-KD uses 8192 tokens.
- `architecture detail` is the specific model architecture used in the tokenizer. For example, `dd2_aglwg075` refers to a model with the depth of discriminator being `2` and the adaptive generator loss weight gain being `0.75`.
- `dataset` is usually just `imagenet`.
- `strategy` is the parallel strategy used for training. Both `ddp` and `fsdp` are supported.

All other arguments are optional:

- `--config-options` and `--override` are related to config files:
  - `--config-options` passes options to the config file.
  - `--override` overrides the config file at runtime.
- `--autocast` enables automatic mixed precision training.
- `--load-model-from` specifies pretrained models to be loaded. For example, training a pixel decoder for VQ-KD requires loading the pretrained VQ-KD tokenizer.
- `--load-from` and `--auto-resume` enables resumption of training.
  - `--load-from work_dirs/${EXPERIMENT_NAME}/checkpoints/iter_${n}` resumes training from iteration `n`.
  - `--auto-resume` automatically resumes training from the latest checkpoint.

> If a training script uses `--load-model-from` and either `--load-from` or `--auto-resume`, the override `--override .trainer.callbacks[-1].load_state_dict:dict\(strict=False\)` should be specified.

This project adopts a two-stage framework:

- Tokenizers encodes images into tokens. Decoders are included for reconstructing images from tokens.
  - [VQGAN](../configs/vqgan/README.md)
  - [CVQ-VAE](../configs/cvqvae/README.md)
  - [FSQ](../configs/fsq/README.md)
  - [VQ-KD](../configs/vqkd/README.md)
  - [Cluster](../configs/cluster/README.md)
  - [decoder](../configs/decoder/README.md) for both VQ-KD and Cluster
- Proposal Networks generates image tokens for image synthesis.
  - [AR](../configs/ar/README.md)
