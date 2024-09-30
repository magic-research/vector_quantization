# Vector Quantization

## To Do List

- implement make backup via hard link
- complete pre-commit-config.yaml
- remove open_clip/eva_clip dependencies

## Preparation

```bash
pipenv run python tools/prepare_checkpoints.py torchvision
pipenv run python tools/prepare_checkpoints.py beitv2
```

```bash
pipenv run python tools/convert_checkpoints.py beitv2 pretrained/beitv2/vqkd_encoder_base_decoder_1x768x12_clip.pth --check
pipenv run python tools/convert_checkpoints.py beitv2 pretrained/beitv2/vqkd_encoder_base_decoder_1x768x12_clip.pth --options with_decoder:True --check --suffix .converted.with_decoder
```

```bash
run -- bash tools/torchrun.sh -m vq.fid imagenet
run -- bash tools/torchrun.sh -m vq.fid imagenet --train --override .dataloader.batch_size:1024
```

## Training

Stage 1

```bash
bash tools/torchrun.sh -m vq.train vqgan/8192_dd2_aglwg075_imagenet_ddp configs/vqgan/8192_dd2_aglwg075_imagenet_ddp.py
bash tools/torchrun.sh -m vq.train vqkd/openclip_bigG_14_8192_imagenet_ddp configs/vqkd/openclip_bigG_14_8192_imagenet_ddp.py
```

Stage 2

```bash
bash tools/torchrun.sh -m vq.train \
    decoder/vqgan/vqkd_clip_8192_imagenet_ddp/vqgan_8192_dd2_aglwg075_imagenet_ddp \
    configs/decoder/vqgan.py \
    [--config-options it::configs/vqkd/clip_8192_imagenet_ddp.py ir::configs/vqgan/8192_dd2_aglwg075_imagenet_ddp.py] \
    --load-model-from work_dirs/vqkd/clip_8192_imagenet_ddp/checkpoints/iter_250000/model.pth \
    [--override .trainer.callbacks[-1].load_state_dict:dict\(strict=False\)]
```

Stage 3

```bash
bash tools/torchrun.sh -m vq.train ar/c2i_llama_medium_cfg_vqgan_8192_imagenet_ddp configs/ar/c2i_llama_medium_cfg_vqgan_8192_imagenet_ddp.py
```

## Validation

```bash
[DRY_RUN=True] run -- torchrun --nproc_per_node=1 -m vq.val cvq_vae_8192_imagenet_ddp_fix2 configs/cvq_vae_8192_imagenet_ddp.py [--load-from iter_{15..26}0000]
```

## Inference

```bash
torchrun --nproc_per_node=1 -m vq.inference configs/vqkd_clip_8192_dd3_imagenet_ddp.py data/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG --load-model-from work_dirs/vqkd_8192_imagenet_ddp_autocast/checkpoints/iter_250000/model.pth
```

## FID

Two approaches can be used to calculate FID: `fidelity` or `fid`.

`fidelity` is easy to start with but only supports single-GPU:

```bash
# ${image_root}/
# └── *.png
run -- python tools/fidelity.py ${dataset:-imagenet} ${image_root} --work-dir ${image_root}/..
run -- python tools/fidelity.py ${dataset:-imagenet} ${image_root} --train --work-dir ${image_root}/..
```

`fid` requires generating FID cache first.
`FIDMetric` relies on these cache files to online calculate FID.
Users can also manually calculate FID:

```bash
bash tools/torchrun.sh tools/fid.py imagenet_val ${image_root} --work-dir ${image_root}/..
bash tools/torchrun.sh tools/fid.py imagenet_train ${image_root} --work-dir ${image_root}/..
```

## Tokenize

To save tokens during validation, use `--tokenize`:

```bash
bash tools/torchrun.sh -m vq.test ${name} ${config} --tokenize --load-model-from ${load_model_from}
```

The above script introduces extra computation cost.
To run an tokenization only:

```bash
bash tools/torchrun.sh -m vq.tasks.image_tokenization.tokenize ${name} ${config} --override .validator.type::VQITRunnerRegistry.Tokenizer --load-model-from ${load_model_from}
bash tools/torchrun.sh -m vq.tasks.image_tokenization.tokenize ${name} ${config} --override .validator.type::VQITRunnerRegistry.Tokenizer --load-model-from ${load_model_from} --train
```

LlamaGen tokenizes the training dataset:

```bash
bash tools/torchrun.sh tools/llamagen.py llamagen/vqkd_8192_imagenet_ddp configs/vqkd/8192_imagenet_ddp.py --load-model-from work_dirs/models/vqkd_openclip_bigG_14_8192_imagenet_ddp_autocast/checkpoints/iter_250000/model.pth
```

## Visualize

```bash
python tools/visualize_patches.py configs/vqkd/clip_8192_imagenet_ddp.py work_dirs/vqkd_clip_tokenize_tokenize/tokens
python tools/visualize_patches.py configs/vqgan/8192_dd2_aglwg075_imagenet_ddp.py work_dirs/vqgan_tokenize_tokenize/tokens
```

## Utils

```bash
for file in work_dirs/conditional_image_transformer_min_vqgan_fsq_8192_imagenet_ddp/checkpoints/iter_*; do num=$(echo $file | grep -o '[0-9]\+0000'); if (( $num % 40000 != 0 )); then rm -r $file; fi; done
for file in work_dirs/teacher_conditional_image_transformer_min_vqgan_dino_cvq_vae_8192_imagenet_ddp/checkpoints/iter_*; do num=$(echo $file | grep -o '[0-9]\+0000'); if (( $num % 40000 != 0 )); then rm -r $file; fi; done
for file in work_dirs/teacher_conditional_image_transformer_min_vqgan_mae_cvq_vae_8192_imagenet_ddp/checkpoints/iter_*; do num=$(echo $file | grep -o '[0-9]\+0000'); if (( $num % 40000 != 0 )); then rm -r $file; fi; done
for file in work_dirs/teacher_conditional_image_transformer_min_vqgan_vit_cvq_vae_8192_imagenet_ddp/checkpoints/iter_*; do num=$(echo $file | grep -o '[0-9]\+0000'); if (( $num % 40000 != 0 )); then rm -r $file; fi; done
```
