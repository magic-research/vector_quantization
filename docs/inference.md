# Inference

TBD.

<!--

TODO

```bash
torchrun --nproc_per_node=1 -m vq.inference configs/vqkd_clip_8192_dd3_imagenet_ddp.py data/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG --load-model-from work_dirs/vqkd_8192_imagenet_ddp_autocast/checkpoints/iter_250000/model.pth
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

-->
