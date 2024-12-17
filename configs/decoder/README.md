# Pixel Decoders

For tokenizers like VQ-KD and Cluster, we need to train pixel decoders to reconstruct images from tokens.

```bash
# VQ-KD
auto_torchrun -m vq.train \
    decoder/llamagen/vqkd_clip_8192_imagenet_ddp/llamagen_8192_dd2_aglwg075_imagenet_ddp \
    configs/decoder/llamagen.py \
    --config-options it_config::configs/vqkd/clip_8192_imagenet_ddp.py \
    --load-model-from work_dirs/vqkd/clip_8192_imagenet_ddp/checkpoints/iter_250000/model.pth

# Cluster
auto_torchrun -m vq.train \
    decoder/llamagen/cluster_clip_8192_imagenet_ddp/llamagen_8192_dd2_aglwg075_imagenet_ddp \
    configs/decoder/llamagen.py \
    --config-options it_config::configs/cluster/clip_8192_imagenet_ddp.py \
    --load-model-from work_dirs/cluster/clip_8192_imagenet_ddp/checkpoints/iter_100000/model.pth
```

<!--

TODO

```bash
# VQ-KD
auto_torchrun -m vq.train \
    decoder/vqgan/vqkd_clip_8192_imagenet_ddp/vqgan_8192_dd2_aglwg075_imagenet_ddp \
    configs/decoder/vqgan.py \
    --config-options it_config::configs/vqkd/clip_8192_imagenet_ddp.py \
    --load-model-from work_dirs/dry_run/vqkd/clip_8192_imagenet_ddp/checkpoints/iter_7/model.pth \
    [--override .trainer.callbacks[-1].load_state_dict:dict\(strict=False\)]

# Cluster
auto_torchrun -m vq.train \
    decoder/llamagen/cluster_clip_8192_imagenet_ddp/llamagen_8192_dd2_aglwg075_imagenet_ddp \
    configs/decoder/llamagen.py \
    --config-options it_config::configs/cluster/clip_8192_imagenet_ddp.py \
    --load-model-from work_dirs/dry_run/cluster/clip_8192_imagenet_ddp/checkpoints/iter_7/model.pth \
    [--override .trainer.callbacks[-1].load_state_dict:dict\(strict=False\)]
```

-->
