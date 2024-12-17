# Auto-Regressive Proposal Networks

The proposal network loads a pre-trained image reconstruction model. The config file of the image reconstruction model is specified via the `ir_config` field of `--config-options`. The state dict of the image reconstruction model is specified via the `ir_state_dict` field of `--config-options`. If the image reconstruction model relies on an image tokenizer that performs feature reconstruction, like VQ-KD, the `ir_state_dict` will not include the state dict of the image tokenizer. Therefore, an additional `it_state_dict` field is required to specify the state dict of the image tokenizer.

```bash
auto_torchrun -m vq.train ar/c2i_llama_medium_cfg_imagenet_ddp configs/ar/c2i_llama_medium_cfg_imagenet_ddp.py \
    --config-options \
        ir_config::work_dirs/decoder/llamagen/vqkd_clip_8192_imagenet_ddp/llamagen_8192_dd2_aglwg075_imagenet_ddp/llamagen.py \
        it_state_dict::work_dirs/vqkd/clip_8192_imagenet_ddp/checkpoints/iter_250000/model.pth \
        ir_state_dict::work_dirs/decoder/llamagen/vqkd_clip_8192_imagenet_ddp/llamagen_8192_dd2_aglwg075_imagenet_ddp/checkpoints/iter_400000/model.pth
```
