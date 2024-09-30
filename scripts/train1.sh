bash tools/torchrun.sh -m vq.train \
    decoder/vqkd_l/vqkd_open_clip_bigG_14_8192_imagenet_ddp_autocast/llamagen_vqgan \
    configs/decoder/vqkd_l.py \
    --config-options ir::configs/exps/llamagen/vqgan.py \
    --load-model-from work_dirs/models/vqkd_open_clip_bigG_14_8192_imagenet_ddp_autocast/checkpoints/iter_250000/model_nd.pth \
    --auto-resume \
    --override .trainer.callbacks[-1].load_state_dict:dict\(strict=False\)
