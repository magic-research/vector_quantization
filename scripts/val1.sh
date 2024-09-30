bash tools/torchrun.sh -m vq.val \
    decoder/vqkd_l/vqkd_open_clip_bigG_14_8192_imagenet_ddp_autocast/llamagen_vqgan \
    configs/decoder/vqkd_l.py \
    --config-options ir::configs/exps/llamagen/vqgan.py \
    --load-model-from work_dirs/models/vqkd_open_clip_bigG_14_8192_imagenet_ddp_autocast/checkpoints/iter_250000/model_nd.pth \
    --load-from iter_{11..15}0000 --override .validator.dataloader.batch_size:12
