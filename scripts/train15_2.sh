bash tools/torchrun.sh -m vq.train exps_decoder_vqkd_convnext_8192_imagenet_ddp_2 configs/exps/decoder_vqkd_convnext_8192_imagenet_ddp_llamagen_cd.py --load-model-from work_dirs/vqkd/convnext_8192_imagenet_ddp_fp32/checkpoints/iter_250000/model.nd.pth
