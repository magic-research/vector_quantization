bash tools/torchrun.sh -m vq.train exps_decoder_vqkd_convnext_8192_imagenet_ddp_3 configs/exps/decoder_vqkd_convnext_8192_imagenet_ddp_cd.py --load-model-from work_dirs/vqkd/convnext_8192_imagenet_ddp_fp32/checkpoints/iter_250000/model.nd.pth --override .trainer.dataloader.batch_size:6
