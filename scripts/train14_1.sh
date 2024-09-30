bash tools/torchrun.sh -m vq.train exps_decoder_vqkd_evaclip_8192_imagenet_ddp_1 configs/exps/decoder_vqkd_evaclip_8192_imagenet_ddp.py --load-model-from work_dirs/exps_decoder_vqkd_evaclip_8192_imagenet_ddp_llamagen/checkpoints/iter_20000/model.nd.pth work_dirs/vqkd/evaclip_8192_imagenet_ddp_fp32/checkpoints/iter_250000/model.nd.pth --override .trainer.discriminator_start:20_000 .trainer.dataloader.batch_size:6
