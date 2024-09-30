bash tools/torchrun.sh -m vq.train exps/gpt2_medium_vqkd_clip_nokmeans configs/exps/gpt2_medium_vqkd_clip_nokmeans.py --override .trainer.dataloader.batch_size:6
