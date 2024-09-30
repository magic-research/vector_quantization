torchrun --nproc_per_node=8 --master_port=5006 -m vq.train vqkd/clip_8192_laion_aesthetics_ddp configs/vqkd/clip_8192_laion_aesthetics_ddp.py
