torchrun --nproc_per_node=8 --master_port=5008 -m vq.train vqkd/clip_8192_sa_med2d_ddp configs/vqkd/clip_8192_sa_med2d_ddp.py
