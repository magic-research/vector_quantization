torchrun --nproc_per_node=8 --master_port=5009 -m vq.train vqgan/8192_sa_med2d_ddp configs/vqgan/8192_sa_med2d_ddp.py
