torchrun --nproc_per_node=8 --master_port=5007 -m vq.train vqgan/8192_laion_aesthetics_ddp configs/vqgan/8192_laion_aesthetics_ddp.py
