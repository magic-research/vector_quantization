HF_DATASETS_OFFLINE=1 torchrun --nproc_per_node=8 --master_port=5011 -m vq.train vqgan/8192_satin_ddp configs/vqgan/8192_satin_ddp.py
