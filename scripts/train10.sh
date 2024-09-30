HF_DATASETS_OFFLINE=1 torchrun --nproc_per_node=8 --master_port=5010 -m vq.train vqkd/clip_8192_satin_ddp configs/vqkd/clip_8192_satin_ddp.py
