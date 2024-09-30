torchrun --nproc_per_node=8 --master_port=5021 -m vq.train exps/llamagen_350m_vqgan_laion_aesthetics configs/exps/llamagen_350m_vqgan_laion_aesthetics.py --autocast
