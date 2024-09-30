torchrun --nproc_per_node=8 --master_port=5031 -m vq.train exps/llamagen_350m_vqkd_laion_aesthetics configs/exps/llamagen_350m_vqkd_laion_aesthetics.py --autocast
