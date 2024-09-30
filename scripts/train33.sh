torchrun --nproc_per_node=8 --master_port=5033 -m vq.train exps/llamagen_350m_vqkd_satin configs/exps/llamagen_350m_vqkd_satin.py --autocast
