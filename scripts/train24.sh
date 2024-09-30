torchrun --nproc_per_node=8 --master_port=5024 -m vq.train exps/llamagen_350m_vqgan_satin configs/exps/llamagen_350m_vqgan_satin.py --autocast
