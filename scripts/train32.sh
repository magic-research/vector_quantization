torchrun --nproc_per_node=8 --master_port=5032 -m vq.train exps/llamagen_350m_vqkd_sa_med2d configs/exps/llamagen_350m_vqkd_sa_med2d.py --autocast
