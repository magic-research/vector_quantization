# VQGAN

[[arXiv](https://arxiv.org/abs/2012.09841)] [[GitHub](https://github.com/CompVis/taming-transformers)]

Training:

```bash
auto_torchrun -m vq.train vqgan/8192_dd2_aglwg075_imagenet_ddp configs/vqgan/8192_dd2_aglwg075_imagenet_ddp.py
```

<!--

TODO: Add more training commands

```bash
auto_torchrun -m vq.train vqgan/1024_imagenet_ddp configs/vqgan/1024_imagenet_ddp.py
auto_torchrun -m vq.train vqgan/8192_imagenet_ddp configs/vqgan/8192_imagenet_ddp.py
auto_torchrun -m vq.train vqgan/8192_laion_aesthetics_ddp configs/vqgan/8192_laion_aesthetics_ddp.py
auto_torchrun -m vq.train vqgan/8192_sa_med2d_20m_ddp configs/vqgan/8192_sa_med2d_20m_ddp.py
auto_torchrun -m vq.train vqgan/8192_satin_ddp configs/vqgan/8192_satin_ddp.py
auto_torchrun -m vq.train vqgan/8192_stylegan2_imagenet_ddp configs/vqgan/8192_stylegan2_imagenet_ddp.py
```

-->
