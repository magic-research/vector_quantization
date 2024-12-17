# VQ-KD

[[arXiv](https://arxiv.org/abs/2208.06366)] [[GitHub](https://github.com/microsoft/unilm/tree/master/beit2)]

Training:

```bash
auto_torchrun -m vq.train vqkd/clip_8192_imagenet_ddp configs/vqkd/clip_8192_imagenet_ddp.py
```
