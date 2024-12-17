# FSQ

[[arXiv](https://arxiv.org/abs/2309.15505)] [[GitHub](https://github.com/google-research/google-research/tree/master/fsq)]

Training:

```bash
auto_torchrun -m vq.train fsq/8000_imagenet_ddp configs/fsq/8000_imagenet_ddp.py
auto_torchrun -m vq.train fsq/64000_imagenet_ddp configs/fsq/64000_imagenet_ddp.py
```
