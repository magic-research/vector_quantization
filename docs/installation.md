# Installation

This repository requires **Python 3.11** or higher. While PyTorch 2.4 is recommended, earlier versions may also be compatible. Ensure that you also install a matching version of TorchVision.

Install the primary dependencies using the following commands:

```bash
GIT_LFS_SKIP_SMUDGE=1 pip install -e .
pip install git+https://github.com/LutingWang/CLIP.git # TODO: remove this dependency
```

For experiments involving **StyleGAN**, install MMCV using:

```bash
mim install mmcv
```

If you prefer to set up the environment manually, refer to the script provided in `setup.sh`.
