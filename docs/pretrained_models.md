# Pretrained Models

Download the required checkpoints with the following commands:

```bash
python tools/prepare_checkpoints.py pytorch_fid
python tools/prepare_checkpoints.py lpips
python tools/prepare_checkpoints.py inception
python tools/prepare_checkpoints.py torchvision
python tools/prepare_checkpoints.py beitv2
```

Some checkpoints can be loaded by our code, after convertion:

```bash
python tools/convert_checkpoints.py beitv2 pretrained/beitv2/vqkd_encoder_base_decoder_1x768x12_clip.pth --options with_decoder:False
python tools/convert_checkpoints.py beitv2 pretrained/beitv2/vqkd_encoder_base_decoder_1x768x12_clip.pth --options with_decoder:True --suffix .converted.with_decoder
```
