# Pretrained Models

The following checkpoints are required to run the code:

```bash
python tools/prepare_checkpoints.py pytorch_fid

python tools/prepare_checkpoints.py lpips
python tools/convert_checkpoints.py lpips pretrained/lpips/vgg.pth

python tools/prepare_checkpoints.py torchvision --weights .VGG16_Weights.DEFAULT
```

The following checkpoints are the IU models that are used in the paper:

```bash

python tools/prepare_checkpoints.py clip --weights ViT-B/16
python tools/prepare_checkpoints.py dino
python tools/prepare_checkpoints.py torchvision --weights .ViT_B_16_Weights.DEFAULT
python tools/prepare_checkpoints.py mae
```

The following checkpoints are used to initialize the AR proposal networks:

```bash
python tools/prepare_checkpoints.py huggingface
```

VQGAN and VQ-KD checkpoints from their original repo can be loaded by our code, after conversion:

```bash
python tools/prepare_checkpoints.py taming_transformers
python tools/convert_checkpoints.py taming_transformers pretrained/taming-transformers/vqgan_imagenet_f16_1024.pth --check configs/vqgan/1024_imagenet_ddp.py
python tools/convert_checkpoints.py taming_transformers pretrained/taming-transformers/vqgan_imagenet_f16_16384.pth --check configs/vqgan/16384_dd2_aglwg075_imagenet_ddp.py

python tools/prepare_checkpoints.py beitv2
python tools/convert_checkpoints.py beitv2 pretrained/beitv2/vqkd_encoder_base_decoder_1x768x12_clip.pth
python tools/convert_checkpoints.py beitv2 pretrained/beitv2/vqkd_encoder_base_decoder_1x768x12_clip.pth --check configs/vqkd/clip_8192_imagenet_ddp.py --suffix .converted.with_decoder --with-decoder
python tools/convert_checkpoints.py beitv2 pretrained/beitv2/vqkd_encoder_base_decoder_1x768x12_dino.pth
python tools/convert_checkpoints.py beitv2 pretrained/beitv2/vqkd_encoder_base_decoder_1x768x12_dino.pth --check configs/vqkd/dino_8192_imagenet_ddp.py --suffix .converted.with_decoder --with-decoder
```

After generating the FID cache, you can run the following command to validate the pretrained models:

```bash
auto_torchrun -m vq.test vqgan/16384_dd2_aglwg075_imagenet_ddp configs/vqgan/16384_dd2_aglwg075_imagenet_ddp.py --load-model-from pretrained/taming-transformers/vqgan_imagenet_f16_16384.pth.converted --visual pred_image
# {'lpips_loss': 0.28323277831077576, 'l1_image_loss': 0.06811775267124176, 'mse_image_loss': 0.013179616071283817, 'psnr': 19.970359802246094, 'ssim': 0.5023356676101685, 'fid': 4.980832106065748, 'codebook_usage': 0.059326171875, 'codebook_ppl': 6.812368392944336}

auto_torchrun -m vq.test vqkd/clip_8192_imagenet_ddp configs/vqkd/clip_8192_imagenet_ddp.py --load-model-from pretrained/beitv2/vqkd_encoder_base_decoder_1x768x12_clip.pth.converted.with_decoder
# {'cosine_embedding_r_loss': 0.16431047022342682, 'codebook_usage': 1.0, 'codebook_ppl': 8.94822883605957}
```
