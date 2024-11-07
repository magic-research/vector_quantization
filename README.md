# Vector Quantization

## Preparation

### Install Requirements:
python 的版本要求为 3.11，以下是一些安装包的需求：
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118 
pip install "numpy<2.0" scipy scikit-image mim transformers==4.38.2 git+https://github.com/lvis-dataset/lvis-api.git@lvis_challenge_2021
mim install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html
make install_todd
```

### Others
下载一些模型需要的 checkpoints
```bash
python tools/prepare_checkpoints.py pytorch_fid 
python tools/prepare_checkpoints.py lpips
python tools/prepare_checkpoints.py inception
python tools/prepare_checkpoints.py torchvision
python tools/prepare_checkpoints.py beitv2
```

convert for checkpoints
```bash
python tools/convert_checkpoints.py beitv2 pretrained/beitv2/vqkd_encoder_base_decoder_1x768x12_clip.pth --options with_decoder:False
python tools/convert_checkpoints.py beitv2 pretrained/beitv2/vqkd_encoder_base_decoder_1x768x12_clip.pth --options with_decoder:True --suffix .converted.with_decoder
```

生成一些 FID 的 cache (把 reference 图片集合提取到的特征提前存储下来，不需要每次重新提取特征与生成图像进行对比)
在生成训练集的 cache 时，可以调整 batch-size 的大小，batch-szie 越大生成的速度越快，可以根据当前 GPU 的显存来选择合适的 batch-size 大小
```bash
mkdir -p pretrained/fid
auto_torchrun -m vq.fid imagenet
auto_torchrun -m vq.fid imagenet --train --override .dataloader.batch_size:1024
```

## Tokenizers

将输入图像转换为离散的代码，对输入图像进行 encoder 得到特征图 x，特征图 x 经过 quantizer Q 得到对应 code book C 的下标序列 z。\
可以在 configs 文件夹下选择不同的模型来进行训练，例如 `configs/vqgan/*`，`configs/fsq/*` ，保存的文件命名要与训练的模型保持一致。

### VQGAN ([arXiv](https://arxiv.org/abs/2012.09841), [GitHub](https://github.com/CompVis/taming-transformers))

**Training**
在路径 `configs/vqgan/*` 下可以选择多种 VQGAN 的具体模型来进行训练
```bash
auto_torchrun -m vq.train vqgan/8192_dd2_aglwg075_imagenet_ddp configs/vqgan/8192_dd2_aglwg075_imagenet_ddp.py
auto_torchrun -m vq.train vqgan/1024_imagenet_ddp configs/vqgan/1024_imagenet_ddp.py
auto_torchrun -m vq.train vqgan/8192_imagenet_ddp configs/vqgan/8192_imagenet_ddp.py
auto_torchrun -m vq.train vqgan/8192_laion_aesthetics_ddp configs/vqgan/8192_laion_aesthetics_ddp.py
auto_torchrun -m vq.train vqgan/8192_sa_med2d_20m_ddp configs/vqgan/8192_sa_med2d_20m_ddp.py
auto_torchrun -m vq.train vqgan/8192_satin_ddp configs/vqgan/8192_satin_ddp.py
auto_torchrun -m vq.train vqgan/8192_stylegan2_imagenet_ddp configs/vqgan/8192_stylegan2_imagenet_ddp.py
```

**Validation**
```bash
auto_torchrun -m vq.val vqgan/8192_dd2_aglwg075_imagenet_ddp configs/vqgan/8192_dd2_aglwg075_imagenet_ddp.py [--load-from iter_{15..26}0000]
auto_torchrun -m vq.val vqgan/1024_imagenet_ddp configs/vqgan/1024_imagenet_ddp.py [--load-from iter_{15..26}0000]
auto_torchrun -m vq.val vqgan/8192_imagenet_ddp configs/vqgan/8192_imagenet_ddp.py [--load-from iter_{15..26}0000]
auto_torchrun -m vq.val vqgan/8192_laion_aesthetics_ddp configs/vqgan/8192_laion_aesthetics_ddp.py [--load-from iter_{15..26}0000]
auto_torchrun -m vq.val vqgan/8192_sa_med2d_20m_ddp configs/vqgan/8192_sa_med2d_20m_ddp.py [--load-from iter_{15..26}0000]
auto_torchrun -m vq.val vqgan/8192_satin_ddp configs/vqgan/8192_satin_ddp.py [--load-from iter_{15..26}0000]
auto_torchrun -m vq.val vqgan/8192_stylegan2_imagenet_ddp configs/vqgan/8192_stylegan2_imagenet_ddp.py [--load-from iter_{15..26}0000]
```



### CVQ-VAE ([arXiv](https://arxiv.org/abs/2307.15139), [GitHub](https://github.com/lyndonzheng/CVQ-VAE))

**Training**
在路径 `configs/cvqvae/*` 下可以选择 CVQ-VAE 的具体模型来进行训练
```bash
auto_torchrun -m vq.train cvqvae/8192_dd2_aglwg075_imagenet_ddp configs/cvqvae/8192_dd2_aglwg075_imagenet_ddp.py
```

**Validation**
```bash
auto_torchrun -m vq.val cvqvae/8192_dd2_aglwg075_imagenet_ddp configs/cvqvae/8192_dd2_aglwg075_imagenet_ddp.py [--load-from iter_{15..26}0000]
```


### FSQ ([arXiv](https://arxiv.org/abs/2309.15505), [GitHub](https://github.com/google-research/google-research/tree/master/fsq))

**Training**
在路径 `configs/fsq/*` 下可以选择 FSQ 的具体模型来进行训练
```bash
auto_torchrun -m vq.train fsq/8000_imagenet_ddp configs/fsq/8000_imagenet_ddp.py
auto_torchrun -m vq.train fsq/64000_imagenet_ddp configs/fsq/64000_imagenet_ddp.py
```

**Validation**
```bash
auto_torchrun -m vq.val fsq/8000_imagenet_ddp configs/fsq/8000_imagenet_ddp.py [--load-from iter_{15..26}0000]
auto_torchrun -m vq.val fsq/64000_imagenet_ddp configs/fsq/64000_imagenet_ddp.py [--load-from iter_{15..26}0000]
```


### VQ-KD ([arXiv](https://arxiv.org/abs/2208.06366), [GitHub]())

**Training**
在路径 `configs/vqkd/*` 下可以选择 VQ-KD 的具体模型来进行训练
```bash
auto_torchrun -m vq.train vqkd/clip_8192_imagenet_ddp configs/vqkd/clip_8192_imagenet_ddp.py
```

由于其参考的 BEiT v2 模型 decoder 并不能很好地支持这个任务，需要对 decoder 进行额外的训练
`configs/decoder/xxxx.py` 选择需要训练的 decoder 对应的模型 \
`--config-options` 传入 image tokenizer 模型路径 \
`--load-model-from` 传入已经训练好的 tokenizer 参数路径 \
`--override` 可以选择传入与否

```bash 
auto_torchrun -m vq.train \
    decoder/vqgan/vqkd_clip_8192_imagenet_ddp/vqgan_8192_dd2_aglwg075_imagenet_ddp \
    configs/decoder/vqgan.py \
    --config-options it_config::configs/vqkd/clip_8192_imagenet_ddp.py \
    --load-model-from work_dirs/dry_run/vqkd/clip_8192_imagenet_ddp/checkpoints/iter_7/model.pth \
    [--override .trainer.callbacks[-1].load_state_dict:dict\(strict=False\)]
auto_torchrun -m vq.train \
    decoder/llamagen/vqkd_clip_8192_imagenet_ddp/llamagen_8192_dd2_aglwg075_imagenet_ddp \
    configs/decoder/llamagen.py \
    --config-options it_config::configs/vqkd/clip_8192_imagenet_ddp.py \
    --load-model-from work_dirs/dry_run/vqkd/clip_8192_imagenet_ddp/checkpoints/iter_7/model.pth \
    [--override .trainer.callbacks[-1].load_state_dict:dict\(strict=False\)]
```

**Validation**
```bash
auto_torchrun -m vq.val vqkd/clip_8192_imagenet_ddp configs/vqkd/clip_8192_imagenet_ddp.py [--load-from iter_{15..26}0000]
```

```bash 
auto_torchrun -m vq.val \
    decoder/vqgan/vqkd_clip_8192_imagenet_ddp/vqgan_8192_dd2_aglwg075_imagenet_ddp \
    configs/decoder/vqgan.py \
    --config-options it_config::configs/vqkd/clip_8192_imagenet_ddp.py \
    --load-model-from work_dirs/dry_run/vqkd/clip_8192_imagenet_ddp/checkpoints/iter_7/model.pth \
    [--load-from iter_{15..26}0000] \
    [--override .trainer.callbacks[-1].load_state_dict:dict\(strict=False\)]
auto_torchrun -m vq.val \
    decoder/llamagen/vqkd_clip_8192_imagenet_ddp/llamagen_8192_dd2_aglwg075_imagenet_ddp \
    configs/decoder/llamagen.py \
    --config-options it_config::configs/vqkd/clip_8192_imagenet_ddp.py \
    --load-model-from work_dirs/dry_run/vqkd/clip_8192_imagenet_ddp/checkpoints/iter_7/model.pth \
    [--load-from iter_{15..26}0000] \
    [--override .trainer.callbacks[-1].load_state_dict:dict\(strict=False\)]
```


### Ours - Cluster

**Training**
在路径 `configs/cluster/*` 下可以选择 Cluster 的具体模型来进行训练
```bash
auto_torchrun -m vq.train cluster/clip_8192_imagenet_ddp configs/cluster/clip_8192_imagenet_ddp.py
```

由于我们的 Cluster 模型根本没有 decoder，所以对别的模型的 decoder 进行额外的训练然后进行嵌入
`configs/decoder/xxxx.py` 选择需要训练的 decoder 对应的模型 \
`--config-options` 传入 image tokenizer 模型路径 \
`--load-model-from` 传入已经训练好的 tokenizer 参数路径 \
`--override` 可以选择传入与否

```bash 
auto_torchrun -m vq.train \
    decoder/vqgan/cluster_clip_8192_imagenet_ddp/vqgan_8192_dd2_aglwg075_imagenet_ddp \
    configs/decoder/vqgan.py \
    --config-options it_config::configs/cluster/clip_8192_imagenet_ddp.py \
    --load-model-from work_dirs/dry_run/cluster/clip_8192_imagenet_ddp/checkpoints/iter_7/model.pth \
    [--override .trainer.callbacks[-1].load_state_dict:dict\(strict=False\)]
auto_torchrun -m vq.train \
    decoder/llamagen/cluster_clip_8192_imagenet_ddp/llamagen_8192_dd2_aglwg075_imagenet_ddp \
    configs/decoder/llamagen.py \
    --config-options it_config::configs/cluster/clip_8192_imagenet_ddp.py \
    --load-model-from work_dirs/dry_run/cluster/clip_8192_imagenet_ddp/checkpoints/iter_7/model.pth \
    [--override .trainer.callbacks[-1].load_state_dict:dict\(strict=False\)]
```

**Validation**
```bash
auto_torchrun -m vq.val auto_torchrun -m vq.train cluster/clip_8192_imagenet_ddp configs/cluster/clip_8192_imagenet_ddp.py [--load-from iter_{15..26}0000]
```

```bash 
auto_torchrun -m vq.val \
    decoder/vqgan/cluster_clip_8192_imagenet_ddp/vqgan_8192_dd2_aglwg075_imagenet_ddp \
    configs/decoder/vqgan.py \
    --config-options it_config::configs/cluster/clip_8192_imagenet_ddp.py \
    --load-model-from work_dirs/dry_run/cluster/clip_8192_imagenet_ddp/checkpoints/iter_7/model.pth \
    [--load-from iter_{15..26}0000] \
    [--override .trainer.callbacks[-1].load_state_dict:dict\(strict=False\)]
auto_torchrun -m vq.val \
    decoder/llamagen/cluster_clip_8192_imagenet_ddp/llamagen_8192_dd2_aglwg075_imagenet_ddp \
    configs/decoder/llamagen.py \
    --config-options it_config::configs/cluster/clip_8192_imagenet_ddp.py \
    --load-model-from work_dirs/dry_run/cluster/clip_8192_imagenet_ddp/checkpoints/iter_7/model.pth \
    [--load-from iter_{15..26}0000] \
    [--override .trainer.callbacks[-1].load_state_dict:dict\(strict=False\)]
```


## Proposal Network
冻结 tokenizer 和 decoder 的所有参数，训练 proposal network 对 z 的分布进行建模 \
`ir_config` 选择需要训练的 proposal network 对应的模型 \
`it_state_dict` 传入已经训练好的 image tokenizer 的参数 \
`ir_state_dict` 传入已经训练好的 image reconstruction 的模型参数

```bash
auto_torchrun -m vq.train \
    ar/c2i_llama_medium_cfg_imagenet_ddp configs/ar/c2i_llama_medium_cfg_imagenet_ddp.py \
    --config-options \
        ir_config::work_dirs/dry_run/decoder/vqgan/vqkd_clip_8192_imagenet_ddp/vqgan_8192_dd2_aglwg075_imagenet_ddp/vqgan.py \
        it_state_dict::work_dirs/dry_run/vqkd/clip_8192_imagenet_ddp/checkpoints/iter_7/model.pth \
        ir_state_dict::work_dirs/dry_run/decoder/vqgan/vqkd_clip_8192_imagenet_ddp/vqgan_8192_dd2_aglwg075_imagenet_ddp/checkpoints/iter_2/model.pth
```


## Inference

```bash
torchrun --nproc_per_node=1 -m vq.inference configs/vqkd_clip_8192_dd3_imagenet_ddp.py data/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG --load-model-from work_dirs/vqkd_8192_imagenet_ddp_autocast/checkpoints/iter_250000/model.pth
```

## FID

Two approaches can be used to calculate FID: `fidelity` or `fid`.

`fidelity` is easy to start with but only supports single-GPU:

```bash
# ${image_root}/
# └── *.png
run -- python tools/fidelity.py ${dataset:-imagenet} ${image_root} --work-dir ${image_root}/..
run -- python tools/fidelity.py ${dataset:-imagenet} ${image_root} --train --work-dir ${image_root}/..
```

`fid` requires generating FID cache first.
`FIDMetric` relies on these cache files to online calculate FID.
Users can also manually calculate FID:

```bash
bash tools/torchrun.sh tools/fid.py imagenet_val ${image_root} --work-dir ${image_root}/..
bash tools/torchrun.sh tools/fid.py imagenet_train ${image_root} --work-dir ${image_root}/..
```

## Tokenize

To save tokens during validation, use `--tokenize`:

```bash
bash tools/torchrun.sh -m vq.test ${name} ${config} --tokenize --load-model-from ${load_model_from}
```

The above script introduces extra computation cost.
To run an tokenization only:

```bash
bash tools/torchrun.sh -m vq.tasks.image_tokenization.tokenize ${name} ${config} --override .validator.type::VQITRunnerRegistry.Tokenizer --load-model-from ${load_model_from}
bash tools/torchrun.sh -m vq.tasks.image_tokenization.tokenize ${name} ${config} --override .validator.type::VQITRunnerRegistry.Tokenizer --load-model-from ${load_model_from} --train
```

LlamaGen tokenizes the training dataset:

```bash
bash tools/torchrun.sh tools/llamagen.py llamagen/vqkd_8192_imagenet_ddp configs/vqkd/8192_imagenet_ddp.py --load-model-from work_dirs/models/vqkd_openclip_bigG_14_8192_imagenet_ddp_autocast/checkpoints/iter_250000/model.pth
```

## Visualize

```bash
python tools/visualize_patches.py configs/vqkd/clip_8192_imagenet_ddp.py work_dirs/vqkd_clip_tokenize_tokenize/tokens
python tools/visualize_patches.py configs/vqgan/8192_dd2_aglwg075_imagenet_ddp.py work_dirs/vqgan_tokenize_tokenize/tokens
```

## Utils

```bash
for file in work_dirs/conditional_image_transformer_min_vqgan_fsq_8192_imagenet_ddp/checkpoints/iter_*; do num=$(echo $file | grep -o '[0-9]\+0000'); if (( $num % 40000 != 0 )); then rm -r $file; fi; done
for file in work_dirs/teacher_conditional_image_transformer_min_vqgan_dino_cvq_vae_8192_imagenet_ddp/checkpoints/iter_*; do num=$(echo $file | grep -o '[0-9]\+0000'); if (( $num % 40000 != 0 )); then rm -r $file; fi; done
for file in work_dirs/teacher_conditional_image_transformer_min_vqgan_mae_cvq_vae_8192_imagenet_ddp/checkpoints/iter_*; do num=$(echo $file | grep -o '[0-9]\+0000'); if (( $num % 40000 != 0 )); then rm -r $file; fi; done
for file in work_dirs/teacher_conditional_image_transformer_min_vqgan_vit_cvq_vae_8192_imagenet_ddp/checkpoints/iter_*; do num=$(echo $file | grep -o '[0-9]\+0000'); if (( $num % 40000 != 0 )); then rm -r $file; fi; done
```

## To Do List

- implement make backup via hard link
- complete pre-commit-config.yaml
- remove open_clip/eva_clip dependencies


## Acknowledgments

特别感谢 VQGAN 模型的开发者和贡献者提供了代码资源，为本研究提供了重要支持。
