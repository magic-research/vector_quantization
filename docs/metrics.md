# Metrics

We adopt FID as the primary metric for evaluating the quality of generative models. In addition, we also provide the Inception Score (IS) for reference.

## FID

FID computes the statistical distance in the feature space between the generated images and the reference (real) images. The lower the FID, the better the generated images.

### Cache

To accelerate the calculation of FID, we compute the feature statistics of the reference images in advance using the following commands:

```bash
mkdir -p pretrained/fid
auto_torchrun -m vq.fid imagenet
```

By default, the reference images are the validation split of ImageNet 1k. Therefore, the feature statistics are stored in `pretrained/fid/imagenet_val.pth`.

Some papers report the FID metric with the training split as referenece images. For a fair comparison, you can generate the FID cache for the training split as follows:

```bash
auto_torchrun -m vq.fid imagenet --train
```

This produces `pretrained/fid/imagenet_train.pth`.

Since the training split is large, it is recommended to use a larger batch size to speed up the process:

```bash
auto_torchrun -m vq.fid imagenet --train --override .dataloader.batch_size:1024
```

> The FID cache is used in multiple scripts including `vq.val`. Make sure you generated the cache before running these scripts.

### Offline Calculation

After generating the cache, it is recommanded to calculate FID online, using `vq.val` or `vq.test`. However, you can also calculate FID offline using the following command:

```bash
bash tools/torchrun.sh tools/fid.py ${cache:-imagenet_val} ${image_root}
```

By offline, we mean that the generated images are stored under `${image_root}`, like this:

```text
${image_root}/
└── *.png
```

`imagenet_val` is the name of the cache file and can be replaced with other names like `imagenet_train`.

`tools/fid.py` also supports writing the FID metric to a file, if the `--work-dir` argument is provided. It is recommended to use `${image_root}/..`:

```bash
bash tools/torchrun.sh tools/fid.py ${cache:-imagenet_val} ${image_root} --work-dir ${image_root}/..
```

The above command appends to a file named `fid.txt` in `${image_root}/..`, so the file content will not be overriden. This is useful if you want to compare the FID metrics against different reference images.

## IS

We rely on `torch_fidelity` to calculate the IS metric. The following commands can be used:

```bash
python tools/fidelity.py ${dataset:-imagenet} ${image_root}
```

As with FID, `${image_root}` should contain the generated images and `--work-dir` is supported:

```bash
python tools/fidelity.py ${dataset:-imagenet} ${image_root} --work-dir ${image_root}/..
```

While the calculation of IS does not involve reference images, the `${dataset}` argument is required. The reason is that `torch_fidelity` supports evaluting FID and IS in the same time, and the `${dataset}` argument helps `torch_fidelity` to calculate the FID score. You can compare the FID scores calculated by `torch_fidelity` and `tools/fid.py` to verify the correctness of our implementation.

`tools/fidelity.py` reads `configs/datasets/${dataset}.py` to load the reference images. The validation split is used by default. If you want to use the training split, you can specify the `--train` argument:

```bash
python tools/fidelity.py ${dataset:-imagenet} ${image_root} --train
```

> `torch_fidelity` is a third-party library used by many projects. It manages the FID cache internally, making it easy to start with. However, it only supports single-GPU. If you want to use multiple GPUs, you can use `tools/fid.py` instead.
