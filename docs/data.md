# Data

Our experiments primarily use the **ImageNet-1k** dataset. Please organize the dataset in the following structure:

```text
data/imagenet/
├── annotations
│   ├── train.json
|   └── val.json
├── train
│   ├── n1440764
│   │   ├── 18.JPEG
│   │   └── ...
|   └── ...
├── val
│   ├── n1440764
│   │   ├── 293.JPEG
│   │   └── ...
|   └── ...
└── synsets.json
```

Both ``train.json`` and ``val.json`` have the following structure:

```text
[{"image":"12925.JPEG","synset_id":449},...]
```

``synsets.json`` maps synset IDs to their corresponding information:

```text
{"1":{"WNID":"n02119789","words":"kit fox, Vulpes macrotis",...},...}
```

For detailed instructions on downloading and preparing the dataset, refer to <https://toddai.readthedocs.io/en/latest/data/imagenet.html#imagenet-1k>
