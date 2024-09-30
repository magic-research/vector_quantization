set -e

project_root=$(dirname $(realpath $0))

curl https://raw.githubusercontent.com/LutingWang/todd/main/bin/pipenv_install | bash -s -- 3.11.10

pipenv run pip install /data/wlt/wheels/torch-2.4.1+cu121-cp311-cp311-linux_x86_64.whl
pipenv run pip install -i https://download.pytorch.org/whl/cu121 torchvision==0.19.1+cu121

# pipenv run pip install git+https://github.com/LutingWang/CLIP.git
# pipenv run pip install torch==2.4.0 "git+https://github.com/LutingWang/EVA.git#egg=eva_clip&subdirectory=EVA-CLIP/rei"

pipenv run pip install \
    accelerate \
    "protobuf<=3.20.1" \
    scikit-image \
    "torch==2.4.1" \
    torch_fidelity \
    "transformers==4.35.2"
    # open_clip_torch \

pipenv run pip install openmim
pipenv run mim install mmcv
pipenv run pip install git+https://github.com/lvis-dataset/lvis-api.git@lvis_challenge_2021
make install_todd

mkdir -p ${project_root}/pretrained/torch
ln -s ~/.cache/torch/hub/checkpoints ${project_root}/pretrained/torch
