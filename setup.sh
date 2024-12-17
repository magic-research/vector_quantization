set -e

project_root=$(dirname $(realpath $0))

curl https://raw.githubusercontent.com/LutingWang/todd/main/bin/pipenv_install | bash -s -- 3.11.10

pipenv run pip install ~/wheels/torch-2.4.1+cu121-cp311-cp311-linux_x86_64.whl
pipenv run pip install -i https://download.pytorch.org/whl/cu121 torchvision==0.19.1+cu121

pipenv run pip install \
    accelerate \
    debugpy \
    "protobuf<=3.20.1" \
    scikit-image \
    torch_fidelity \
    transformers==4.35.2

pipenv run pip install openmim
pipenv run mim install mmcv
pipenv run pip install git+https://github.com/lvis-dataset/lvis-api.git@lvis_challenge_2021
make install_todd

pip install git+https://github.com/LutingWang/CLIP.git # TODO: remove this dependency
