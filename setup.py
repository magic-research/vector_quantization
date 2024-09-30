import pathlib

from setuptools import setup


def todd_ai() -> str:
    path = pathlib.Path(__file__)
    path = path.parent / '.todd_version'
    todd_version = path.read_text().strip()
    return (
        f'todd_ai[optional,dev,lint,doc,test] @ git+https://github.com/LutingWang/todd.git@{todd_version}'
    )


def symlink_configs() -> None:
    path = pathlib.Path(__file__)
    path = path.parent / 'vq' / 'configs'
    if path.exists():
        return
    path.symlink_to('../configs')


def symlink_todd_version() -> None:
    path = pathlib.Path(__file__)
    path = path.parent / 'vq' / '.todd_version'
    if path.exists():
        return
    path.symlink_to('../.todd_version')


symlink_configs()
symlink_todd_version()
setup(
    install_requires=[
        'accelerate',
        'debugpy',
        # 'open_clip_torch',
        'protobuf<=3.20.1',
        # 'pycocotools',
        'scikit-image',
        'torch_fidelity',
        'transformers==4.35.2',
        todd_ai(),
    ],
)
