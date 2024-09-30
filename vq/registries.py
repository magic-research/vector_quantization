__all__ = [
    'VQRegistry',
    'VQDatasetRegistry',
    'VQModelRegistry',
    'VQRunnerRegistry',
    'VQTaskRegistry',
]

import todd
from todd.registries import (
    DatasetRegistry,
    ModelRegistry,
    RunnerRegistry,
    TaskRegistry,
)


class VQRegistry(todd.Registry):
    pass


class VQDatasetRegistry(VQRegistry, DatasetRegistry):
    pass


class VQModelRegistry(VQRegistry, ModelRegistry):
    pass


class VQRunnerRegistry(VQRegistry, RunnerRegistry):
    pass


class VQTaskRegistry(VQRegistry, TaskRegistry):
    pass
