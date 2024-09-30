_base_ = [
    'num_embeddings.py',
    'embedding_dim.py',
    'distance.py',
]

type_ = 'torch_nn_modules_sparse_Embedding'  # pylint: disable=invalid-name
runner = dict(model=dict(quantizer=dict(embedding=dict(type=type_))))

_export_ = dict(trainer=runner, validator=runner)
