#!bash
ssh -fN -L ${PORT}:localhost:${PORT} -p 9000 root@$1
pipenv run python \
    -m debugpy \
    --connect localhost:${PORT} \
    $(pipenv --venv)/bin/torchrun \
        --nnodes ${ARNOLD_WORKER_NUM} \
        --nproc-per-node ${ARNOLD_WORKER_GPU} \
        --node-rank ${ARNOLD_ID} \
        --master-addr ${METIS_WORKER_0_HOST} \
        --master-port ${METIS_WORKER_0_PORT} \
        ${@:2}
