#!/usr/bin/env zsh

run \
    --gpu ${GPU} \
    --memory ${MEMORY} \
    --cpu ${CPU_} \
    -- tools/debugpy.sh \
        $(hostname -i | awk '{print $1}') \
        $@
