#!/bin/bash

set -eu -o pipefail

INPUT="--repeatInput 32k data/lorem_small.txt"

for ALGORITHM in serial naive atomic atomic_shm overlap final
do
    srun -p volta-hp --nodelist=volta01 -c 32 --gpus=1 ./histogram --algorithm ${ALGORITHM} --pinned ${INPUT}
done