#!/bin/bash

set -euxo pipefail

docker build . -t colab

dataroot="/mnt/data"

docker stop colab || true && docker rm colab || true

docker run --gpus all --shm-size=1g --rm --name colab --ulimit memlock=-1 --ulimit stack=67108864 -d --ipc=host --ip 0.0.0.0 -p 9999:8888 -v /mnt/tmp:/mnt/tmp -v $(pwd):/workspace colab
