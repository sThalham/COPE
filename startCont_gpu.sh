#!/bin/bash

sudo docker build --no-cache -t cope_gpu .
thispid=$(sudo docker run --gpus 0 --network=host --name=cope_gpu -t -d cope_gpu)

sudo nvidia-docker exec -it $thispid bash



