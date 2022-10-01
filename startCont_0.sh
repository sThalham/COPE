#!/bin/bash

sudo docker build --no-cache -t cope_gpu .
thispid=$(sudo docker run --gpus 0 --network=host --name=cope_gpu -t -d -v /home/stefan/data/train_data:/cope/data cope_gpu)

#sudo nvidia-docker exec -it $thispid bash

#sudo nvidia-docker container kill $thispid
#sudo nvidia-docker container rm $thispid


