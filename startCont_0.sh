#!/bin/bash

sudo docker build --no-cache -t cope_gpu .
thispid=$(sudo docker run --gpus all --network=host --name=cope_gpu -t -d -v /home/stefan:/home cope_gpu)

#sudo nvidia-docker exec -it $thispid bash

#sudo nvidia-docker container kill $thispid
#sudo nvidia-docker container rm $thispid


