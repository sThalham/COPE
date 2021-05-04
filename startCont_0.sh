#!/bin/bash

sudo docker build --no-cache -t pyrapose_gpu_1 .
thispid=$(sudo docker run --gpus device=1 --network=host --name=pyrapose_gpu_1 -t -d -v ~/data/train_data/linemod_PBR_BOP:/PyraPose/data -v ~/data/train_data/linemod_RGBD_val:/PyraPose/val -v ~/data/Meshes:/PyraPose/Meshes pyrapose_gpu_1)

#sudo nvidia-docker exec -it $thispid bash

#sudo nvidia-docker container kill $thispid
#sudo nvidia-docker container rm $thispid


