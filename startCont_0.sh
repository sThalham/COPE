#!/bin/bash

sudo docker build --no-cache -t uape_gpu .
thispid=$(sudo docker run --gpus all --network=host --name=uape_gpu -t -d -v ~/data:/PyraPoseAF/data -v ~/data/Meshes:/PyraPose/Meshes uape_gpu)

#sudo nvidia-docker exec -it $thispid bash

#sudo nvidia-docker container kill $thispid
#sudo nvidia-docker container rm $thispid


