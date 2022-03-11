#!/bin/bash

sudo docker build --no-cache -t cope .
thispid=$(sudo docker run --network=host --name=cope -t -d cope)

sudo nvidia-docker exec -it $thispid bash



