sudo docker build -t cope_ros .
thispid=$(sudo docker run --gpus all --network=host --name=cope_ros -t -d -v /home/stefan:/stefan cope_ros)
