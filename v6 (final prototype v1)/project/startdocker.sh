#! /bin/bash
source /home/pi/project/env/bin/activate
cd /home/pi/project/vechtor/v2/

#start docker 
sudo rm -rf /home/pi/project/logs/docker.log
sudo docker run --net=host roboflow/inference-server:cpu > /home/pi/project/logs/docker.log &
dockerPid="$!"
echo "Docker process started $dockerPid"
