#! /bin/bash
source /home/pi/project/env/bin/activate
cd /home/pi/project/vechtor/v2/
echo $HOME
export HOME="/home/pi"

#start vechtor
sudo rm -rf /home/pi/project/logs/vechtor.log
echo "Vechtor process starting.."	
python3 main-robo-haptic.py &
pid="$!"
echo "Vechtor process started $pid"
