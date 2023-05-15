#! /bin/bash
source /home/pi/project/env/bin/activate
cd /home/pi/project/vechtor/v2/
python3 main-haptic.py & 
pid="$!"
echo "Vechtor process started $pid"	



