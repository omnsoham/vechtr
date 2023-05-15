import cv2
import numpy as np
from gpiozero import Device, PWMOutputDevice, Button
from time import sleep
import threading
import os
import logging

#init log
logging.basicConfig(filename='/home/pi/project/logs/vechtor.log',format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

#This script detects following objects
vClassNames = ['person','bus','chair']
classCount = len(vClassNames)

#Counter for button clicks
buttonClickCount=0
button = Button(17)

#Video init 
thres = 0.45 # Threshold to detect object
nms_threshold = 0.2
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

#Init coco class names
classNames = []
classFile = r'/home/pi/project/vechtor/v2/coco.names.txt'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    logger.debug(classNames)

#Coco setup     
configPath = r'/home/pi/project/vechtor/v2/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = r'/home/pi/project/vechtor/v2/frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#Initilize motors
motorLeft = PWMOutputDevice(23)
motorRight = PWMOutputDevice(24)
motorPower= 0.2
motorStopPower = 0.0
sleepTimer=2

#Function to draw rectangle around the image
def drawRectangle(img,box,classId,confidence):
    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
    cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30),
        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    return

#Thread will call following fuction to count button clicks
def registerButtonClicked():
    while True:
        logger.info('Button Clicked Start ---------------------------')
        global buttonClickCount
        button.wait_for_press()           
        buttonClickCount = (buttonClickCount + 1)% classCount
        logger.info('Button Clicked count set to detect :%s' ,vClassNames[buttonClickCount])
        logger.info('Button Clicked End ---------------------------')
        sleep(sleepTimer)
    return

#Button Click thread init
logger.info('Starting Button thread *******************************')
buttonThread = threading.Thread(target=registerButtonClicked, name='ButtonClick')
buttonThread.start()

lastCount=-1
soundCmd="mpg123 /home/pi/project/vechtor/v2/mp3/"
logger.info('-------------- Cap Initilization complete --------------')
os.system(soundCmd +'welcome.mp3')

#main thread
while True:
    success,img = cam.read()
    logger.debug('Reading new image *******************************')
    #Uncomment below to see the webcam feed
    #cv2.imshow('Imagetest',img)
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    #print(classIds,bbox)
    
    #Identify if user clicked button and play object name                                                                 ])
    if lastCount != buttonClickCount:
        mp3Name = vClassNames[buttonClickCount] + '.mp3'
        logger.info('-------------playing mp3 %s' ,soundCmd+mp3Name)        
        os.system(soundCmd + mp3Name)
        lastCount = buttonClickCount
    
    motorLeft.value =0.0
    motorRight.value =0.0
    
    if len(classIds) == 0:
        logger.info('Did not detect any object *******************************')
    else:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):            
            if classId < len(classIds):
                motorLeft.value =0.0
                motorRight.value =0.0                
                logger.info('Found class %s in camera',classNames[classId-1])                
                logger.debug('ButtonClickCount=%d ', buttonClickCount)
                logger.debug('Should detect %s', vClassNames[buttonClickCount])
                
                #If person found then activate the motor based on if person is in left or right
                if ( (buttonClickCount == 1) or (buttonClickCount == 2) or (buttonClickCount ==0)):
                    if classNames[classId -1]  == vClassNames[buttonClickCount]:                        
                        logger.info(' %s detected at x cordinate %f', vClassNames[buttonClickCount] , box[0])
                        #Open Cv box prints (startX,startY,endX,endY)
                        midX = (box[0]+box[2]) / 2
                        
                        drawRectangle(img,box,classId,confidence)
                        #Uncomment below to see the rectangle drawn
                        #cv2.imshow("Output",img)
                        if midX < 120.0:
                            logger.info(' %s detected at left ', vClassNames[buttonClickCount])
                            motorLeft.value =motorPower
                            motorRight.value=motorStopPower
                            #os.system(soundCmd + vClassNames[buttonClickCount] + '_left.mp3')
                            buttonClickCount
                            sleep(sleepTimer)
                            motorLeft.value =motorStopPower
                        elif midX > 210.0:
                            logger.info(' %s detected at right ', vClassNames[buttonClickCount])                            
                            motorRight.value=motorPower
                            motorLeft.value =motorStopPower
                            #os.system(soundCmd + vClassNames[buttonClickCount] + '_right.mp3')
                            sleep(sleepTimer)
                            motorRight.value =motorStopPower
                        else:
                            logger.info(' %s detected at center ', vClassNames[buttonClickCount])                            
                            motorLeft.value =motorPower
                            motorRight.value =motorPower
                            #os.system(soundCmd + vClassNames[buttonClickCount] + '_center.mp3')
                            sleep(sleepTimer)
                            motorRight.value =motorStopPower
                            motorLeft.value =motorStopPower
                    else:
                        logger.info(' %s was not detected  ', vClassNames[buttonClickCount])                            
                                
        k=cv2.waitKey(1)

    if k != -1:
        break;

buttonThread.join()
cam.release()
logger.info('-----------------------End-------------------')    
cv2.destroyAllWindows()
