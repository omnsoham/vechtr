import cv2
import numpy as np
from gpiozero import Device, PWMOutputDevice, Button
from time import sleep
import threading
import os
import logging
from roboflow import Roboflow

#init log
logging.basicConfig(filename='/home/pi/project/logs/vechtor.log',format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.info("log setup done")

rf=Roboflow(api_key="esXiQyyY9AGa6JY8zL1W")
project = rf.workspace("vechtor").project("vechtor-plkjm")

#This script detects following objects
vClassNames = ['Chair','Door','Bus']
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

#Initilize motors
motorLeft = PWMOutputDevice(23)
motorRight = PWMOutputDevice(24)
motorPower= 0.5
motorStopPower = 0.0
sleepTimer=2

#roboflow variables setup
def initRoboflow():
    #Check if docker container started otherwise wait
    logger.info(" Checking docker log..")
    dlog = open("/home/pi/project/logs/docker.log", "r")
    logger.info(" Docker logfile opened..")
    logLine = dlog.readline()
    while True:
        if "inference-server is ready to receive traffic." in logLine :
            logger.info(" Docker container started..")
            break;
        else:
            logger.debug(" Docker log line %s ",logLine)
            logger.info(" Docker container not started so sleeping ....")
            sleep(10)
            logLine = dlog.readline()
            
    dlog.close()
    logger.info(" Docker container has started so starting vechtor process ..")
    #roboflow setup
    global model    
    model = project.version(1,local="http://localhost:9001/").model

#Function to draw rectangle around the image
def drawRfRectangle(img,spt, ept,className,confidence):
    cv2.rectangle(img,spt, ept, color=(0,255,0),thickness=2)
    cv2.putText(img,className,(spt[0]+10,spt[1]+30),
        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.putText(img,str(round(confidence*100,2)),(spt[0]+200,spt[1]+30),
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
#os.system(soundCmd +'welcome.mp3')
welcomePlayed=False

#main thread
while True:
    if welcomePlayed != True :
        initRoboflow()        
        welcomePlayed=True
        os.system(soundCmd +'welcome.mp3')
        logger.info('-------------- Cap Initilization complete --------------')
        
    success,img = cam.read()
    motorLeft.value =0.0
    motorRight.value =0.0
    #Identify if user clicked button and play object name                                                                 ])
    if lastCount != buttonClickCount:
        mp3Name = vClassNames[buttonClickCount] + '.mp3'
        logger.info('-------------playing mp3 %s' ,soundCmd+mp3Name)        
        os.system(soundCmd + mp3Name)
        lastCount = buttonClickCount
    
    logger.debug('Should detect %s', vClassNames[buttonClickCount])
    
    if success==False :
        logger.info('Did not detect any object *******************************')
    else:
        logger.debug('Reading new image *******************************')
        height, width,channels = img.shape
        
        result = model.predict(img)
        #cv2.imshow('Imagetest',img)
        preds =result.json()    
        predictions = preds["predictions"]        
        
        for prediction in predictions:            
            object_class = prediction["class"]
            #logger.info('***** INDIVIDUAL prediction results---')            
            
            if vClassNames[buttonClickCount] == object_class:
                motorLeft.value =0.0
                motorRight.value =0.0                
                logger.info('Found class %s in camera',object_class)
                logger.debug('ButtonClickCount=%d ', buttonClickCount)
                logger.debug('Should detect %s', vClassNames[buttonClickCount])                
                confidence = prediction["confidence"]
                x = prediction["x"]
                y = prediction["y"]
                w = prediction["width"]
                h = prediction["height"]
                
                logger.info('Found class in camera %s',object_class)           
                logger.info('Confidence =%d ',confidence)
                logger.info('x =%d ',x)
                logger.info('y =%d  ',y)
                logger.info('width =%d ',w)
                logger.info('height=%d  ',h)
                start_point = (int(x-width/2),int(y-height/2))
                end_point = (int(x+width/2), int(y+height/2))                
                #Uncomment below to see the rectangle drawn                
                if ( (buttonClickCount == 1) or (buttonClickCount == 2) or (buttonClickCount ==0)):                                           
                    logger.info(' %s detected at x cordinate %f', vClassNames[buttonClickCount] , x)
                    #Open Cv box prints (startX,startY,endX,endY)
                    midX = (x+w) / 2
                    
                    #TODO Need to write function to draw rectangle
                    #Uncomment below to see the rectangle drawn
                    #logger.info('Drawing rectangle.')
                    #drawRfRectangle(img,start_point, end_point, object_class, confidence)
                    #cv2.imshow("Output",img)
                    if x < 120.0:
                        logger.info(' %s detected at left ', vClassNames[buttonClickCount])
                        motorLeft.value =motorPower
                        motorRight.value=motorStopPower
                        #os.system(soundCmd + vClassNames[buttonClickCount] + '_left.mp3')
                        buttonClickCount
                        sleep(sleepTimer)
                        motorLeft.value =motorStopPower
                    elif x > 210.0:
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

