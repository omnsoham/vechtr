import tensorflow as tf
import torch
import torchvision
from imageai.Detection.Custom import CustomVideoObjectDetection
import os as os
import cv2 as cv
import requests as req

execution_path = os.getcwd()
camera = cv.VideoCapture(0)


def showImage(img):
    window_name = 'image'
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


doorLoc = 'http://www.image-net.org/api/download/imagenet.bbox.synset?wnid=n04239074'

doorImages = req.get(doorLoc).text
noOfImages = 0

if not os.path.exists('door'):
    os.makedirs('door')

for i in doorImages.split('\n'):
    file = "testfiletext"
    try:
        r = req.get(i, timeout=0.5)
        file = i.split("/")[-1].split('\r')[0]
        if 'image/jpeg' in r.headers['Content-Type']:
            if len(r.content) > 8192:
                with open('door\\' + file, 'wb') as outfile:
                    outfile.write(r.content)
                    noOfImages += 1
                    print('Success: ' + file)
            else:
                print('Failed: ' + file + ' -- Image too small')
        else:
            print('Failed: ' + file + ' -- Not an image')
    except Exception as e:
        print('Failed: ' + file + ' -- Error')
        
print('*********** Download Finished **************')

modelRetinaNet = 'https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5'
modelYOLOv3 = 'https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5'
modelTinyYOLOv3 = 'https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny.h5'

if not os.path.exists('yolo.h5'):
    r = req.get(modelYOLOv3, timeout=0.5)
    with open('yolo.h5', 'wb') as outfile:
        outfile.write(r.content)


video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("yolov3_hololens-yolo_mAP-0.82726_epoch-73.pt")
video_detector.loadModel()




doorImages = os.listdir("door")


video_detector.detectObjectsFromVideo(camera_input=camera,
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          frames_per_second=20,
                                          minimum_percentage_probability=40,
                                          log_progress=True)

for i in doorImages:
    imageFile = "door/{0}".format(i)
    detectedImage, detections = video_detector.detectCustomObjectsFromImage( output_type="array",
                                                                      input_image=imageFile, 
                                                                      minimum_percentage_probability=30)
    if len(detections) < 0:
        os.remove(imageFile)


if not os.path.exists('door/train/images'):
    os.makedirs('door/train/images')
if not os.path.exists('door/validation/images'):
    os.makedirs('door/validation/images')

doorImages = os.listdir("door")
doorTrainNums = round(len(doorImages) * 0.90)

for i in range(0, doorTrainNums):
    file = "door/" + doorImages[i]
    if os.path.isfile(file):
        os.rename(file, "door/train/images/" + doorImages[i])
    
doorImages = os.listdir("door")

for i in doorImages:
    file = "door/" + i
    if os.path.isfile(file):
        os.rename(file, "door/validation/images/" + i)

from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="door")

trainer.setTrainConfig(object_names_array=["door"], batch_size=4, num_experiments=200, 
                       train_from_pretrained_model="yolo.h5")

trainer.trainModel()


model05 = trainer.evaluateModel(model_path="door\models\detection_model-ex-005--loss-0014.238.h5", 
                      json_path="door\json\detection_config.json", iou_threshold=0.5, 
                      object_threshold=0.3, nms_threshold=0.5)
model10 = trainer.evaluateModel(model_path="door\models\detection_model-ex-010--loss-0011.053.h5", 
                      json_path="door\json\detection_config.json", iou_threshold=0.5, 
                      object_threshold=0.3, nms_threshold=0.5)
model15 = trainer.evaluateModel(model_path="door\models\detection_model-ex-015--loss-0009.620.h5", 
                      json_path="door\json\detection_config.json", iou_threshold=0.5, 
                      object_threshold=0.3, nms_threshold=0.5)
model20 = trainer.evaluateModel(model_path="door\models\detection_model-ex-020--loss-0008.462.h5", 
                      json_path="door\json\detection_config.json", iou_threshold=0.5, 
                      object_threshold=0.3, nms_threshold=0.5)

print('---------------------------------------------------------')
print('Iteration 05:', model05[0]['average_precision']['door'],
     'Iteration 10:', model10[0]['average_precision']['door'],
     'Iteration 15:', model15[0]['average_precision']['door'],
     'Iteration 20:', model20[0]['average_precision']['door'])
print('---------------------------------------------------------')

from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("door\models\detection_model-ex-020--loss-0008.462.h5")
detector.setJsonPath("door\json\detection_config.json")
detector.loadModel()


import random

testImages = os.listdir("door/validation/images")
randomFile = testImages[random.randint(0, len(testImages) - 1)]

detectedImage, detections = detector.detectObjectsFromVideo(output_type="array", 
                                                            input_image="door/validation/images/{0}".format(randomFile), 
                                                            minimum_percentage_probability=30)
showImage(detectedImage)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")





