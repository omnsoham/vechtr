from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("hardhat\models\detection_model-ex-020--loss-0008.462.h5")
detector.setJsonPath("hardhat\json\detection_config.json")
detector.loadModel()


import random

testImages = os.listdir("hardhat/validation/images")
randomFile = testImages[random.randint(0, len(testImages) - 1)]

detectedImage, detections = detector.detectObjectsFromVideo(output_type="array", 
                                                            input_image="hardhat/validation/images/{0}".format(randomFile), 
                                                            minimum_percentage_probability=30)
showImage(detectedImage)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")


