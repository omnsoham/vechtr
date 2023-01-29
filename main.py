from imageai.Detection.Custom import CustomVideoObjectDetection
import os as os
import cv2 as cv
import 

execution_path = os.getcwd()
camera = cv2.VideoCapture(0)


def showImage(img):
    window_name = 'image'
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


hardhatLoc = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03492922'

hardhatImages = req.get(hardhatLoc).text
noOfImages = 0

if not os.path.exists('hardhat'):
    os.makedirs('hardhat')

for i in hardhatImages.split('\n'):
    try:
        r = req.get(i, timeout=0.5)
        file = i.split("/")[-1].split('\r')[0]
        if 'image/jpeg' in r.headers['Content-Type']:
            if len(r.content) > 8192:
                with open('hardhat\\' + file, 'wb') as outfile:
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


peopleLoc = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152'

peopleImages = req.get(peopleLoc).text
noOfImages = 0

if not os.path.exists('people'):
    os.makedirs('people')

for i in peopleImages.split('\n'):
    try:
        r = req.get(i, timeout=0.5)
        file = i.split("/")[-1].split('\r')[0]
        if 'image/jpeg' in r.headers['Content-Type']:
            if len(r.content) > 8192:
                with open('people\\' + file, 'wb') as outfile:
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




hardhatImages = os.listdir("hardhat")
peopleOnly = video_detector.CustomObjects(person=True)


video_detector.detectObjectsFromVideo(camera_input=camera,
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          frames_per_second=20,
                                          minimum_percentage_probability=40,
                                          log_progress=True)


for i in hardhatImages:
    imageFile = "hardhat/{0}".format(i)
    detectedImage, detections = video_detector.detectCustomObjectsFromImage(custom_objects=peopleOnly, output_type="array",
                                                                      input_image=imageFile, 
                                                                      minimum_percentage_probability=30)
    if len(detections) < 0:
        os.remove(imageFile)


if not os.path.exists('hardhat/train/images'):
    os.makedirs('hardhat/train/images')
if not os.path.exists('hardhat/validation/images'):
    os.makedirs('hardhat/validation/images')

hardhatImages = os.listdir("hardhat")
hardhatTrainNums = round(len(hardhatImages) * 0.90)

for i in range(0, hardhatTrainNums):
    file = "hardhat/" + hardhatImages[i]
    if os.path.isfile(file):
        os.rename(file, "hardhat/train/images/" + hardhatImages[i])
    
hardhatImages = os.listdir("hardhat")

for i in hardhatImages:
    file = "hardhat/" + i
    if os.path.isfile(file):
        os.rename(file, "hardhat/validation/images/" + i)







