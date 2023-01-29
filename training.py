from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hardhat")

trainer.setTrainConfig(object_names_array=["person hardhat"], batch_size=4, num_experiments=200, 
                       train_from_pretrained_model="yolo.h5")

trainer.trainModel()


model05 = trainer.evaluateModel(model_path="hardhat\models\detection_model-ex-005--loss-0014.238.h5", 
                      json_path="hardhat\json\detection_config.json", iou_threshold=0.5, 
                      object_threshold=0.3, nms_threshold=0.5)
model10 = trainer.evaluateModel(model_path="hardhat\models\detection_model-ex-010--loss-0011.053.h5", 
                      json_path="hardhat\json\detection_config.json", iou_threshold=0.5, 
                      object_threshold=0.3, nms_threshold=0.5)
model15 = trainer.evaluateModel(model_path="hardhat\models\detection_model-ex-015--loss-0009.620.h5", 
                      json_path="hardhat\json\detection_config.json", iou_threshold=0.5, 
                      object_threshold=0.3, nms_threshold=0.5)
model20 = trainer.evaluateModel(model_path="hardhat\models\detection_model-ex-020--loss-0008.462.h5", 
                      json_path="hardhat\json\detection_config.json", iou_threshold=0.5, 
                      object_threshold=0.3, nms_threshold=0.5)

print('---------------------------------------------------------')
print('Iteration 05:', model05[0]['average_precision']['person hardhat'],
     'Iteration 10:', model10[0]['average_precision']['person hardhat'],
     'Iteration 15:', model15[0]['average_precision']['person hardhat'],
     'Iteration 20:', model20[0]['average_precision']['person hardhat'])
print('---------------------------------------------------------')




