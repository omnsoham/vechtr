#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
import numpy as np 
import pandas as pd 
import cv2
import random
from pathlib import Path
from tqdm import tqdm
#from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import to_categorical
import numpy as np
#from sklearn.mode_selection import train_test_split
import os


labels = [] 
ds_dir = "C:/Users/Soham Kulkarni/OneDrive/Documents/GitHub/vechtor/v4/doorimages/"

image_size = 224 # Define the image size here

def create_dataset(category, label, dataset):
    for img in tqdm(category):
        image_path = os.path.join(ds_dir, img)
        try:
            image = cv2.imread(image_path,cv2.IMREAD_COLOR)
            image = cv2.resize(image,(image_size, image_size))
        except:
            continue
        dataset.append([np.array(image),np.array(label)])
    random.shuffle(dataset)
    return dataset
trash = "C:/Users/Soham Kulkarni/OneDrive/Documents/GitHub/vechtor/v4/trash/"
door = "C:/Users/Soham Kulkarni/OneDrive/Documents/GitHub/vechtor/v4/doortestimages/"
dataset = []
dataset = create_dataset(trash, 1, dataset)
dataset = create_dataset(door, 2, dataset)