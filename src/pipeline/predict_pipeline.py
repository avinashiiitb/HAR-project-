import numpy as np
import pandas as pd
from src.exception import CustomException
import tensorflow as tf
import cv2

class Predictpipeline:
    def __init__(self) -> None:
        pass

    def give_model(self,path):
        model = tf.keras.models.load_model(path)
        return model

class Preprocessingpipeline:
    def __init__(self) -> None:
        pass

    def preprocessing(self,img_path,model):
        img = cv2.imread(img_path)
        image = cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA)
        print(image.shape)
        image = np.array(image)
        image = np.reshape(image,(1,128,128,3))
        #x = np.expand_dims(image,axis=1)

        pred = np.argmax(model.predict(image) ,axis=1)
        return pred
