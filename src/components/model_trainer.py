import os
import sys
import cv2
import tensorflow as tf
import numpy as np

from dataclasses import dataclass
from src.utils import create_model
from tensorflow import keras

from src.logger import logging
from src.exception import CustomException
from tensorflow.keras.callbacks import ModelCheckpoint
logging.info('we are in the model')
@dataclass

class ModelTrainer():

        
    def initiate_model_trainer(self,x,y,xdash,ydash):
        try:
            logging.info('running the model')
            callbacks = [ModelCheckpoint('C:/Users/avina/HAR_END_TO_END./model_128_endtoend_STRATIFY_{epoch:4d}_val_accuracy{val_accuracy:.5f}_val_loss{val_loss:.5f}.h5', save_best_only=True, mode = 'max',  monitor='val_accuracy' )]

            model = create_model()
            model.compile(
            optimizer = keras.optimizers.Adam(lr=1e-03, decay=1e-6),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],)
            logging.info('model compiled')
            print(x.shape),print(y.shape),print(xdash.shape),print(ydash.shape)
            history = model.fit(x,y,epochs=1000,verbose=1, callbacks=callbacks,validation_data=(xdash,ydash))# Suppress chatty output
                        #callbacks=callbacks,,
                        
            logging.info('run the model ')
            return model
            
            
        except Exception as e:
            CustomException(e,sys)


