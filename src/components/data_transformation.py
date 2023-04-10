import sys
import numpy as np
from dataclasses import dataclass
import pandas as pd
import cv2
import os
from src.exception import CustomException
from src.logger import logging
import os
import imgaug.augmenters as iaa
from src.utils import data_read_preprocessed
from src.utils import augmentation
from src.utils import normalization

@dataclass
class DataTransformationConfig:
    preprocessed_train_image_file_path = os.path.join('C:\\Users\\avina\\HAR_END_TO_END\\preprocessed','train_images')
    preprocessed_test_image_file_path = os.path.join('C:\\Users\\avina\\HAR_END_TO_END\\preprocessed','test_images')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            df_train = data_read_preprocessed(train_path)
            df_test = data_read_preprocessed(test_path)
            label = ['00' ,  '01' , '02' , '03' , '04' , '05' , '06' ,'07' , '08' , '09' , '10']

                            
            logging.info("Datframe creation completed")

            os.makedirs((os.path.join('C:\\Users\\avina\\HAR_END_TO_END\\preprocessed','train_images')),exist_ok=True)
            os.makedirs((os.path.join('C:\\Users\\avina\\HAR_END_TO_END\\preprocessed','test_images')),exist_ok=True)
            y_train = df_train['classvalue']

            y_train = y_train.to_numpy()

            y_test = df_test['classvalue']

            y_test = y_test.to_numpy()
            def create_data(x_train):
                listimage = []
                for i in range(len(x_train)):
                    path = x_train['path'].iloc[i]

                    image = cv2.imread(path)
                    image = cv2.resize(image,(128,128),interpolation=cv2.INTER_AREA)
        
     
                    listimage.append(image)
       
            
    
                return  listimage  
            
            x_train_resize = create_data(df_train)

            x_train_resize_1 = np.array(x_train_resize)

            x_test_resize = create_data(df_test)
            x_test_resize_1 = np.array(x_test_resize)

            logging.info("Resizing of  images done")

            x_train_aug_2,aug_y_train_1 = augmentation(x_train_resize_1,y_train)
            
            y_test = np.array(y_test ,dtype ='float32')

            logging.info("Augmentation of  images done")
            

            def create_preprocessed_data(x_train,y,flag):
               
    
                for i in range(len(x_train)):
                    
                    if flag == 0:
                        
                        l = y[i]
                        
                        path1 =os.path.join('C:\\Users\\avina\\HAR_END_TO_END\\preprocessed','train_images')
                        cv2.imwrite(os.path.join(path1 ,str(i)+'_'+ str(l)+'.jpg'),x_train[i])
                    if flag == 1:
                        
                        l = y[i]
                        
                        path1 =os.path.join('C:\\Users\\avina\\HAR_END_TO_END\\preprocessed','test_images')
                        cv2.imwrite(os.path.join(path1 ,str(i)+'_'+str(l)+'.jpg'),x_train[i])


            logging.info('Writing Preprocessed images')
            create_preprocessed_data(x_train_aug_2,aug_y_train_1,0)
            create_preprocessed_data(x_test_resize_1,y_test,1)
         

            x_train,x_test = normalization(x_train_aug_2,x_test_resize_1)
            print(x_test.shape),print(x_train.shape)
            print(np.mean(x_train)),print(np.std(x_train)),print(np.mean(x_test)),print(np.std(x_test))
            logging.info("Normalzation of  images done")
            logging.info('Writing done')
            logging.info('returning normalized train and test arrays along with labels and path')
            return (x_train,x_test,aug_y_train_1,y_test,self.data_transformation_config.preprocessed_train_image_file_path,self.data_transformation_config.preprocessed_test_image_file_path)

        except Exception as e:
            raise CustomException(e,sys)
        





