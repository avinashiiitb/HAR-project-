import os
import sys
import cv2
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
import numpy as np
from src.utils import data_read
os.environ['NUMEXPR_MAX_THREADS'] = '16'

os.environ['NUMEXPR_NUM_THREADS'] = '8'
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.model_trainer import ModelTrainer
#from src.components.model_trainer import ModelTrainerConfig
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('C:\\Users\\avina\\HAR_END_TO_END\\artifacts','train_images')
    test_data_path: str=os.path.join('C:\\Users\\avina\\HAR_END_TO_END\\artifacts','test_images')
    raw_data_path: str=os.path.join('C:\\Users\\avina\\HAR_END_TO_END\\artifacts','raw_images')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('entered data ingestion method')
        try:#here i can read from any place
            df = data_read()

            for i in range(len(df)-1):
                path = df['path'].iloc[i]
                if(path == 'C:/Users/avina/Downloads/77\\activity_spectogram_77GHz\\08_bending\\activity_spectogram_77GHz - Shortcut.lnk'):
                    df.drop(index = i,axis =0,inplace = True)
            logging.info('Created the dataframe consisting of all images ')
            os.makedirs(self.ingestion_config.raw_data_path,exist_ok=True)
            os.makedirs(self.ingestion_config.test_data_path,exist_ok=True)
            os.makedirs(self.ingestion_config.train_data_path,exist_ok=True)
           
            def create_data(x_train,flag):
               
    
                for i in range(len(x_train)):
                    
                    if flag == 0:
                        path = x_train['path'].iloc[i]
                        l = x_train['classvalue'].iloc[i] 
                        image = cv2.imread(path)
                        path1 =os.path.join('C:\\Users\\avina\\HAR_END_TO_END\\artifacts','raw_images')
                        cv2.imwrite(os.path.join(path1 ,str(i)+'_'+ l+'.jpg'),image)
                    if flag == 1:
                        path = x_train['path'].iloc[i]
                        l = x_train['classvalue'].iloc[i]
                        image = cv2.imread(path)
                        path1 =os.path.join('C:\\Users\\avina\\HAR_END_TO_END\\artifacts','train_images')
                        cv2.imwrite(os.path.join(path1 ,str(i)+'_'+l+'.jpg'),image)
                    if flag == 2:
                        path = x_train['path'].iloc[i]
                        l = x_train['classvalue'].iloc[i]
                        image = cv2.imread(path)
                        path1 =os.path.join('C:\\Users\\avina\\HAR_END_TO_END\\artifacts','test_images')
                        cv2.imwrite(os.path.join(path1 ,str(i)+ '_'+l+'.jpg'),image)
                
            create_data(df,0)
            y_train = df['classvalue']
            y_train
            y_train = y_train.to_numpy()
            logging.info('train test split initiated')
            train_set,test_set,y_train,y_test=train_test_split(df,y_train,test_size=0.2,random_state=42,stratify=y_train)        
            create_data(train_set,1)
            create_data(test_set,2)
            logging.info('ingestion done')
            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)

            
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,train_label,test_label,processed_train_path,processed_test_path=data_transformation.initiate_data_transformation(train_data,test_data)
    modeltrainer=ModelTrainer()
    modeltrainer.initiate_model_trainer(train_arr,train_label,test_arr,test_label)

