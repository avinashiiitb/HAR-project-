import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
import imgaug.augmenters as iaa
import tensorflow
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import models,layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Activation,BatchNormalization
from tensorflow.keras.optimizers import Adam

def data_read_preprocessed(file_path):
    try: 
        label = ['00' ,  '01' , '02' , '03' , '04' , '05' , '06' ,'07' , '08' , '09' , '10']
        dirname1 = []
        filename1 = []
        fullpath = []
        for dirname, _, filenames in os.walk(file_path):
            for filename in filenames:
                dirname1.append((dirname))
                filename1.append(filename)
                fullpath.append((os.path.join(dirname, filename)))

            
        df = pd.DataFrame(columns=['path' , 'label' ])
        label_train = []
        path = []
        def create_dataframe(fullpath,label):
            for i in range(len(fullpath)):
                for j in range(len(label)):
                    if (fullpath[i].split('\\')[6][-6:-4]== label[j]):
                        path.append(fullpath[i])
                        label_train.append(label[j])
    
           
            return path,label_train  

        path,label_train=create_dataframe(fullpath,label)
        df['path'] = path
        df['label'] = label_train

        print(len(df))
        labelname = []
        classvalue = []
        for i in range(len(df)):
            for j in range(len(label)):
                if (df['label'][i] ==  label[j]):
                    if (label[j] == '00'):
                        labelname.append('Walkingtowards')
                        classvalue.append('00')
                    elif (label[j] =='01'):
                        labelname.append('walkingaway')
                        classvalue.append('01')
                    elif (label[j] =='02'):
                        labelname.append('pickingobject')
                        classvalue.append('02')
                    elif (label[j] =='03'):
                        labelname.append('bending')
                        classvalue.append('03')
                    elif (label[j] =='04'):
                        labelname.append('sitting')
                        classvalue.append('04')
                    elif (label[j] =='05'):
                        labelname.append('kneeling')
                        classvalue.append('05')
                    elif (label[j] =='06'):
                        labelname.append('crawling')
                        classvalue.append('06')
                    elif (label[j] =='07'):
                        labelname.append('walkingontoes')
                        classvalue.append('07')
                    elif (label[j] =='08'):
                        labelname.append('limping')
                        classvalue.append('08')
                    elif (label[j] =='09'):
                        labelname.append('shortsteps')
                        classvalue.append('09')
                    else :
                        labelname.append('scissorsgait')
                        classvalue.append('10')


        df['labelname'] = labelname
        df['classvalue'] = classvalue 
        
        return df
    except Exception as e:
        raise CustomException(e,sys)
    

def data_read():
    try: 
        label = ['05' ,  '06' , '07' , '08' , '09' , '10' , '11' ,'16' , '17' , '18' , '19']
        dirname1 = []
        filename1 = []
        fullpath = []
        for dirname, _, filenames in os.walk('C:/Users/avina/Downloads/77'):
            for filename in filenames:
                dirname1.append((dirname))
                filename1.append(filename)
                fullpath.append((os.path.join(dirname, filename)))

            
        df = pd.DataFrame(columns=['path' , 'label' ])
        label_1 = []
        path = []
        def create_dataframe(fullpath,label):
            for i in range(len(fullpath)):
                for j in range(len(label)):
                    if (fullpath[i].split('/')[4][29:31]== label[j]):
                        path.append(fullpath[i])
                        label_1.append(label[j])
    
           
            return path,label_1  

        path,label_1=create_dataframe(fullpath,label)
        df['path'] = path
        df['label'] = label_1

        
        labelname = []
        classvalue = []
        for i in range(len(df)):
            for j in range(len(label)):
                if (df['label'][i] ==  label[j]):
                    if (label[j] == '05'):
                        labelname.append('Walkingtowards')
                        classvalue.append('00')
                    elif (label[j] =='06'):
                        labelname.append('walkingaway')
                        classvalue.append('01')
                    elif (label[j] =='07'):
                        labelname.append('pickingobject')
                        classvalue.append('02')
                    elif (label[j] =='08'):
                        labelname.append('bending')
                        classvalue.append('03')
                    elif (label[j] =='09'):
                        labelname.append('sitting')
                        classvalue.append('04')
                    elif (label[j] =='10'):
                        labelname.append('kneeling')
                        classvalue.append('05')
                    elif (label[j] =='11'):
                        labelname.append('crawling')
                        classvalue.append('06')
                    elif (label[j] =='16'):
                        labelname.append('walkingontoes')
                        classvalue.append('07')
                    elif (label[j] =='17'):
                        labelname.append('limping')
                        classvalue.append('08')
                    elif (label[j] =='18'):
                        labelname.append('shortsteps')
                        classvalue.append('09')
                    else :
                        labelname.append('scissorsgait')
                        classvalue.append('10')


        df['labelname'] = labelname
        df['classvalue'] = classvalue 
        print(len(df))
        return df
    except Exception as e:
        raise CustomException(e,sys)
    
def augmentation(x_train_resize_1,y_train):
    try:
        aug1 = iaa.GaussianBlur(sigma = (0,1))
        aug10 = iaa.AdditiveGaussianNoise(loc = 0 , scale = (0,1) , per_channel= True )
        aug2 = iaa.Emboss(alpha = (1), strength = 1.5)
        aug3 = iaa.Sharpen(alpha = (1.0) , lightness = (1.5))
        aug4 = iaa.Crop(px=(0, 16))

        aug6 = iaa.ImpulseNoise(0.1)

        aug8 = iaa.LinearContrast((0.4, 1.6))


        def augment(data,data1):
            aug_X_train = []
            aug_y_train = []
            for i  in range(len(data)):
                aug_X_train.append(data[i])
        
                aug_X_train.append(aug1.augment_image(data[i]))
                aug_X_train.append(aug10.augment_image(data[i]))
                aug_X_train.append(aug3.augment_image(data[i]))
                aug_X_train.append(aug6.augment_image(data[i]))
                for j in range(5):
                    aug_y_train.append(data1[i])

            return   aug_X_train,aug_y_train
        
        x_train_aug,aug_y_train = augment(x_train_resize_1,y_train)
        x_train_aug_2 = np.array(x_train_aug,dtype = 'float32')
        aug_y_train_1 = np.array(aug_y_train ,dtype ='float32')
        

        return x_train_aug_2,aug_y_train_1
    
    except Exception as e:
        CustomException(e,sys)


def normalization(x,xdash):
    try:
        mean = np.mean(x)
        std = np.mean(xdash)
        train = (x - mean)/std
        test = (xdash -mean)/std
        return train,test
    except Exception as e:
        CustomException(e,sys)


def create_model():
    try:
        def activation_block(x):
            return layers.BatchNormalization()(layers.Activation("relu")(x))


        def conv_stem(x, filters:int, patch_size:int):
            x = layers.Conv2D(filters,kernel_size = patch_size, strides = patch_size)(x)
            return activation_block(x)


        def conv_mixer_block(x,filters:int,kernel_size:int):
            x0=x
            x = layers.DepthwiseConv2D(kernel_size = kernel_size , padding ='same')(x)
    #x = layers.Dropout(0.5)(x)
            x = layers.Add()([activation_block(x),x0])
    
            x = layers.Conv2D(filters, kernel_size =1)(x)
            x=activation_block(x)
            return x


        def get_conv_mixer_256_8(image_size =72,filters =128 , depth=2,kernel_size =5,patch_size=2,num_classes=11):
            inputs = keras.Input((128,128,3))
            x = conv_stem(inputs,filters,patch_size)
            for _ in range(depth):
                x = conv_mixer_block(x,filters,kernel_size)
            x = layers.GlobalAvgPool2D()(x)
    #x = layers.Dropout(0.5)(x)
            outputs = layers.Dense(num_classes, activation="softmax")(x)
            return keras.Model(inputs, outputs)


        conv_mixer_model = get_conv_mixer_256_8()
        print(conv_mixer_model.summary())
        return conv_mixer_model
    except Exception as e:
         CustomException(e,sys)
    



