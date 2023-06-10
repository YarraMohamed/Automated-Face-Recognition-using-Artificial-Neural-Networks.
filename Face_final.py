# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 06:50:47 2022

@author: 20100
"""
#import libraries
import os
import cv2
import random as rn
from PIL import Image
import numpy as np
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam,SGD,Adadelta,Adagrad,RMSprop
from tensorflow.keras.utils import to_categorical
import tensorflow
import tensorflow as tf
import fnmatch
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Activation,Flatten,Dropout
import tkinter as tk


#Create our lists and folders
X=[]
Z=[]

Alexandra_Daddario='C:\\Users\\Lenovo\\Desktop\\AI Project\\105_classes_pins_dataset\\pins_Alexandra Daddario'
Johnny_Depp='C:\\Users\\Lenovo\\Desktop\\AI Project\\105_classes_pins_dataset\\pins_Johnny Depp'
Robert_Downey ='C:\\Users\\Lenovo\\Desktop\\AI Project\\105_classes_pins_dataset\\pins_Robert Downey Jr'
Jennifer_Lawrence='C:\\Users\\Lenovo\\Desktop\\AI Project\\105_classes_pins_dataset\\pins_Jennifer Lawrence'
CRISTIANO_RONALDO='C:\\Users\\Lenovo\\Desktop\\AI Project\\105_classes_pins_dataset\\pins_Cristiano Ronaldo'
Dwayne_Johnson='C:\\Users\\Lenovo\\Desktop\\AI Project\\105_classes_pins_dataset\\pins_Dwayne Johnson'
Megan_Fox='C:\\Users\\Lenovo\\Desktop\\AI Project\\105_classes_pins_dataset\\pins_Megan Fox'

#making train data for each folder
def assign_label(img,name):
    return name

IMG_SIZE=100
def make_train_data(name,DIR):
    for img in tqdm(os.listdir(DIR)):
        if fnmatch.fnmatch(img,'*jpg'):
            label=assign_label(img,name)
            path=os.path.join(DIR,img)
            img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            X.append(np.array(img))
            Z.append(str(label))

make_train_data('Alexandra Daddario',Alexandra_Daddario)
make_train_data('Johnny Depp',Johnny_Depp)
make_train_data('Robert Downey',Robert_Downey)
make_train_data('Jennifer Lawrence',Jennifer_Lawrence)
make_train_data('Cristiano Ronaldo',CRISTIANO_RONALDO)
make_train_data('Dwayne Johnson',Dwayne_Johnson)
make_train_data('Megan Fox',Megan_Fox)

#converting labels to numbers
le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,8)

#Normalize X 
X=np.array(X)
X=X/255.0

#split the data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)
x_train=np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1) 

#ANN model
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='same',activation='relu',input_shape=( 100, 100,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))  
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=96,kernel_size=(3,3),padding='same',activation='relu'))  
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=96,kernel_size=(3,3),padding='same',activation='relu'))  
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(8,activation='softmax')) 
batch_size=128
epochs=1

#reducing the learning rate
from keras.callbacks import ReduceLROnPlateau
red_lr=ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=.1)

#preprocess the images
datagenerator=ImageDataGenerator(featurewise_center=False,samplewise_center=False,
                          featurewise_std_normalization=False,
                          samplewise_std_normalization=False,
                          zca_whitening=False,
                          rotation_range=10,
                          zoom_range=.1,
                          width_shift_range=.2,
                          height_shift_range=.2,
                          horizontal_flip=True,
                          vertical_flip=False)

datagenerator.fit(x_train)
model.compile(optimizer=Adam(lr=.001),loss="categorical_crossentropy",metrics=["accuracy"])
model.fit_generator(datagenerator.flow(x_train,y_train,batch_size=batch_size)
                    ,epochs=50
                    ,validation_data=(x_test,y_test),
                    verbose=1
                    )

#model accuracy
score=model.evaluate(x_test,y_test,batch_size=128)
print('Test accuracy:', format(score[1]*100),'%')

#preparing the data to be reconginzed
from tkinter import*
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import cv2
import Face_GUI as K 

root=Tk()
b=K.window(root)   
    
#prepare the photo to be predicted
def prepare(filepath):
      IMG_SIZE=100
      img=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
      img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
      return img

pred_img = np.array([prepare((b.filename))])
prediction = model.predict(pred_img)
print (prediction)

#predictions
if prediction[0][0]==1:
       b.out_label = "this image for: Alexandra Daddario"
       print("this image for: Alexandra Daddario")
if prediction[0][1]==1:
       b.out_label = "this image for: CRISTIANO RONALDO"
       print("this image for: CRISTIANO RONALDO")
if prediction[0][2]==1:
         b.out_label = "this image for: Dwayne Johnson"
         print("this image for: Dwayne Johnson")
if prediction[0][3]==1:
         b.out_label = "this image for: Jennifer Lawrence "
         print("this image for: Jennifer Lawrence ")
if prediction[0][4]==1:
         b.out_label = "this image for: Johnny Depp"
         print("this image for: Johnny Depp")    
if prediction[0][5]==1:
         b.out_label = "this image for: Megan Fox"
         print("this image for: Megan Fox")     
if prediction[0][6]==1:
         b.out_label = "this image for: Robert Downey"
         print("this image for: Robert Downey")   
root.mainloop() 


# alex 0 , chris 1 ,the rock 2 , jen 3 , johny 4 , 5 maria ,6 megan , 7 robert   #
 