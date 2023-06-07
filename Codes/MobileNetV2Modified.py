#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
#import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.applications.resnet import ResNet50
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
import keras
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np

train_path=".../train"
test_path=".../test"
val_path=".../validation"

x_train=[]

for folder in os.listdir(train_path):

    sub_path=train_path+"/"+folder

    for img in os.listdir(sub_path):

        image_path=sub_path+"/"+img

        img_arr=cv2.imread(image_path)
        im_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

        img_arr=cv2.resize(im_rgb,(299,299))
        
        

        x_train.append(img_arr)

x_test=[]

for folder in os.listdir(test_path):

    sub_path=test_path+"/"+folder

    for img in os.listdir(sub_path):

        image_path=sub_path+"/"+img

        img_arr=cv2.imread(image_path)
        im_rgb_t = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

        img_arr=cv2.resize(im_rgb_t,(299,299))

        x_test.append(img_arr)

x_val=[]

for folder in os.listdir(val_path):

    sub_path=val_path+"/"+folder

    for img in os.listdir(sub_path):

        image_path=sub_path+"/"+img

        img_arr=cv2.imread(image_path)
        im_rgb_v = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        

        img_arr=cv2.resize(im_rgb_v,(299,299))

        x_val.append(img_arr)

train_x=np.array(x_train)
test_x=np.array(x_test)
val_x=np.array(x_val)

train_x=train_x/255.0
test_x=test_x/255.0
val_x=val_x/255.0

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range=0.2, zoom_range=0.2, height_shift_range=0.2, 
    width_shift_range=0.2, rotation_range=70, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (299,299),
                                                 batch_size = 32,
                                                 class_mode = 'sparse')
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (299,299),
                                            class_mode = 'sparse',
                                            batch_size = 32)
val_set = val_datagen.flow_from_directory(val_path,
                                            target_size = (299,299),
                                            class_mode = 'sparse',
                                            batch_size = 32)

train_y=training_set.classes
test_y=test_set.classes
val_y=val_set.classes

training_set.class_indices

train_y.shape,test_y.shape,val_y.shape

IMAGE_SIZE=(299,299, 3)




MobileNet= tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(299,299, 3), include_top=False, weights="imagenet")

x=MobileNet.layers[-12].output

#orjinal hali
x = MobileNet.layers[-12].output

x= Conv2D(16, (5, 5), activation='relu', padding='same')(x)
x = GlobalAveragePooling2D()(x)
x= Dropout(0.2)(x)
x = Dense(2096, activation='relu')(x)
x= Dense(3, activation='softmax')(x)    

model=tf.keras.models.Model(MobileNet.input,x)

for layer in MobileNet.layers[:-23]:
    layer.trainable = False
    
model.summary()   
    
    
opt = tf.keras.optimizers.Adamax(learning_rate=0.0001)



model.compile(optimizer=opt, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["acc"])

mc = ModelCheckpoint("MobileNetV2_Modified.h5", monitor='val_acc', mode='max', save_best_only=True)

history = model.fit(
  train_x,
  train_y,
  validation_data=(val_x,val_y),
  epochs=100,
  batch_size=32,
  callbacks=[mc])

plt.plot(history.history['acc'], label='train acc')

plt.plot(history.history['val_acc'], label='val acc')

plt.legend()

plt.savefig('---.jpg')

plt.show()

# loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
#plt.ylim(0.0, 1)
plt.legend()
plt.savefig('---.jpg')
plt.show()


model.evaluate(test_x,test_y,batch_size=32)


y_pred=model.predict(test_x)

y_pred=np.argmax(y_pred,axis=1)

#get classification report
print(classification_report(y_pred,test_y))


mat=(confusion_matrix(y_pred,test_y))



df_cm = pd.DataFrame(mat, range(3), range(3))

sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 15}) # font size

plt.show()


