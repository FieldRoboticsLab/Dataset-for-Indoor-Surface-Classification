#!/usr/bin/env python
# coding: utf-8

# In[1]:
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.applications.resnet import ResNet50

#import keras
#from keras.models import Sequential
#from keras.layers import Conv2D
#from keras.layers import MaxPool2D
#from keras.layers import Flatten
#from keras.layers import Dense
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

train_path="/home/asiyedemirtas/Desktop/Dataset/YeniDataset/train"
test_path="/home/asiyedemirtas/Desktop/Dataset/YeniDataset/test"
val_path="/home/asiyedemirtas/Desktop/Dataset/YeniDataset/validation"


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





resnet_model= tf.keras.applications.ResNet50(input_shape=(299,299, 3), include_top=False, weights="imagenet")

for layer in resnet_model.layers:
    layer.trainable = False


    
#orjinal hali
x = resnet_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x= Dense(3, activation='softmax')(x)    



model=tf.keras.models.Model(resnet_model.input,x)
opt = tf.keras.optimizers.Adamax(learning_rate=0.0001)


model.compile(optimizer=opt, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["acc"])

mc = ModelCheckpoint("ResNet50.h5", monitor='val_acc', mode='max', save_best_only=True)


history = model.fit(
  train_x,
  train_y,
  validation_data=(val_x,val_y),
  epochs=100,
  batch_size=32,
  callbacks=[mc])


plt.plot(history.history['acc'], label='train acc')

plt.plot(history.history['val_acc'], label='val acc')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()

plt.savefig('ResNet_SGD_100_lr0.0001.jpg')

plt.show()

# loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('ResNet_loss_SGD_100_lr0.0001.jpg')
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
