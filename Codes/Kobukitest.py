#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import tensorflow as tf
import os


class ZeminSiniflandirma:
    def __init__(self):
        self.bridge = CvBridge()

        
        self.camera = cv2.VideoCapture(2)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        
        self.model = tf.keras.models.load_model('/home/turtle6/catkin_ws/src/Mobilenet_Adamax.h5')

        
        self.classes = ['carpet', 'tiles', 'wood']
        self.img_width, self.img_height = 299, 299
        self.result_pub = rospy.Publisher("/zemin_siniflandirma/sonuc", Image, queue_size=1)

        
        self.save_dir = os.path.join(os.getcwd(), "Surfaces")
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        for cls in self.classes:
            cls_dir = os.path.join(self.save_dir, cls)
            if not os.path.exists(cls_dir):
                os.makedirs(cls_dir)
	
    def run(self):
        rate = rospy.Rate(0.2) 
        while not rospy.is_shutdown():
            
            ret, frame = self.camera.read()
            if not ret:
                continue

            
            img = cv2.resize(frame, (self.img_width, self.img_height))
            x = img.astype('float32')
            x /= 255
            x = np.expand_dims(x, axis=0)

            
            predictions = self.model.predict(x)
           
            class_idx = np.argmax(predictions)
            class_name = self.classes[class_idx]
            print("Classification Result:", class_name, predictions)

            
            save_path = os.path.join(self.save_dir, class_name, str(rospy.get_time())+".png")
            cv2.imwrite(save_path, frame)

            
            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.result_pub.publish(img_msg)

            rate.sleep()
		
if __name__ == '__main__':
    rospy.init_node('zemin_siniflandirma')
    zemin_siniflandirma = ZeminSiniflandirma()
    zemin_siniflandirma.run()
    zemin_siniflandirma.camera.release()

