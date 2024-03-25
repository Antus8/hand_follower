#!/usr/bin/env python


import os
import cv2
import glob
import math
import time
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CameraPublisher():
   def __init__(self):
       self.image = None
       self.br = CvBridge()
       self.loop_rate = rospy.Rate(1)


       # Publishers
       self.pub = rospy.Publisher('/bebop/image_raw', Image, queue_size=10)


   def start(self):
       rospy.loginfo("Publisher starting...")
       cap = cv2.VideoCapture(0)
      
       while not rospy.is_shutdown():
            if(cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    # rospy.logwarn('Publishing image...')
                    self.pub.publish(self.br.cv2_to_imgmsg(frame))
                    time.sleep(0.1)
                else:
                    rospy.loginfo('No input video!')
                    break


if __name__ == '__main__':
   rospy.init_node("Camera_publisher")
   my_node = CameraPublisher()
   my_node.start()