#!/usr/bin/env python

import cv2
import rospy
import numpy as np
import mediapipe as mp
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# https://youtu.be/EgjwKM3KzGU?si=lKMVMeepe7SQR7KS

class GestureRecognizer:
    def __init__(self):
        self.br = CvBridge()
        # self.out_pub = rospy.Publisher("/bebop_ws/out_image", Image, queue_size=10)
        self.sub = rospy.Subscriber("/bebop_ws/camera_image", Image, self.image_callback)
        self.mp_detector = mp.solutions.hands # mp_hands
        self.hand_detector = self.mp_detector.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.5) # hands_videos TODO: set max_num_hands

        self.image_size = None

        self.mp_draw = mp.solutions.drawing_utils # mp_drawing


    def image_callback(self, msg):
        rospy.loginfo("I received an image")
        frame = self.br.imgmsg_to_cv2(msg)

        self.image_size = [frame.shape[1], frame.shape[0]]

        flipped_frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result = self.hand_detector.process(rgb_frame)

        image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        # rgb_frame.flags.writeable = True
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, hand, self.mp_detector.HAND_CONNECTIONS,
                                            self.mp_draw.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                                            self.mp_draw.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
                
                

                self.out_pub.publish(self.br.cv2_to_imgmsg(cv2.flip(image, 1)))






def main():
    rospy.init_node("gesture_recognizer_node")
    my_node = GestureRecognizer()
    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == "__main__":
    main()