#!/usr/bin/env python

import cv2
import rospy
import numpy as numpy
import mediapipe as mp
from cv_bridge import CvBridge
from sensor_msgs.msg import Image



class HandDetector:
    def __init__(self):
        self.br = CvBridge()
        self.sub = rospy.Subscriber("/camera_image", Image, self.callback)
        self.mp_detector = mp.solutions.hands
        self.hand_detector = self.mp_detector.Hands()

        self.mp_draw = mp.solutions.drawing_utils


    def callback(self, msg):
        rospy.loginfo("I received an image")
        frame = self.br.imgmsg_to_cv2(msg)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #cv2.imshow('image', frame)
        #cv2.waitKey(0)
        result = self.hand.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                print(hand)
                self.mp_draw.draw_landmarks(frame, hand, mp.hands.HAND_CONNECTIONS)

                cv2.imshow("final img", frame)
                if cv2.waitKey(1) == ord('q'):
                    break



def main():
    rospy.init_node("subscriber_node")
    my_node = Subscriber()
    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == "__main__":
    main()