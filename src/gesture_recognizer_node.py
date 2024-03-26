#!/usr/bin/env python

import cv2
import time
import rospy
import numpy as np
import mediapipe as mp
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, UInt8, Empty

# https://youtu.be/EgjwKM3KzGU?si=lKMVMeepe7SQR7KS

class GestureRecognizer:
    def __init__(self):
        self.br = CvBridge()
        self.hands_status_pub = rospy.Publisher("/bebop/hands_status", String, queue_size=10)
        self.flight_animations_pub = rospy.Publisher("/bebop/flip_fake", UInt8, queue_size=1)

        self.sub = rospy.Subscriber("/bebop/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.drone_status_sub = rospy.Subscriber("bebop/enable", Empty, self.drone_status_callback)
        self.mp_detector = mp.solutions.hands
        self.hand_detector = self.mp_detector.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.5, max_num_hands=1)

        self.mp_draw = mp.solutions.drawing_utils

        self.finger_tips_ids = [self.mp_detector.HandLandmark.INDEX_FINGER_TIP, self.mp_detector.HandLandmark.MIDDLE_FINGER_TIP, self.mp_detector.HandLandmark.RING_FINGER_TIP,
                                self.mp_detector.HandLandmark.PINKY_TIP]

        self.finger_status = {'RIGHT_THUMB':False, 'RIGHT_INDEX':False, 'RIGHT_MIDDLE':False, 'RIGHT_RING':False, 'RIGHT_PINKY':False,
                                'LEFT_THUMB':False, 'LEFT_INDEX':False, 'LEFT_MIDDLE':False, 'LEFT_RING':False, 'LEFT_PINKY':False}
        
        self.count = {'RIGHT':0, "LEFT":0}
        self.flight_animation_msg = UInt8()

        self.drone_armed = False
        self.last_animation_time = 0
        self.ignore_first = False


    def drone_status_callback(self, msg):
        self.drone_armed = not self.drone_armed
        rospy.logwarn(f"Drone state CHANGED! Drone is armed {self.drone_armed}")
                    


    def image_callback(self, msg):
        self.count = {'RIGHT':0, "LEFT":0}

        frame = self.br.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.height, self.width = frame.shape[:2]
        flipped_frame = cv2.flip(frame, 1)

        if self.drone_armed:
            result = self.hand_detector.process(flipped_frame)

            image = cv2.cvtColor(flipped_frame, cv2.COLOR_RGB2BGR)

            if result.multi_handedness:
                for hand_index, hand_info in enumerate(result.multi_handedness):
                    hand_label = hand_info.classification[0].label

                    hand_landmarks = result.multi_hand_landmarks[hand_index]

                    for tip_index in self.finger_tips_ids:
                        finger_name = tip_index.name.split("_")[0]

                        if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                            self.finger_status[hand_label.upper() + "_" + finger_name] = True
                            self.count[hand_label.upper()] += 1

                thumb_tip_x = hand_landmarks.landmark[self.mp_detector.HandLandmark.THUMB_TIP].x
                thumb_mcp_x = hand_landmarks.landmark[self.mp_detector.HandLandmark.THUMB_TIP - 2].x

                if (hand_label=="Right" and (thumb_tip_x < thumb_mcp_x)) or (hand_label=="Left" and (thumb_tip_x > thumb_mcp_x)):
                    self.finger_status[hand_label.upper() + "_THUMB"] = True
                    self.count[hand_label.upper()] += 1

                self.publish_flight_animation()
                self.publish_hand_status()

    
    def publish_flight_animation(self):
        # Use only right hand for now
        current_time = time.time()
        if self.count['RIGHT'] != 5 and (current_time - self.last_animation_time) > 5 and self.ignore_first:
            self.flight_animation_msg.data = self.count['RIGHT'] - 1
            rospy.logwarn(f"PUBLISHING flight animation command {self.count['RIGHT'] - 1}")
            self.flight_animations_pub.publish(self.flight_animation_msg)
            self.last_animation_time = current_time
        
        if not self.ignore_first:
            self.ignore_first = not self.ignore_first

            
    
    def publish_hand_status(self):
        out_string = ""
        for hand in self.count.keys():
            out_string = out_string + hand + " " + str(self.count[hand]) + " "

        self.hands_status_pub.publish(out_string)
        self.count = {'RIGHT':0, "LEFT":0}


def main():
    rospy.init_node("gesture_recognizer_node")
    my_node = GestureRecognizer()
    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == "__main__":
    main()