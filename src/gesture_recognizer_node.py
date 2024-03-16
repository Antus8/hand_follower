#!/usr/bin/env python

import cv2
import rospy
import numpy as np
import mediapipe as mp
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image

# https://youtu.be/EgjwKM3KzGU?si=lKMVMeepe7SQR7KS

class GestureRecognizer:
    def __init__(self):
        self.br = CvBridge()
        self.hands_status_pub = rospy.Publisher("/bebop_ws/hands_status", String, queue_size=10)
        self.sub = rospy.Subscriber("/bebop_ws/camera_image", Image, self.image_callback, queue_size=1)
        self.mp_detector = mp.solutions.hands # mp_hands
        self.hand_detector = self.mp_detector.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.5) # hands_videos TODO: set max_num_hands

        self.mp_draw = mp.solutions.drawing_utils # mp_drawing

        self.finger_tips_ids = [self.mp_detector.HandLandmark.INDEX_FINGER_TIP, self.mp_detector.HandLandmark.MIDDLE_FINGER_TIP, self.mp_detector.HandLandmark.RING_FINGER_TIP,
                                self.mp_detector.HandLandmark.PINKY_TIP]

        self.finger_status = {'RIGHT_THUMB':False, 'RIGHT_INDEX':False, 'RIGHT_MIDDLE':False, 'RIGHT_RING':False, 'RIGHT_PINKY':False,
                                'LEFT_THUMB':False, 'LEFT_INDEX':False, 'LEFT_MIDDLE':False, 'LEFT_RING':False, 'LEFT_PINKY':False}
        
        self.count = {'RIGHT':0, "LEFT":0}
                    


    def image_callback(self, msg):
        self.count = {'RIGHT':0, "LEFT":0}
        frame = self.br.imgmsg_to_cv2(msg)

        self.height, self.width = frame.shape[:2]

        flipped_frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result = self.hand_detector.process(rgb_frame)

        image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

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

            self.publish_hand_status()

        
    
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