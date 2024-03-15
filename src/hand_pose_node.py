#!/usr/bin/env python

import cv2
import rospy
import numpy as numpy
import mediapipe as mp
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# https://youtu.be/EgjwKM3KzGU?si=lKMVMeepe7SQR7KS

class HandDetector:
    def __init__(self):
        self.br = CvBridge()
        self.sub = rospy.Subscriber("/bebop_ws/camera_image", Image, self.image_callback)
        self.mp_detector = mp.solutions.hands
        self.hand_detector = self.mp_detector.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.5)

        self.mp_draw = mp.solutions.drawing_utils


    def image_callback(self, msg):
        rospy.loginfo("I received an image")
        frame = self.br.imgmsg_to_cv2(msg)
        
        with self.hand_detector as hands:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            result = hands.process(rgb_frame)
            # rgb_frame.flags.writeable = True
            print(result.multi_handedness)
            if result.multi_hand_landmarks:
                for num, hand in enumerate(result.multi_hand_landmarks):
                    print(hand)
                    wrist_lmk = hand.landmark[self.hand_detector.HandLandmark.WRIST]
                    self.mp_draw.draw_landmarks(frame, hand, mp.hands.HAND_CONNECTIONS,
                                                self.mp_draw.DrawingSpecs(color=(121,22,76), thickness=2, circle_radius=4),
                                                self.mp_draw.DrawingSpecs(color=(121,44,250), thickness=2, circle_radius=2))
                    
                    get_label(num, hand, result)

                    cv2.imshow("final img", frame)
                    if cv2.waitKey(1) == ord('q'):
                        break


    def get_label(index, hands, results):
        output = None
        for idx, classification in enumerate(results.multi_handedness):
            if classification.classification[0].index = index:
                label = classification.classification[0].label
                text = str(label)

                coords = tuple(np.multiply(
                    np.array((hand.landmark[self.hand_detector.HandLandmark.WRIST].x, hand.landmark[self.hand_detector.HandLandmark.WRIST].y)),
                    [640,480], astype(int)))
                
                output = text, coords

        return output

def main():
    rospy.init_node("subscriber_node")
    my_node = Subscriber()
    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == "__main__":
    main()