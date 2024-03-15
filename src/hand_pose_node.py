#!/usr/bin/env python

import cv2
import rospy
import numpy as np
import mediapipe as mp
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# https://youtu.be/EgjwKM3KzGU?si=lKMVMeepe7SQR7KS

class HandDetector:
    def __init__(self):
        self.br = CvBridge()
        self.out_pub = rospy.Publisher("/bebop_ws/out_image", Image, queue_size=10)
        self.sub = rospy.Subscriber("/bebop_ws/camera_image", Image, self.image_callback)
        self.mp_detector = mp.solutions.hands
        self.hand_detector = self.mp_detector.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.5)

        self.image_size = None

        self.mp_draw = mp.solutions.drawing_utils


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
            for num, hand in enumerate(result.multi_hand_landmarks):
                self.mp_draw.draw_landmarks(image, hand, self.mp_detector.HAND_CONNECTIONS,
                                            self.mp_draw.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                                            self.mp_draw.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
                
                out = self.get_hand_center(num, hand, result)
                if out:
                    hand_center_x = out[0]
                    hand_center_y = out[1]
                    cv2.circle(image, (hand_center_x, hand_center_y), 3, (0, 255, 0), 2)
                    cv2.circle(image, (int(self.image_size[0]/2), int(self.image_size[1]/2)), 3, (0, 255, 0), 2) # place a circle in the center of the image


                    cv2.line(image, (hand_center_x, hand_center_y), (int(self.image_size[0]/2), int(self.image_size[1]/2)), (0,255,0), 2)

                self.out_pub.publish(self.br.cv2_to_imgmsg(cv2.flip(image, 1)))


    def get_hand_center(self, index, hands, results):
        for idx, classification in enumerate(results.multi_handedness):
            coords = tuple(np.multiply(
                np.array((hands.landmark[self.mp_detector.HandLandmark.MIDDLE_FINGER_MCP].x, hands.landmark[self.mp_detector.HandLandmark.MIDDLE_FINGER_MCP].y)),
                self.image_size).astype(int))

            return coords
        return None


    def get_hand_bbox(self, index, hands, results):
        pass


def main():
    rospy.init_node("hand_detector_node")
    my_node = HandDetector()
    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == "__main__":
    main()