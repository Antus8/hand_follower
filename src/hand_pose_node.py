#!/usr/bin/env python

import cv2
import rospy
import numpy as np
import mediapipe as mp
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image

# https://youtu.be/EgjwKM3KzGU?si=lKMVMeepe7SQR7KS

class HandDetector:
    def __init__(self):
        self.br = CvBridge()
        self.out_pub = rospy.Publisher("/bebop/out_image", Image, queue_size=1)
        self.sub = rospy.Subscriber("/bebop/image_raw", Image, self.image_callback, queue_size=1)
        self.hands_status_sub = rospy.Subscriber("/bebop/hands_status", String, self.hands_status_callback, queue_size=1)
        self.mp_detector = mp.solutions.hands
        self.hand_detector = self.mp_detector.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.5)

        self.image_size = None

        self.mp_draw = mp.solutions.drawing_utils
        self.right_counter, self.left_counter = 0, 0 

    def fake_img_callback(self, msg):
        # Convert image message to cv2 image with RGB encoding
        frame_rgb = self.br.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        
        # Convert RGB to BGR format (if needed)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Publish the converted image message
        self.out_pub.publish(self.br.cv2_to_imgmsg(frame_bgr))
        

    def image_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        # frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)
        self.image_size = [frame.shape[1], frame.shape[0]]
        
        flipped_frame = cv2.flip(frame, 1)
        
        # rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        # rgb_frame.flags.writeable = False
        result = self.hand_detector.process(flipped_frame)

        image = cv2.cvtColor(flipped_frame, cv2.COLOR_RGB2BGR)
        # image = flipped_frame

        # rgb_frame.flags.writeable = True
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, hand, self.mp_detector.HAND_CONNECTIONS,
                                            self.mp_draw.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                                            self.mp_draw.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
                
                # out = self.get_hand_center(hand, result)
                x_min, y_min, x_max, y_max = self.get_hand_bbox(hand, result)
                if x_min:
                    # hand_center_x = out[0]
                    # hand_center_y = out[1]

                    hand_center_x = int(round((x_max + x_min)/2))
                    hand_center_y = int(round((y_max + y_min)/2))

                    cv2.circle(image, (hand_center_x, hand_center_y), 3, (0, 255, 0), 2)
                    cv2.circle(image, (int(self.image_size[0]/2), int(self.image_size[1]/2)), 3, (0, 255, 0), 2) # place a circle in the center of the image

                    cv2.line(image, (hand_center_x, hand_center_y), (int(self.image_size[0]/2), int(self.image_size[1]/2)), (0,255,0), 2)
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

                    final_img = cv2.flip(image, 1)

                    cv2.putText(final_img, f"Total Fingers: {self.right_counter + self.left_counter}", (10,25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                    # cv2.putText(final_img, str(self.right_counter + self.left_counter), (self.image_size[0]//2-100, 25), cv2.FONT_HERSHEY_COMPLEX, 3, (20,255,155), 10, 10)

                    # final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                self.out_pub.publish(self.br.cv2_to_imgmsg(final_img))

        else:
            rospy.loginfo("No detection")


    def get_hand_center(self, hands, results):
        for classification in results.multi_handedness:
            coords = tuple(np.multiply(
                np.array((hands.landmark[self.mp_detector.HandLandmark.MIDDLE_FINGER_MCP].x, hands.landmark[self.mp_detector.HandLandmark.MIDDLE_FINGER_MCP].y)),
                self.image_size).astype(int))

            return coords
        return None


    def get_hand_bbox(self, hands, results):
        coords = {'index_finger_mcp':None, 'pinky_mcp':None, 'wrist':None}

        for classification in results.multi_handedness:
            coords['index_finger_mcp'] = tuple(np.multiply(
                np.array((hands.landmark[self.mp_detector.HandLandmark.INDEX_FINGER_MCP].x, hands.landmark[self.mp_detector.HandLandmark.INDEX_FINGER_MCP].y)),
                self.image_size).astype(int))

            coords['pinky_mcp'] = tuple(np.multiply(
                np.array((hands.landmark[self.mp_detector.HandLandmark.PINKY_MCP].x, hands.landmark[self.mp_detector.HandLandmark.PINKY_MCP].y)),
                self.image_size).astype(int))

            coords['wrist'] = tuple(np.multiply(
                np.array((hands.landmark[self.mp_detector.HandLandmark.WRIST].x, hands.landmark[self.mp_detector.HandLandmark.WRIST].y)),
                self.image_size).astype(int))

        
            x_min, y_min = min(coords['index_finger_mcp'][0], coords['pinky_mcp'][0], coords['wrist'][0]), min(coords['index_finger_mcp'][1], coords['pinky_mcp'][1], coords['wrist'][1])
            x_max, y_max = max(coords['index_finger_mcp'][0], coords['pinky_mcp'][0], coords['wrist'][0]), max(coords['index_finger_mcp'][1], coords['pinky_mcp'][1], coords['wrist'][1])

            return [x_min, y_min, x_max, y_max]

        return None


    def hands_status_callback(self, msg):
        self.right_counter = int(msg.data.split(" ")[1])
        self.left_counter = int(msg.data.split(" ")[3])
        


def main():
    rospy.init_node("hand_detector_node")
    my_node = HandDetector()
    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == "__main__":
    main()