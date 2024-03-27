#!/usr/bin/env python

import cv2
import time
import rospy
import math
import numpy as np
import mediapipe as mp
from pid_package.pid_regulator import PID
from cv_bridge import CvBridge
from std_msgs.msg import String, Empty
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# https://youtu.be/EgjwKM3KzGU?si=lKMVMeepe7SQR7KS

class HandDetector:
    def __init__(self):

        self.sub = rospy.Subscriber("/bebop/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.hands_status_sub = rospy.Subscriber("/bebop/hands_status", String, self.hands_status_callback, queue_size=1)
        self.drone_status_sub = rospy.Subscriber("bebop/enable", Empty, self.drone_status_callback)
        
        self.flight_pub = rospy.Publisher("/bebop/cmd_vel", Twist, queue_size=1)
        self.out_pub = rospy.Publisher("/bebop/out_image", Image, queue_size=1)

        self.drone_armed = False

        # Gesture Recognizer 
        self.br = CvBridge()
        self.mp_detector = mp.solutions.hands
        self.hand_detector = self.mp_detector.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.5, max_num_hands=1)

        self.image_size = None
        self.lower_error_bound, self.upper_error_bound = 0, 0
        self.mp_draw = mp.solutions.drawing_utils
        self.right_counter, self.left_counter = 0, 0

        self.display = True
        self.max_euclidean_error = 0

        # Control 
        self.dt = 0.1
        self.yaw_pid = PID(p = 1, i = 0.2, d = 0.01, sat = 1, dt = self.dt)
        self.z_pid = PID(p = 1, i = 0.2, d = 0.01, sat = 1, dt = self.dt)
        self.x_pid = PID(p = 1, i = 0.2, d = 0.01, sat = 1, dt = self.dt)
        
        # self.yaw_pid = [1, 0.2, 0.01]
        # self.previous_yaw_error = 0
        # self.yaw_integral = 0

        #self.z_pid = [1, 0.2, 0.01]
        #self.previous_z_error = 0
        #self.z_integral = 0

        #self.x_pid = [1, 0.2, 0.01]
        #self.previous_x_error = 0
        self.safe_zone = [40, 65]
        #self.x_integral = 0
        
    

    def drone_status_callback(self, msg):
        self.drone_armed = not self.drone_armed
        rospy.logwarn(f"Drone state CHANGED! Drone is armed {self.drone_armed}")


    def image_callback(self, msg):
        #frame = self.br.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        #self.image_size = [frame.shape[1], frame.shape[0]]
        
        #flipped_frame = cv2.flip(frame, 1)
        #result = self.hand_detector.process(flipped_frame)

        #image = cv2.cvtColor(flipped_frame, cv2.COLOR_RGB2BGR)



        frame = self.br.imgmsg_to_cv2(msg)
        frame = cv2.resize(frame, (856, 480))
        self.image_size = [frame.shape[1], frame.shape[0]]

        flipped_frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result = self.hand_detector.process(rgb_frame)

        image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mf_coords, wrist_coords = self.get_hand_reference(hand, result)
                if mf_coords:
                    hand_center_x = int((mf_coords[0] + wrist_coords[0]) // 2)
                    hand_center_y = int((mf_coords[1] + wrist_coords[1]) // 2)

                    dist = wrist_coords[1] - mf_coords[1]

                    if self.drone_armed:
                        print("Working........")
                        self.track_hand(hand_center_x, hand_center_y, dist)
                        
                        if self.display:
                            x_min, y_min, x_max, y_max = self.get_hand_bbox(hand, result)
                            self.prepare_final_img(image, hand, hand_center_x, hand_center_y, x_min, y_min, x_max, y_max)
        else:
            self.send_stop_command()


    def track_hand(self, hand_center_x, hand_center_y, dist):

        '''/bebop/out_image
        Publish a geometry_msgs/Twist to cmd_vel topic 
        linear.x  (+)      Translate forward       (-)      Translate backward
        linear.y  (+)      Translate to left       (-)      Translate to right
        linear.z  (+)      Ascend                  (-)      Descend
        angular.z (+)      Rotate counter clockwise(-)      Rotate clockwise
        '''

        # YAW
        yaw_offset = self.image_size[0]//2
        yaw_error = (hand_center_x - yaw_offset) / yaw_offset # Normalized error between -1 and 1
        yaw_speed = yaw_pid.regulate(yaw_error,0)
        
        # Z Level
        z_offset = self.image_size[1]//2
        z_error = (hand_center_y - z_offset) / z_offset # Normalized error between -1 and 1
        z_speed = z_pid.regulate(z_error,0)
        # z_speed = - z_pid.regulate(z_error,0)
        
        # X - Forward Backward
        x_error = dist - np.mean(self.safe_zone) 
        normalized_x_error = self.normalize_x_error(x_error) 
        x_speed = x_pid.regulate(x_error,0)
        # x_speed = - x_pid.regulate(x_error,0)
        
        
        # yaw_proportional = self.yaw_pid[0]*yaw_error
        # self.yaw_integral = self.yaw_integral + self.yaw_pid[1] * yaw_error * self.delta_time
        # yaw_derivative = self.yaw_pid[2]*(yaw_error - self.previous_yaw_error)/self.delta_time
        # yaw_speed =  yaw_proportional  + yaw_derivative # + self.yaw_integral

        # yaw_speed = int(np.clip(yaw_speed, -yaw_limit, yaw_limit))
        # yaw_speed = yaw_speed / yaw_limit # Normalize between -1 and 1
        
        # z_proportional = self.z_pid[0]*z_error
        # self.z_integral = self.z_integral + self.yaw_pid[1] * z_error * self.delta_time
        # z_derivative = self.z_pid[2]*(z_error - self.previous_z_error)
        # z_speed = -(z_proportional + z_derivative) # self.z_integral

        

        # x_proportional = self.x_pid[0]*normalized_x_error
        # self.x_integral = self.x_integral + self.x_pid[1] * normalized_x_error * self.delta_time
        # x_derivative = self.x_pid[2]*(normalized_x_error - self.previous_x_error)
        # x_speed = -(x_proportional + x_derivative)# + self.x_integral 
        # x_speed = np.clip(x_speed, -0.3, 0.3)
        # x_speed = int(np.clip(fb_speed, -20000, 2500))
        # x_speed = ((x_speed - (-20000)) / (2500 - (-20000))) * (1 - (-1)) + (-1)
        
        if (dist > self.safe_zone[0] and dist < self.safe_zone[1]) or dist == 0:
            # stay stationary on x axis
            x_speed = 0
    
        flight_commands_msg = Twist()
        flight_commands_msg.linear.x = x_speed
        flight_commands_msg.linear.y = 0
        flight_commands_msg.linear.z = 0
        flight_commands_msg.angular.z = yaw_speed

        self.flight_pub.publish(flight_commands_msg)
        rospy.loginfo(f"SPEED: {z_speed}")
        # rospy.loginfo(f"DIST: {dist}")
        # rospy.loginfo(f"Normalized error {normalized_x_error}")
    
        self.previous_yaw_error = yaw_error
        self.previous_z_error = z_error
        self.previous_x_error = normalized_x_error
        

    def normalize_x_error(self, x_error):
        if self.lower_error_bound == 0:
            self.lower_error_bound = np.abs(25 - ((self.safe_zone[0] + self.safe_zone[1]) / 2))
            self.upper_error_bound = 120 - ((self.safe_zone[0] + self.safe_zone[1]) / 2)
        
        if x_error < 0:
            return x_error / self.lower_error_bound * 0.3
        else:
            return x_error / self.upper_error_bound



    def send_stop_command(self):
        self.flight_pub.publish(Twist())

        

    def prepare_final_img(self, image, hand, hand_center_x, hand_center_y, x_min, y_min, x_max, y_max):
        self.mp_draw.draw_landmarks(image, hand, self.mp_detector.HAND_CONNECTIONS,
                                    self.mp_draw.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                                    self.mp_draw.DrawingSpec(color=(180,100,100), thickness=2, circle_radius=2))

        error_palette = [(0, 255, 0), (0, 215, 255), (0, 140, 255), (0, 0 ,255)]
        euclidean_error = int(math.sqrt(np.abs(hand_center_x - int(self.image_size[0]//2))**2 + np.abs(hand_center_y - int(self.image_size[1]//2))**2))
        if self.max_euclidean_error == 0:
            self.max_euclidean_error = int(math.sqrt(np.abs(self.image_size[0] - int(self.image_size[0]//2))**2 + np.abs(self.image_size[1] - int(self.image_size[1]//2))**2))
        
        color = error_palette[int(str(euclidean_error/100)[0])]

        cv2.circle(image, (hand_center_x, hand_center_y), 3, (0, 255, 0), 2)
        cv2.circle(image, (int(self.image_size[0]/2), int(self.image_size[1]/2)), 3, (0, 255, 0), 2) # place a circle in the center of the image
        cv2.line(image, (hand_center_x, hand_center_y), (int(self.image_size[0]/2), int(self.image_size[1]/2)), color, 2)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # cv2.line(image, (mf_coords[0], mf_coords[1]), (wrist_coords[0], wrist_coords[1]), (0,0,255), 2)

        final_img = cv2.flip(image, 1)
        cv2.putText(final_img, f"Total Fingers: {self.right_counter + self.left_counter}", (10,25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        self.out_pub.publish(self.br.cv2_to_imgmsg(final_img))

    
    def get_bbox_area(self, x_min, y_min, x_max, y_max):
        return (x_max-x_min) * (y_max - y_min)


    def get_hand_reference(self, hands, results):
        coords = {'middle_finger_mcp':None, 'wrist':None}

        for classification in results.multi_handedness:
            # MIDDLE_FINGER_MCP
            coords['middle_finger_mcp'] = tuple(np.multiply(
                np.array((hands.landmark[self.mp_detector.HandLandmark.MIDDLE_FINGER_MCP].x, hands.landmark[self.mp_detector.HandLandmark.MIDDLE_FINGER_MCP].y)),
                self.image_size).astype(int))

            # WRIST
            coords['wrist'] = tuple(np.multiply(
                np.array((hands.landmark[self.mp_detector.HandLandmark.WRIST].x, hands.landmark[self.mp_detector.HandLandmark.WRIST].y)),
                self.image_size).astype(int))
        
            return coords['middle_finger_mcp'], coords['wrist']

        return None


    def get_hand_bbox(self, hands, results):
        coords = {'index_finger_mcp':None, 'middle_finger_mcp':None, 'pinky_mcp':None, 'wrist':None}

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

            # rospy.logwarn(hands.landmark[self.mp_detector.HandLandmark.WRIST].z)
        
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