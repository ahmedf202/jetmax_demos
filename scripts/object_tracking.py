#!/usr/bin/env python3
import math
import rospy
import time
import queue
import threading
from sensor_msgs.msg import Image
import cv2
import numpy as np
import hiwonder


image_queue = queue.Queue(maxsize=3)
ROS_NODE_NAME = 'object_tracking'
DEFAULT_X, DEFAULT_Y, DEFAULT_Z = 0, 138 + 8.14, 84 + 128.4
TARGET_PIXEL_X, TARGET_PIXEL_Y = 320, 240

class ObjectTracking:
    def __init__(self):
        self.image_sub = None
        self.heartbeat_timer = None
        self.lock = threading.RLock()
        self.servo_x = 500
        self.servo_y = 500

        self.face_x_pid = hiwonder.PID(0.09, 0.01, 0.015)
        self.face_z_pid = hiwonder.PID(0.16, 0.0, 0.0)

        self.color_x_pid = hiwonder.PID(0.07, 0.01, 0.0015)
        self.color_y_pid = hiwonder.PID(0.08, 0.008, 0.001)

        self.target_color_range = None
        self.target_color_name = None
        self.last_color_circle = None
        self.lost_target_count = 0

        self.is_running_color = False
        self.is_running_face = False

        self.fps = 0.0
        self.tic = time.time()

    def reset(self):
        self.tracking_face = None
        self.image_sub = None
        self.heartbeat_timer = None
        self.tracking_face_encoding = None
        self.no_face_count = 0
        self.servo_x = 500
        self.servo_y = 500

        self.last_color_circle = None
        self.lost_target_count = 0

        self.is_running_color = False
        self.is_running_face = False

        self.tic = time.time()


state = ObjectTracking()
jetmax = hiwonder.JetMax()
sucker = hiwonder.Sucker()


def init():
    state.reset()
    sucker.set_state(False)
    jetmax.go_home(1)


def image_proc():
    ros_image = image_queue.get(block=True)
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    image = color_tracking(image)

    toc = time.time()
    curr_fps = 1.0 / (state.tic - toc)
    state.fps = curr_fps if state.fps == 0.0 else (state.fps * 0.95 + curr_fps * 0.05)
    state.tic = toc
    rgb_image = image.tostring()
    ros_image.data = rgb_image
    image_pub.publish(ros_image)


def color_tracking(image):
    org_image = np.copy(image)
    image = cv2.resize(image, (320, 240))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # RGB to LAB space
    image = cv2.GaussianBlur(image, (5, 5), 5)

    with state.lock:
        target_color_range = state.target_color_range
        target_color_name = state.target_color_name

    if target_color_range is not None:
        mask = cv2.inRange(image, tuple(target_color_range['min']), tuple(target_color_range['max'])) # Binarization
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # Corrosion
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # Inflation
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # Find the contour
        contour_area = map(lambda c: (c, math.fabs(cv2.contourArea(c))), contours)  # Calculate the area of each contour
        contour_area = list(filter(lambda c: c[1] > 200, contour_area)) # Remove contours with too small an area
        circle = None
        if len(contour_area) > 0:
            if state.last_color_circle is None:
                contour, area = max(contour_area, key=lambda c_a: c_a[1])
                circle = cv2.minEnclosingCircle(contour)
            else:
                (last_x, last_y), last_r = state.last_color_circle
                circles = map(lambda c: cv2.minEnclosingCircle(c[0]), contour_area)
                circle_dist = list(map(lambda c: (c, math.sqrt(((c[0][0] - last_x) ** 2) + ((c[0][1] - last_y) ** 2))),
                                       circles))
                circle, dist = min(circle_dist, key=lambda c: c[1])
                if dist < 50:
                    circle = circle

        if circle is not None:
            state.lost_target_count = 0
            (c_x, c_y), c_r = circle
            c_x = hiwonder.misc.val_map(c_x, 0, 320, 0, 640)
            c_y = hiwonder.misc.val_map(c_y, 0, 240, 0, 480)
            c_r = hiwonder.misc.val_map(c_r, 0, 320, 0, 640)

            x = c_x - TARGET_PIXEL_X
            if abs(x) > 30:
                state.color_x_pid.SetPoint = 0
                state.color_x_pid.update(x)
                state.servo_x += state.color_x_pid.output
            else:
                state.color_x_pid.update(0)

            y = c_y - TARGET_PIXEL_Y
            if abs(y) > 30:
                state.color_y_pid.SetPoint = 0
                state.color_y_pid.update(y)
                state.servo_y -= state.color_y_pid.output
            else:
                state.color_y_pid.update(0)
            if state.servo_y < 350:
                state.servo_y = 350
            if state.servo_y > 650:
                state.servo_y = 650
            jetmax.set_servo(1, int(state.servo_x), duration=0.02)
            jetmax.set_servo(2, int(state.servo_y), duration=0.02)
            color_name = target_color_name.upper()
            org_image = cv2.circle(org_image, (int(c_x), int(c_y)), int(c_r), hiwonder.COLORS[color_name], 3)
            state.last_color_circle = circle
        else:
            state.lost_target_count += 1
            if state.lost_target_count > 15:
                state.lost_target_count = 0
                state.last_color_circle = None
    return org_image

def image_callback(ros_image):
    try:
        image_queue.put_nowait(ros_image)
    except queue.Full:
        pass

if __name__ == '__main__':
    rospy.init_node(ROS_NODE_NAME, anonymous=True)
    init()
    state.target_color_name = "red" # red, blue, green
    color_ranges = rospy.get_param('/color_range_list', {})
    state.target_color_range = color_ranges[state.target_color_name]
    
    image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)  # Subscribe to the camera feed
    image_pub = rospy.Publisher('/%s/image_result' % ROS_NODE_NAME, Image, queue_size=1)  # register result image pub

    hiwonder.buzzer.on()
    rospy.sleep(0.2)
    hiwonder.buzzer.off()


    while True:
        try:
            image_proc()
            if rospy.is_shutdown():
                break
        except KeyboardInterrupt:
            break
