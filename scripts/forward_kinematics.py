#!/usr/bin/env python3
import rospy
import time
import math
# links' lengths
L0 = 84.4
L1 = 8.14
L2 = 128.4
L3 = 138.0
L4 = 16.8
RAD_PER_DEG = math.pi / 180
DEG_PER_RAD = 180 / math.pi
DOUBLE_PI = math.pi * 2
def pulse_to_deg(pulses):
    pulse1, pulse2, pulse3 = pulses
    angle1 = pulse1 * 240 / 1000 # 0~1000 map to 0~240
    angle2 = pulse2 * -240 / 1000 + 210 # 0~1000 map to 210~-30
    angle3 = pulse3 * 240 / 1000 - 120 # 0~1000 map to -120~120
    return angle1, angle2, angle3
def deg_to_pulse(angles):
    angle1, angle2, angle3 = angles
    pulse1 = angle1 * 1000 / 240 # 0~240 map to 0~1000
    pulse2 = (angle2 - 210) * 1000 / -240 # 210~-30 map to 0~1000
    pulse3 = (angle3 - -120) * 1000 / 240 # -120~120 map to 0~1000
    return pulse1, pulse2, pulse3
def forward_kinematics(angles):
    """
    JetMax forward kinematics
    @param angles: active angles [rotate, angle left, angle right]
    @return: end point position (x, y, z)
    """
    alpha1, alpha2, alpha3 = [angle * RAD_PER_DEG for angle in angles]
    alpha1 += 150 * RAD_PER_DEG
    alpha1 = alpha1 - DOUBLE_PI if alpha1 > DOUBLE_PI else alpha1
    beta = alpha2 - alpha3
    side_beta = math.sqrt(L2 ** 2 + L3 ** 2 - 2 * L2 * L3 * math.cos(beta))
    cos_gamma = ((side_beta ** 2 + L2 ** 2) - L3 ** 2) / (2 * side_beta * L2)
    cos_gamma = cos_gamma if cos_gamma < 1 else 1
    gamma = math.acos(cos_gamma)
    alpha_gamma = math.pi - alpha2
    alpha = alpha_gamma - gamma
    z = side_beta * math.sin(alpha)
    r = math.sqrt(side_beta ** 2 - z ** 2)
    z = z + L0
    r = r + L1
    x = r * math.cos(alpha1)
    y = r * math.sin(alpha1)
    return x, y, z
if __name__ == '__main__':
    rospy.init_node('forward_kinematics', anonymous=True)
    # Input: joint1: 0~240, joint2: 210~-30 , joint3: -120~120 degs
    joints = 120, 150, 19 # home position
    #joints = 100, 40, 20
    #joints = 120, 150, 19
    # Output: end-effector position (x, y, z)
    ee_position = forward_kinematics(joints)
    print('End-effector Position: x= %d, y= %d, z= %d,' % (ee_position[0], ee_position[1],
    ee_position[2]))
    rospy.sleep(5)