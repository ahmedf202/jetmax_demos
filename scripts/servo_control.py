#!/usr/bin/env python3
import os
import sys
import rospy
import hiwonder # expamsion board interface driver
if __name__ == '__main__':
    rospy.init_node('servo_control', log_level=rospy.INFO)
    #go_home(duration=2)
    # ========= Camera Gimble Servo [pwm_servo 1] ============
    # Rotate servo coneected to channel pwm_servo 1 with angle 90 degrees in 1 second
    hiwonder.pwm_servo1.set_position(180, 1)
    print('================== Joint 1 [rotate/base Servo]===================')
    # Move Serial servo with ID 1 from the current position to position 180 deg in 3000 milliseconds
    id_ = 1 # Serial_servo ID [Joint 1]
    angle1_cmd = 150 # 0~240
    # Convert 180 deg to the cooresponding value in digital as the motor driver accepts vlaue (0 to 1000) this is should be mapped to 0:240 (base joint constraints)
    pulse1_cmd = int(angle1_cmd/240*1000) # 0~240 map to 0~1000
    duration_ms = 3000
    hiwonder.serial_servo.set_position(id_, pulse1_cmd, duration_ms)
    # wait 1 second more the required duration
    rospy.sleep(duration_ms/1000 + 1)
    # read and print measured angle value of the servo of ID 1 [Joint 1]
    pulse1_measured = hiwonder.serial_servo.read_position(id_)
    print('pulse2_measured = %d' % pulse1_measured)
    angle1_measured = int(pulse1_measured*240/1000) # 0~1000 map to 0~240
    print('Measured Angle of joint 1 = %d' % angle1_measured)
    # ================ Joint 2 (Left) [range: 210~-30 degrees] ==========
    print('================== Joint 2 [Left Servo]===================')
    # Move Serial servo with ID 1 from the current position to position 180 deg in 3000 milliseconds
    id_ = 2 # [Joint 2] Serial_servo ID
    angle2_cmd = 120 # 30~160
    pulse2_cmd = int((angle2_cmd - 210) * 1000 / -240) # 210~-30 map to 0~1000
    print('pulse2_cmd = %d' % pulse2_cmd)
    # ================ Joint 3 (Right) [range: -120~120 degrees] ==========
    print('================== Joint 3 [Right Servo] ===================')
    # Move Serial servo with ID 1 from the current position to position 180 deg in 3000milliseconds
    id_ = 3 # [Joint 2] Serial_servo ID
    angle3_cmd = -20 # -10~25 -10~80
    pulse3_cmd = int((angle3_cmd - -120) * 1000 / 240 ) # -120~120 map to 0~1000
    print('pulse3_cmd = %d' % pulse3_cmd)
    duration_ms = 3000
    hiwonder.serial_servo.set_position(id_, pulse3_cmd, duration_ms)
    # wait 1 second more the required duration
    rospy.sleep(duration_ms/1000 + 1)
    # read and print measured angle value of the servo of ID 2
    pulse3_measured = hiwonder.serial_servo.read_position(id_)
    print('pulse3_measured = %d' % pulse3_measured)
    angle3_measured = int(pulse3_measured * 240 / 1000 - 120 ) # 0~1000 map to -120~120
    print('Measured Angle of joint 3 = %d' % angle3_measured)
        # ====================== Sucker ========================# suck
    hiwonder.motor2.set_speed(0) # Close the vent valve
    hiwonder.motor1.set_speed(100) # Turn on the air pump
    rospy.sleep(4)
    #release
    hiwonder.motor1.set_speed(0) # Turn off the air pump
    hiwonder.motor2.set_speed(100) # Open the vent valve
