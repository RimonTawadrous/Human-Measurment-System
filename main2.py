import cv2
import numpy as np
import math
from keypoint_config import *

def eclidian_distance(x1, y1, x2, y2):
    return math.sqrt(((x1-x2)**2)+((y1-y2)**2))

if __name__ == "__main__":
    img = cv2.imread("images/hatem.jpg", cv2.IMREAD_COLOR)
    

    #left sholder outer
    cv2.circle(img, (150,208), point_radius, left_sholder_outer_color, -1)
    #left sholder inner
    cv2.circle(img, (228,181), point_radius, left_sholder_inner_color, -1)
    #right sholder inner
    cv2.circle(img, (304,184), point_radius, right_sholder_inner_color, -1)
    #right sholder outer
    cv2.circle(img, (380,208), point_radius, right_sholder_outer_color, -1)
    #left elbow outer
    cv2.circle(img, (124,374), point_radius, left_elbow_outer_color, -1)
    #left elbow inner
    cv2.circle(img, (163,371), point_radius, left_elbow_inner_color, -1)
    #right elbow inner
    cv2.circle(img, (387,367), point_radius, right_elbow_inner_color, -1)
    #right elbow outer
    cv2.circle(img, (431,364), point_radius, right_elbow_outer_color, -1)
    #left chest and arm meeting
    cv2.circle(img, (168,282), point_radius, left_chest_arm_meeting_color, -1)
    #right chest and arm meeting
    cv2.circle(img, (372,295), point_radius, right_chest_arm_meeting_color, -1)
    #left waist
    cv2.circle(img, (174,395), point_radius, left_waist_color, -1)
    #right waist
    cv2.circle(img, (366,390), point_radius, right_waist_color, -1)
  
 
    #left hip
    cv2.circle(img, (280,324), point_radius, left_hip_color, -1)
    #right hip
    cv2.circle(img, (372,318), point_radius, right_hip_color, -1)
    #stone of trousers
    cv2.circle(img, (325,340), point_radius, stone_of_trousers_color, -1)

    #left Knee outer 
    cv2.circle(img, (280,421), point_radius, left_Knee_outer_color, -1)
    #left Knee inner
    cv2.circle(img, (308,421), point_radius, left_Knee_inner_color, -1)
    #left ankle outer
    cv2.circle(img, (291,511), point_radius, left_ankle_outer_color, -1)
    #left ankle inner
    cv2.circle(img, (313,511), point_radius, left_ankle_inner_color, -1)

    #right Knee inner
    cv2.circle(img, (354,428), point_radius, right_Knee_inner_color, -1)
    #right Knee outer
    cv2.circle(img, (383,426), point_radius, right_Knee_outer_color, -1)
    #right Knee inner
    cv2.circle(img, (366,525), point_radius, right_Knee_inner_color, -1)
    #right Knee outer
    cv2.circle(img, (388,521), point_radius, right_Knee_outer_color, -1)    


    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #hsv hue set value  (color)

    yellow = cv2.inRange(hsv, right_sholder_outer_lower, right_sholder_outer_upper)

    kernal = np.ones((2,2),"uint8")
    yellow = cv2.dilate(yellow,kernal)
    res = cv2.bitwise_and(img, img, mask = yellow)


    (contors,hierarchy) = cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for pic, contour in enumerate(contors):
        area = cv2.contourArea(contour)
        print(area)
        if(area>30):
            x,y,w,h = cv2.boundingRect(contour)
            print(x+3," ",y+3)
            cv2.putText(img,str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255))
            i += 1

    cv2.imshow("hsv", hsv)
    cv2.imshow("img", img)
    cv2.imshow("mask", yellow)
    cv2.imshow("Result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
