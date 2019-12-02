import cv2
import numpy as np
import sys
from keypoint_config import *
image_hsv = None
pixel = (0,0,0) #RANDOM DEFAULT VALUE

ftypes = [
    ('JPG', '*.jpg;*.JPG;*.JPEG'), 
    ('PNG', '*.png;*.PNG'),
    ('GIF', '*.gif;*.GIF'),
]

def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        #HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        print(lower, upper)

        #A MONOCHROME MASK FOR GETTING A BETTER VISION OVER THE COLORS 
        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("Mask",image_mask)

def main():

    global image_hsv, pixel

    #OPEN DIALOG FOR READING THE IMAGE FILE
    img = cv2.imread("images/hatem.jpg", cv2.IMREAD_COLOR)
    

    # sholder outer
    cv2.circle(img, (150,208), point_radius, right_sholder_outer_color, -1)
    #right sholder inner
    cv2.circle(img, (228,181), point_radius, right_sholder_inner_color, -1)
    #left sholder inner
    cv2.circle(img, (304,184), point_radius, left_sholder_inner_color, -1)
    #left sholder outer
    cv2.circle(img, (380,208), point_radius, left_sholder_outer_color, -1)
    #right elbow outer
    cv2.circle(img, (124,374), point_radius, right_elbow_outer_color, -1)
    #right elbow inner
    cv2.circle(img, (163,371), point_radius, right_elbow_inner_color, -1)
    #left elbow inner
    cv2.circle(img, (387,367), point_radius, left_elbow_inner_color, -1)
    #left elbow outer
    cv2.circle(img, (431,364), point_radius, left_elbow_outer_color, -1)
    #right chest and arm meeting
    cv2.circle(img, (168,282), point_radius, right_chest_arm_meeting_color, -1)
    #left chest and arm meeting
    cv2.circle(img, (372,295), point_radius, left_chest_arm_meeting_color, -1)
    #right waist
    cv2.circle(img, (174,395), point_radius, right_waist_color, -1)
    #left waist
    cv2.circle(img, (366,390), point_radius, left_waist_color, -1)

 
    #right hip
    cv2.circle(img, (280,324), point_radius, right_hip_color, -1)
    #left hip
    cv2.circle(img, (372,318), point_radius, left_hip_color, -1)
    #stone of trousers
    cv2.circle(img, (325,340), point_radius, stone_of_trousers_color, -1)

    #right Knee outer 
    cv2.circle(img, (280,421), point_radius, right_Knee_outer_color, -1)
    #right Knee inner
    cv2.circle(img, (308,421), point_radius, right_Knee_inner_color, -1)
    #right ankle outer
    cv2.circle(img, (291,511), point_radius, right_ankle_outer_color, -1)
    #right ankle inner
    cv2.circle(img, (313,511), point_radius, right_ankle_inner_color, -1)

    #left Knee inner
    cv2.circle(img, (354,428), point_radius, left_Knee_inner_color, -1)
    #left Knee outer
    cv2.circle(img, (383,426), point_radius, left_Knee_outer_color, -1)
    #left Knee inner
    cv2.circle(img, (366,525), point_radius, left_ankle_inner_color, -1)
    #left Knee outer
    cv2.circle(img, (388,521), point_radius, left_ankle_outer_color, -1)    

    cv2.imshow("BGR",img)

    #CREATE THE HSV FROM THE BGR IMAGE
    image_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV",image_hsv)

    #CALLBACK FUNCTION
    cv2.setMouseCallback("HSV", pick_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
