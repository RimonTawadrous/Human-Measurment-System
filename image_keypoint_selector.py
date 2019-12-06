import math
import cv2
import numpy as np
import math
from keypoint_config import *



POINTS_DICT = {}
iterator = 0
scale_percent = 20
FILE_NAME = "test.py"
POINTS_COLOR = {0 : right_sholder_outer_color, 1 : right_sholder_inner_color, 2 : left_sholder_inner_color, 3 : left_sholder_outer_color, 4 : right_elbow_outer_color,
                5 : right_elbow_inner_color, 6 : left_elbow_inner_color, 7 : left_elbow_outer_color, 8 : right_chest_arm_meeting_color, 9 : left_chest_arm_meeting_color,
                10 : right_chest_color, 11 : left_chest_color, 12 : right_waist_color, 13 : left_waist_color, 14 : right_hip_color, 15 : left_hip_color, 16 : stone_of_trousers_color, 
                17 : right_Knee_outer_color, 18 : right_Knee_inner_color, 19 : left_Knee_inner_color, 20 : left_Knee_outer_color, 21 : right_ankle_outer_color, 
                22 : right_ankle_inner_color, 23 : left_ankle_inner_color, 24 : left_ankle_outer_color }

POINTS_NAMES = {0 : "right_sholder_outer_point", 1 : "right_sholder_inner_point", 2 : "left_sholder_inner_point", 3 : "left_sholder_outer_point", 4 : "right_elbow_outer_point", 
                5 : "right_elbow_inner_point", 6 : "left_elbow_inner_point", 7 : "left_elbow_outer_point", 8 : "right_chest_arm_meeting_point", 9 : "left_chest_arm_meeting_point",
                10 : "right_chest_point", 11 : "left_chest_point", 12 : "right_waist_point", 13 : "left_waist_point", 14 : "right_hip_point", 15 : "left_hip_point", 16 : "stone_of_trousers_point", 
                17 : "right_Knee_outer_point", 18 : "right_Knee_inner_point", 19 : "left_Knee_inner_point", 20 : "left_Knee_outer_point", 21 : "right_ankle_outer_point", 
                22 : "right_ankle_inner_point", 23 : "left_ankle_inner_point", 24 : "left_ankle_outer_point" }


def draw_circle(event,x,y,flags,param):
    global POINTS_DICT, iterator
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS_DICT[iterator] = (x,y)
        iterator += 1
        print("now select ---> ",POINTS_NAMES[iterator])
    if event == cv2.EVENT_MBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN :
        iterator -= 1
        print("now select ---> ",POINTS_NAMES[iterator])
        del POINTS_DICT[iterator]

if __name__ == "__main__":
    image_name = "images/Rimon1.1.jpg"
    img = cv2.imread(image_name, cv2.IMREAD_COLOR)
    print("now select ---> ",POINTS_NAMES[iterator])

    # scaling 
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    # resize image
    img = cv2.resize(img, dsize)
    # end of scalling
    true_image = img.copy()
    try:
        while True:
            img = true_image.copy()
            for point in POINTS_DICT:
                cv2.circle(img, POINTS_DICT[point], point_radius, POINTS_COLOR[point], -1)
            cv2.imshow("img", img)
            cv2.setMouseCallback('img',draw_circle)
            cv2.waitKey(1)
    except:
        print("yess", iterator)
        pass
    finally:
        with open(FILE_NAME, 'a') as the_file:
            the_file.write('image_name = "'+image_name+'"\nscale_percent = 20\nUPPER_RATIO = 0\nLOWER_RATIO = 0\n')
            try:
                for point in POINTS_DICT:
                    the_file.write(POINTS_NAMES[point] +" = "+str(POINTS_DICT[point])+"\n")
            except :
                pass
            finally:
                the_file.write("HIP_HIGHT = 377\n")
                the_file.write("RIGHT_HIP_PIXEL = (143, 377)\n")
                the_file.write("LEFT_HIP_PIXEL = (250, 377)\n")
            cv2.destroyAllWindows()

    