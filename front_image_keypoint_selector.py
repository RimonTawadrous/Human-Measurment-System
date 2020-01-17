import math
import cv2
import numpy as np
import math
from keypoint_config import *

POINTS_DICT = {}

SCALE_PERCENT = 100
FILE_NAME = "kishk2_1.py"
image_name = "kishk2.1.jpg"
PERSON_HEIGHT = 179




OUTPUT_DIRECTORY = "front_keypoints/"
INPUT_DIRECTORY = "images/"
iterator = 0
FILE_NAME = OUTPUT_DIRECTORY+FILE_NAME
image_name = INPUT_DIRECTORY+image_name

POINTS_COLOR = {0 : right_sholder_outer_color, 1 : right_sholder_inner_color, 2 : left_sholder_inner_color, 3 : left_sholder_outer_color, 4 : right_elbow_outer_color,
                5 : right_elbow_inner_color, 6 : left_elbow_inner_color, 7 : left_elbow_outer_color, 8 : right_chest_arm_meeting_color, 9 : left_chest_arm_meeting_color,
                10 : right_chest_color, 11 : left_chest_color, 12 : right_waist_color, 13 : left_waist_color, 14 : right_hip_color, 15 : left_hip_color, 16 : stone_of_trousers_color, 
                17 : right_Knee_outer_color, 18 : right_Knee_inner_color, 19 : left_Knee_inner_color, 20 : left_Knee_outer_color, 21 : right_ankle_outer_color, 
                22 : right_ankle_inner_color, 23 : left_ankle_inner_color, 24 : left_ankle_outer_color, 25 : Head_color, 26:foot_color }

POINTS_NAMES = {0 : "right_sholder_outer_point", 1 : "right_sholder_inner_point", 2 : "left_sholder_inner_point", 3 : "left_sholder_outer_point", 4 : "right_elbow_outer_point", 
                5 : "right_elbow_inner_point", 6 : "left_elbow_inner_point", 7 : "left_elbow_outer_point", 8 : "right_chest_arm_meeting_point", 9 : "left_chest_arm_meeting_point",
                10 : "right_chest_point", 11 : "left_chest_point", 12 : "right_waist_point", 13 : "left_waist_point", 14 : "right_hip_point", 15 : "left_hip_point", 16 : "stone_of_trousers_point", 
                17 : "right_Knee_outer_point", 18 : "right_Knee_inner_point", 19 : "left_Knee_inner_point", 20 : "left_Knee_outer_point", 21 : "right_ankle_outer_point", 
                22 : "right_ankle_inner_point", 23 : "left_ankle_inner_point", 24 : "left_ankle_outer_point", 25 : "head_point", 26 : "foot_color" }


def eclidian_distance(point1 , point2):
    (x1, y1) = point1
    (x2, y2) = point2
    return (math.sqrt(((x1-x2)**2)+((y1-y2)**2)))

def draw_circle(event,x,y,flags,param):
    global POINTS_DICT, iterator
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS_DICT[iterator] = (x,y)
        iterator += 1
        if iterator in POINTS_NAMES:
            print("now select ---> ",POINTS_NAMES[iterator],(x,y))
        else:
            print("one left click more to save")
    if event == cv2.EVENT_MBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN :
        iterator -= 1
        print("now select ---> ",POINTS_NAMES[iterator])
        del POINTS_DICT[iterator]

if __name__ == "__main__":

    img = cv2.imread(image_name, cv2.IMREAD_COLOR)
    print("now select ---> ",POINTS_NAMES[iterator])

    # scaling 
    width = int(img.shape[1] * SCALE_PERCENT / 100)
    height = int(img.shape[0] * SCALE_PERCENT / 100)
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
            ratio = eclidian_distance(POINTS_DICT[25],POINTS_DICT[26])/PERSON_HEIGHT
            the_file.write('front_image_name = "'+str(image_name)+'"\n')
            the_file.write('front_image_scale_percent = '+str(SCALE_PERCENT)+'\n')
            the_file.write('front_image_upper_ratio = '+str(ratio)+'\n')
            the_file.write('front_image_lower_ratio = '+str(ratio)+'\n')
            the_file.write('front_image_ratio = '+str(ratio)+'\n')
            the_file.write('\n')

            try:
                for point in POINTS_DICT:
                    the_file.write(POINTS_NAMES[point] +" = "+str(POINTS_DICT[point])+"\n")
            except :
                pass
            finally:
                the_file.write('\n')
                the_file.write("front_image_hip_hight = "+str(POINTS_DICT[14][1])+"\n")
                the_file.write("front_image_right_hip_pixel ="+str(POINTS_DICT[14])+"\n")
                the_file.write("front_image_left_hip_pixel ="+str(POINTS_DICT[15])+"\n")
            cv2.destroyAllWindows()

    
