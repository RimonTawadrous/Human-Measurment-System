import math
import cv2
import numpy as np
import math
from keypoint_config import *

POINTS_DICT = {}

SCALE_PERCENT = 100
image_name = "kisk1.2.jpg"
FILE_NAME = "kisk2_2.py"
PERSON_HEIGHT = 179




OUTPUT_DIRECTORY = "side_keypoints/"
INPUT_DIRECTORY = "images/"
iterator = 0
FILE_NAME = OUTPUT_DIRECTORY+FILE_NAME
image_name = INPUT_DIRECTORY+image_name


POINTS_COLOR = {0 : chest_front_color, 1 : chest_back_color, 2 : waist_front_color, 3 : waist_back_color, 4 : hip_front_color,
                5 : hip_back_color, 6 : trousers_front_color, 7 : trousers_back_color, 8 : Head_color, 9 : foot_color}

POINTS_NAMES = {0 : "chest_front_point", 1 : "chest_back_point", 2 : "waist_front_point", 3 : "waist_back_point", 4 : "hip_front_point", 
                5 : "hip_back_point", 6 : "trousers_front_point", 7 : "trousers_back_point", 8 : "head_point", 9 : "foot_color"}


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
        pass
    finally:
        with open(FILE_NAME, 'a') as the_file:
            ratio = eclidian_distance(POINTS_DICT[8],POINTS_DICT[9])/PERSON_HEIGHT
            the_file.write('side_image_name = "'+str(image_name)+'"\n')
            the_file.write('side_image_scale_percent = '+str(SCALE_PERCENT)+'\n')
            the_file.write('side_image_upper_ratio = '+str(ratio)+'\n')
            the_file.write('side_image_upper_ratio = '+str(ratio)+'\n')
            the_file.write('side_image_ratio = '+str(ratio)+'\n')
            the_file.write('\n')

            try:
                for point in POINTS_DICT:
                    the_file.write(POINTS_NAMES[point] +" = "+str(POINTS_DICT[point])+"\n")
            except :
                pass
            finally:
                the_file.write('\n')
                the_file.write("side_image_HIP_HIGHT = "+str(POINTS_DICT[4][1])+"\n")
            cv2.destroyAllWindows()

    
