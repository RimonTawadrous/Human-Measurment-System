import cv2
import numpy as np
import math
from keypoint_config import *
from side_keypoints.Rimon2_2 import *


def apply_filter_return_countoures(hsv_image, hsv_color_lower, hsv_color_upper, min_contor_area = 15, img = None , contor_label=""):
    filtered_image = cv2.inRange(hsv_image, hsv_color_lower, hsv_color_upper)
    kernal = np.ones((2,2),"uint8")
    filtered_image = cv2.dilate(filtered_image,kernal)
    (contors,hierarchy) = cv2.findContours(filtered_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    res = cv2.bitwise_and(img, img, mask = filtered_image)
    if True:
        i = 0
        for pic, contour in enumerate(contors):
            area = cv2.contourArea(contour)
            if(area>min_contor_area):
                x,y,w,h = cv2.boundingRect(contour)
                cv2.putText(img,str(contor_label),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255))
                i += 1
            return (res, filtered_image, (x+3,y+3), hierarchy)

    return (res, filtered_image, contors, hierarchy)

def eclidian_distance2(point1 , point2):
    global RATIO
    (x1, y1) = point1
    (x2, y2) = point2
    return (math.sqrt(((x1-x2)**2)+((y1-y2)**2)))/RATIO

if __name__ == "__main__":

    SIDE_KEYPOINTS = {"chest_front":[0,0], "chest_back":[0,0], "waist_front":[0,0], "waist_back":[0,0],"hip_front":[0,0],
    "hip_back":[0,0], "trousers_front":[0,0], "trousers_back":[0,0]}

    side_image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    
    # scaling 
    print("side_image.shape[1]",side_image.shape[1])
    print("side_image.shape[0]",side_image.shape[0])
    #calculate the 50 percent of original dimensions
    width = int(side_image.shape[1] * SCALE_PERCENT / 100)
    height = int(side_image.shape[0] * SCALE_PERCENT / 100)
    dsize = (width, height)
    # resize image
    side_image = cv2.resize(side_image, dsize)
    print("side_image.shape[1]",side_image.shape[1])
    print("side_image.shape[0]",side_image.shape[0])
    # end of scalling

    # chest front
    cv2.circle(side_image, chest_front_point, point_radius, chest_front_color, -1)
    # chest back
    cv2.circle(side_image, chest_back_point, point_radius, chest_back_color, -1)
    # waist front
    cv2.circle(side_image, waist_front_point, point_radius, waist_front_color, -1)
    # waist back
    cv2.circle(side_image, waist_back_point, point_radius, waist_back_color, -1)

    # hip front
    cv2.circle(side_image, hip_front_point, point_radius, hip_front_color, -1)
    # hip back
    cv2.circle(side_image, hip_back_point, point_radius, hip_back_color, -1)
    # trousers front
    cv2.circle(side_image, trousers_front_point, point_radius, trousers_front_color, -1)
    # trousers back
    cv2.circle(side_image, trousers_back_point, point_radius, trousers_back_color, -1)
    
    hsv = cv2.cvtColor(side_image, cv2.COLOR_BGR2HSV)
   
    # hsv hue set value  (color)
    (res,yellow,SIDE_KEYPOINTS["chest_front"],_) = apply_filter_return_countoures(hsv,chest_front_lower,chest_front_upper,min_contor_area= 30, img=side_image,contor_label="CF")
    (res,yellow,SIDE_KEYPOINTS["chest_back"],_) = apply_filter_return_countoures(hsv,chest_back_lower,chest_back_upper,min_contor_area= 30, img=side_image,contor_label="CB")
    
    (res,yellow,SIDE_KEYPOINTS["waist_front"],_) = apply_filter_return_countoures(hsv,waist_front_lower,waist_front_upper,min_contor_area= 30, img=side_image,contor_label="WF")
    (res,yellow,SIDE_KEYPOINTS["waist_back"],_) = apply_filter_return_countoures(hsv,waist_back_lower,waist_back_upper,min_contor_area= 30, img=side_image,contor_label="WB")
    
    (res,yellow,SIDE_KEYPOINTS["hip_front"],_) = apply_filter_return_countoures(hsv,hip_front_lower,hip_front_upper,min_contor_area= 30, img=side_image,contor_label="HF")
    (res,yellow,SIDE_KEYPOINTS["hip_back"],_) = apply_filter_return_countoures(hsv,hip_back_lower,hip_back_upper,min_contor_area= 30, img=side_image,contor_label="HB")
   
    (res,yellow,SIDE_KEYPOINTS["trousers_front"],_) = apply_filter_return_countoures(hsv,butt_front_lower,butt_front_upper,min_contor_area= 30, img=side_image,contor_label="TF")
    (res,yellow,SIDE_KEYPOINTS["trousers_back"],_) = apply_filter_return_countoures(hsv,butt_back_lower,butt_back_upper,min_contor_area= 30, img=side_image,contor_label="TB")


    print("chest = ",eclidian_distance2(SIDE_KEYPOINTS["chest_front"],SIDE_KEYPOINTS["chest_back"]))
    print("waist = ",eclidian_distance2(SIDE_KEYPOINTS["waist_front"],SIDE_KEYPOINTS["waist_back"]))
    print("hip = ",eclidian_distance2(SIDE_KEYPOINTS["hip_front"],SIDE_KEYPOINTS["hip_back"]))
    print("butt  = ",eclidian_distance2(SIDE_KEYPOINTS["trousers_front"],SIDE_KEYPOINTS["trousers_back"]))
       
    cv2.imshow("side_image", side_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
