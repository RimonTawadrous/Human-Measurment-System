import cv2
import numpy as np
import math
from math import pi
from math import sqrt
from keypoint_config import *
from front_keypoints.Rimon2_1 import *
from side_keypoints.Rimon2_2 import *


SIDE_KEYPOINTS = {"chest_front":[0,0], "chest_back":[0,0], "waist_front":[0,0], "waist_back":[0,0],"hip_front":[0,0],"hip_back":[0,0],
 "trousers_front":[0,0], "trousers_back":[0,0]}

FRONT_KEYPOINTS = {"left_sholder_outer":[0,0], "left_sholder_inner":[0,0], "right_sholder_inner":[0,0], "right_sholder_outer":[0,0],
    "left_elbow_inner":[0,0],"left_sholder_outer":[0,0], "right_elbow_inner":[0,0], "right_elbow_outer":[0,0], "left_chest_arm_meeting":[0,0],
    "right_chest_arm_meeting":[0,0], "left_chest":[0,0], "right_chest":[0,0], "left_waist":[0,0], "right_waist":[0,0], "left_hip":[0,0],
    "right_hip":[0,0], "stone_of_trousers":[0,0],"left_Knee_outer":[0,0], "left_Knee_inner":[0,0], "left_ankle_outer":[0,0], "left_ankle_inner":[0,0],
    "right_Knee_outer":[0,0],"right_Knee_inner":[0,0], "right_ankle_inner":[0,0], "right_ankle_outer":[0,0]}    

straight_measurements_dict = {"sholder":[], "front_chest":[], "front_waist":[], "front_hip":[], "under_arm":[],"hip_to_knee_left":[],"hip_to_knee_right":[],
    "waist_knee_left":[], "waist_knee_right":[], "waist_hip_left":[], "waist_hip_right":[], "left_knee_width":[], "right_knee_width":[], "knee_to_ankle_right":[],
    "knee_to_ankle_left":[], "T_stone_to_left_knee":[], "T_stone_to_right_knee":[], "side_chest":[], "side_waist":[], "side_hip":[], "natural_waist":[]}    

straight_measurements_average_dict = {"sholder":[], "front_chest":[], "front_waist":[], "front_hip":[], "under_arm":[],"hip_to_knee_left":[],"hip_to_knee_right":[],
    "waist_knee_left":[], "waist_knee_right":[], "waist_hip_left":[], "waist_hip_right":[], "left_knee_width":[], "right_knee_width":[], "knee_to_ankle_right":[],
    "knee_to_ankle_left":[], "T_stone_to_left_knee":[], "T_stone_to_right_knee":[], "side_chest":[], "side_waist":[], "side_hip":[], "natural_waist":[]}   

true_measurments_dict = {"chest_nipple":[], "waist":[], "hip":[], "natural_waist":[]} 

def rectangle_premiter(r1,r2):

    return 2*(r1+r2)

def circle_circumfrance(radius):
    return 2*pi*radius

def ellipse_perimeter(r1,r2):

    permeter1 = (2 * pi * sqrt( (r1**2 + r2**2) / (2 * 1.0) ) ) 
    permeter2 = pi*(3*(r1+r2)-sqrt((3*r1+r2)*(r1+3*r2)))

    permeter = ((permeter1*1.02 + permeter2)/2)
    return permeter2

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

def eclidian_distance(point1 , point2):
    global front_image_hip_hight, front_image_left_hip_pixel, front_image_right_hip_pixel
    (x1, y1) = point1
    (x2, y2) = point2
    result = 0
    mid = (front_image_left_hip_pixel[0]+front_image_right_hip_pixel[0]) / 2
    if y1 >= front_image_hip_hight and y2 >= front_image_hip_hight:
        return (math.sqrt(((x1-x2)**2)+((y1-y2)**2)))/front_image_lower_ratio
    elif y1 <= front_image_hip_hight and y2 <= front_image_hip_hight:
        return (math.sqrt(((x1-x2)**2)+((y1-y2)**2)))/front_image_upper_ratio
    else:
        if y1 > front_image_hip_hight:
            if x1 > mid:
                result += ((math.sqrt(((x1-front_image_left_hip_pixel[0])**2)+((y1-front_image_left_hip_pixel[1])**2)))/front_image_lower_ratio)
            else:
                result += ((math.sqrt(((x1-front_image_right_hip_pixel[0])**2)+((y1-front_image_right_hip_pixel[1])**2)))/front_image_lower_ratio)
        else:
            if x1 > mid:
                result += ((math.sqrt(((x1-front_image_left_hip_pixel[0])**2)+((y1-front_image_left_hip_pixel[1])**2)))/front_image_upper_ratio)
            else:
                result += ((math.sqrt(((x1-front_image_right_hip_pixel[0])**2)+((y1-front_image_right_hip_pixel[1])**2)))/front_image_upper_ratio)
        if y2 > front_image_hip_hight:
            if x2 > mid:
                result += ((math.sqrt(((front_image_left_hip_pixel[0]-x2)**2)+((front_image_left_hip_pixel[1]-y2)**2)))/front_image_lower_ratio)
            else:
                result += ((math.sqrt(((front_image_right_hip_pixel[0]-x2)**2)+((front_image_right_hip_pixel[1]-y2)**2)))/front_image_lower_ratio)
        else:
            if x2 > mid:
                result += ((math.sqrt(((front_image_left_hip_pixel[0]-x2)**2)+((front_image_left_hip_pixel[1]-y2)**2)))/front_image_upper_ratio)
            else:
                result += ((math.sqrt(((front_image_right_hip_pixel[0]-x2)**2)+((front_image_right_hip_pixel[1]-y2)**2)))/front_image_upper_ratio)

    return result

def eclidian_distance2(point1 , point2, ratio):
    (x1, y1) = point1
    (x2, y2) = point2
    return (math.sqrt(((x1-x2)**2)+((y1-y2)**2)))/ratio


def front_image_measurments(front_image):
    global FRONT_KEYPOINTS
    # sholder outer
    cv2.circle(front_image, right_sholder_outer_point, point_radius, right_sholder_outer_color, -1)
    #right sholder inner
    cv2.circle(front_image, right_sholder_inner_point, point_radius, right_sholder_inner_color, -1)
    #left sholder inner
    cv2.circle(front_image, left_sholder_inner_point, point_radius, left_sholder_inner_color, -1)
    #left sholder outer
    cv2.circle(front_image, left_sholder_outer_point, point_radius, left_sholder_outer_color, -1)

    #right elbow outer
    cv2.circle(front_image, right_elbow_outer_point, point_radius, right_elbow_outer_color, -1)
    #right elbow inner
    cv2.circle(front_image, right_elbow_inner_point, point_radius, right_elbow_inner_color, -1)
    #left elbow inner
    cv2.circle(front_image, left_elbow_inner_point, point_radius, left_elbow_inner_color, -1)
    #left elbow outer
    cv2.circle(front_image, left_elbow_outer_point, point_radius, left_elbow_outer_color, -1)

    #right chest and arm meeting
    cv2.circle(front_image, right_chest_arm_meeting_point, point_radius, right_chest_arm_meeting_color, -1)
    #left chest and arm meeting
    cv2.circle(front_image, left_chest_arm_meeting_point, point_radius, left_chest_arm_meeting_color, -1)
    
    #right chest
    cv2.circle(front_image, right_chest_point, point_radius, right_chest_color, -1)
    #left chest
    cv2.circle(front_image, left_chest_point, point_radius, left_chest_color, -1)

    #right waist
    cv2.circle(front_image, right_waist_point, point_radius, right_waist_color, -1)
    #left waist
    cv2.circle(front_image, left_waist_point, point_radius, left_waist_color, -1)

    #right hip
    cv2.circle(front_image, right_hip_point, point_radius, right_hip_color, -1)
    #left hip
    cv2.circle(front_image, left_hip_point, point_radius, left_hip_color, -1)
    
    #stone of trousers
    cv2.circle(front_image, stone_of_trousers_point, point_radius, stone_of_trousers_color, -1)

    #right Knee outer 
    cv2.circle(front_image, right_Knee_outer_point, point_radius, right_Knee_outer_color, -1)
    #right Knee inner
    cv2.circle(front_image, right_Knee_inner_point, point_radius, right_Knee_inner_color, -1)
    
    #left Knee inner
    cv2.circle(front_image, left_Knee_inner_point, point_radius, left_Knee_inner_color, -1)
    #left Knee outer
    cv2.circle(front_image, left_Knee_outer_point, point_radius, left_Knee_outer_color, -1)

    #right ankle outer
    cv2.circle(front_image, right_ankle_outer_point, point_radius, right_ankle_outer_color, -1)
    #right ankle inner
    cv2.circle(front_image, right_ankle_inner_point, point_radius, right_ankle_inner_color, -1)
    
    #left Knee inner
    cv2.circle(front_image, left_ankle_inner_point, point_radius, left_ankle_inner_color, -1)
    #left Knee outer
    cv2.circle(front_image, left_ankle_outer_point, point_radius, left_ankle_outer_color, -1)    
    
    hsv = cv2.cvtColor(front_image, cv2.COLOR_BGR2HSV)
    #hsv hue set value  (color)

    (res,yellow,FRONT_KEYPOINTS["left_sholder_outer"],_) = apply_filter_return_countoures(hsv,left_sholder_outer_lower,left_sholder_outer_upper,min_contor_area= 30, img=front_image,contor_label="LSO")
    (res,yellow,FRONT_KEYPOINTS["left_sholder_inner"],_) = apply_filter_return_countoures(hsv,left_sholder_inner_lower,left_sholder_inner_upper,min_contor_area= 30, img=front_image,contor_label="LSI")
    
    (res,yellow,FRONT_KEYPOINTS["right_sholder_inner"],_) = apply_filter_return_countoures(hsv,right_sholder_inner_lower,right_sholder_inner_upper,min_contor_area= 30, img=front_image,contor_label="RSI")
    (res,yellow,FRONT_KEYPOINTS["right_sholder_outer"],_) = apply_filter_return_countoures(hsv,right_sholder_outer_lower,right_sholder_outer_upper,min_contor_area= 30, img=front_image,contor_label="RSO")
    
    (res,yellow,FRONT_KEYPOINTS["left_elbow_inner"],_) = apply_filter_return_countoures(hsv,left_elbow_inner_lower,left_elbow_inner_upper,min_contor_area= 30, img=front_image,contor_label="LEI")
    (res,yellow,FRONT_KEYPOINTS["left_elbow_outer"],_) = apply_filter_return_countoures(hsv,left_elbow_outer_lower,left_elbow_outer_upper,min_contor_area= 30, img=front_image,contor_label="LEO ")

    (res,yellow,FRONT_KEYPOINTS["right_elbow_inner"],_) = apply_filter_return_countoures(hsv,right_elbow_inner_lower,right_elbow_inner_upper,min_contor_area= 30, img=front_image,contor_label="REI")
    (res,yellow,FRONT_KEYPOINTS["right_elbow_outer"],_) = apply_filter_return_countoures(hsv,right_elbow_outer_lower,right_elbow_outer_upper,min_contor_area= 30, img=front_image,contor_label="REO")
    
    (res,yellow,FRONT_KEYPOINTS["left_chest_arm_meeting"],_) = apply_filter_return_countoures(hsv,left_chest_arm_meeting_lower,left_chest_arm_meeting_upper,min_contor_area= 30, img=front_image,contor_label="LCAM")
    (res,yellow,FRONT_KEYPOINTS["right_chest_arm_meeting"],_) = apply_filter_return_countoures(hsv,right_chest_arm_meeting_lower,right_chest_arm_meeting_upper,min_contor_area= 30, img=front_image,contor_label="RCAM")
    
    (res,yellow,FRONT_KEYPOINTS["left_chest"],_) = apply_filter_return_countoures(hsv,left_chest_lower,left_chest_upper,min_contor_area= 30, img=front_image,contor_label="LC")
    (res,yellow,FRONT_KEYPOINTS["right_chest"],_) = apply_filter_return_countoures(hsv,right_chest_lower,right_chest_upper,min_contor_area= 30, img=front_image,contor_label="RC")
    
    (res,yellow,FRONT_KEYPOINTS["left_waist"],_) = apply_filter_return_countoures(hsv,left_waist_lower,left_waist_upper,min_contor_area= 30, img=front_image,contor_label="LW")
    (res,yellow,FRONT_KEYPOINTS["right_waist"],_) = apply_filter_return_countoures(hsv,right_waist_lower,right_waist_upper,min_contor_area= 30, img=front_image,contor_label="RW")
    
    (res,yellow,FRONT_KEYPOINTS["left_hip"],_) = apply_filter_return_countoures(hsv,left_hip_lower,left_hip_upper,min_contor_area= 30, img=front_image,contor_label="LH") 
    (res,yellow,FRONT_KEYPOINTS["right_hip"],_) = apply_filter_return_countoures(hsv,right_hip_lower,right_hip_upper,min_contor_area= 30, img=front_image,contor_label="RH")
   
    (res,yellow,FRONT_KEYPOINTS["stone_of_trousers"],_) = apply_filter_return_countoures(hsv,stone_of_trousers_lower,stone_of_trousers_upper,min_contor_area= 30, img=front_image,contor_label="SOT")
    
    (res,yellow,FRONT_KEYPOINTS["left_Knee_outer"],_) = apply_filter_return_countoures(hsv,left_Knee_outer_lower,left_Knee_outer_upper,min_contor_area= 30, img=front_image,contor_label="LKO")
    (res,yellow,FRONT_KEYPOINTS["left_Knee_inner"],_) = apply_filter_return_countoures(hsv,left_Knee_inner_lower,left_Knee_inner_upper,min_contor_area= 30, img=front_image,contor_label="LKI")
    
    (res,yellow,FRONT_KEYPOINTS["left_ankle_outer"],_) = apply_filter_return_countoures(hsv,left_ankle_outer_lower,left_ankle_outer_upper,min_contor_area= 30, img=front_image,contor_label="LAO")
    (res,yellow,FRONT_KEYPOINTS["left_ankle_inner"],_) = apply_filter_return_countoures(hsv,left_ankle_inner_lower,left_ankle_inner_upper,min_contor_area= 30, img=front_image,contor_label="LAI")

    (res,yellow,FRONT_KEYPOINTS["right_Knee_outer"],_) = apply_filter_return_countoures(hsv,right_Knee_outer_lower,right_Knee_outer_upper,min_contor_area= 30, img=front_image,contor_label="RKO")
    (res,yellow,FRONT_KEYPOINTS["right_Knee_inner"],_) = apply_filter_return_countoures(hsv,right_Knee_inner_lower,right_Knee_inner_upper,min_contor_area= 30, img=front_image,contor_label="RKI")
    
    (res,yellow,FRONT_KEYPOINTS["right_ankle_inner"],_) = apply_filter_return_countoures(hsv,right_ankle_inner_lower,right_ankle_inner_upper,min_contor_area= 30, img=front_image,contor_label="RAI")
    (res,yellow,FRONT_KEYPOINTS["right_ankle_outer"],_) = apply_filter_return_countoures(hsv,right_ankle_outer_lower,right_ankle_outer_upper,min_contor_area= 30, img=front_image,contor_label="RAO")

def side_image_measurments(side_image):
    global SIDE_KEYPOINTS
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


def resize_image(img,scale):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dsize = (width, height)
    # resize image
    img = cv2.resize(img, dsize)
    return img

if __name__ == "__main__":


    side_image = cv2.imread(side_image_name, cv2.IMREAD_COLOR)
    # scaling 
    
    print("side_image.shape[1]",side_image.shape[1])
    print("side_image.shape[0]",side_image.shape[0])
    #calculate the 50 percent of original dimensions
    side_image = resize_image(side_image, side_image_scale_percent)
    print("side_image.shape[1]",side_image.shape[1])
    print("side_image.shape[0]",side_image.shape[0])
    # end of scalling



    front_image = cv2.imread(front_image_name, cv2.IMREAD_COLOR)
    # scaling 
    print("front_image.shape[1]",front_image.shape[1])
    print("front_image.shape[0]",front_image.shape[0])
    #calculate the 50 percent of original dimensions
    front_image = resize_image(front_image, front_image_scale_percent)
    print("front_image.shape[1]",front_image.shape[1])
    print("front_image.shape[0]",front_image.shape[0])
    # end of scalling

    front_image_measurments(front_image)
    side_image_measurments(side_image)
   
    #side image measurments     #################################################################################################################################
    straight_measurements_dict["side_chest"].append( eclidian_distance2(SIDE_KEYPOINTS["chest_front"],SIDE_KEYPOINTS["chest_back"],side_image_ratio) )
    straight_measurements_dict["side_waist"].append( eclidian_distance2(SIDE_KEYPOINTS["waist_front"],SIDE_KEYPOINTS["waist_back"],side_image_ratio) )
    straight_measurements_dict["side_hip"].append( eclidian_distance2(SIDE_KEYPOINTS["hip_front"],SIDE_KEYPOINTS["hip_back"],side_image_ratio) )
    straight_measurements_dict["natural_waist"].append( eclidian_distance2(SIDE_KEYPOINTS["trousers_front"],SIDE_KEYPOINTS["trousers_back"],side_image_ratio) )
    #side image measurments end ##################################################################################################################################
    
   
    #front image measurments  ##################################################################################################################################
    straight_measurements_dict["sholder"].append( eclidian_distance(FRONT_KEYPOINTS["left_sholder_outer"],FRONT_KEYPOINTS["right_sholder_outer"]) )
    straight_measurements_dict["sholder"].append( eclidian_distance2(FRONT_KEYPOINTS["left_sholder_outer"],FRONT_KEYPOINTS["right_sholder_outer"],front_image_ratio) )

    straight_measurements_dict["front_chest"].append( eclidian_distance(FRONT_KEYPOINTS["left_chest"],FRONT_KEYPOINTS["right_chest"]) )
    straight_measurements_dict["front_chest"].append( eclidian_distance2(FRONT_KEYPOINTS["left_chest"],FRONT_KEYPOINTS["right_chest"],front_image_ratio) )

    straight_measurements_dict["front_waist"].append( eclidian_distance(FRONT_KEYPOINTS["left_waist"],FRONT_KEYPOINTS["right_waist"]) )
    straight_measurements_dict["front_waist"].append( eclidian_distance2(FRONT_KEYPOINTS["left_waist"],FRONT_KEYPOINTS["right_waist"],front_image_ratio) )

    straight_measurements_dict["front_hip"].append( eclidian_distance(FRONT_KEYPOINTS["left_hip"],FRONT_KEYPOINTS["right_hip"]) )
    straight_measurements_dict["front_hip"].append( eclidian_distance2(FRONT_KEYPOINTS["left_hip"],FRONT_KEYPOINTS["right_hip"],front_image_ratio) )

    straight_measurements_dict["under_arm"].append( eclidian_distance(FRONT_KEYPOINTS["left_chest_arm_meeting"],FRONT_KEYPOINTS["right_chest_arm_meeting"]) )
    straight_measurements_dict["under_arm"].append( eclidian_distance2(FRONT_KEYPOINTS["left_chest_arm_meeting"],FRONT_KEYPOINTS["right_chest_arm_meeting"],front_image_ratio) )

    straight_measurements_dict["hip_to_knee_left"].append( eclidian_distance(FRONT_KEYPOINTS["left_hip"],FRONT_KEYPOINTS["left_Knee_outer"]) )
    straight_measurements_dict["hip_to_knee_left"].append( eclidian_distance2(FRONT_KEYPOINTS["left_hip"],FRONT_KEYPOINTS["left_Knee_outer"],front_image_ratio) )

    straight_measurements_dict["hip_to_knee_right"].append( eclidian_distance(FRONT_KEYPOINTS["right_hip"],FRONT_KEYPOINTS["right_Knee_outer"]) )
    straight_measurements_dict["hip_to_knee_right"].append( eclidian_distance2(FRONT_KEYPOINTS["right_hip"],FRONT_KEYPOINTS["right_Knee_outer"],front_image_ratio) )

    straight_measurements_dict["waist_knee_left"].append( eclidian_distance(FRONT_KEYPOINTS["left_waist"],FRONT_KEYPOINTS["left_Knee_outer"]) )
    straight_measurements_dict["waist_knee_left"].append( eclidian_distance2(FRONT_KEYPOINTS["left_waist"],FRONT_KEYPOINTS["left_Knee_outer"],front_image_ratio) )

    straight_measurements_dict["waist_knee_right"].append( eclidian_distance(FRONT_KEYPOINTS["right_waist"],FRONT_KEYPOINTS["right_Knee_outer"]) )
    straight_measurements_dict["waist_knee_right"].append( eclidian_distance2(FRONT_KEYPOINTS["right_waist"],FRONT_KEYPOINTS["right_Knee_outer"],front_image_ratio) )

    straight_measurements_dict["waist_hip_left"].append( eclidian_distance(FRONT_KEYPOINTS["left_waist"],FRONT_KEYPOINTS["left_hip"]) )
    straight_measurements_dict["waist_hip_left"].append( eclidian_distance2(FRONT_KEYPOINTS["left_waist"],FRONT_KEYPOINTS["left_hip"],front_image_ratio) )

    straight_measurements_dict["waist_hip_right"].append( eclidian_distance(FRONT_KEYPOINTS["right_waist"],FRONT_KEYPOINTS["right_hip"]) )
    straight_measurements_dict["waist_hip_right"].append( eclidian_distance2(FRONT_KEYPOINTS["right_waist"],FRONT_KEYPOINTS["right_hip"],front_image_ratio) )
    
    straight_measurements_dict["left_knee_width"].append( eclidian_distance(FRONT_KEYPOINTS["left_Knee_outer"],FRONT_KEYPOINTS["left_Knee_inner"]) )
    straight_measurements_dict["left_knee_width"].append( eclidian_distance2(FRONT_KEYPOINTS["left_Knee_outer"],FRONT_KEYPOINTS["left_Knee_inner"],front_image_ratio) )

    straight_measurements_dict["right_knee_width"].append( eclidian_distance(FRONT_KEYPOINTS["right_Knee_inner"],FRONT_KEYPOINTS["right_Knee_outer"]) )
    straight_measurements_dict["right_knee_width"].append( eclidian_distance2(FRONT_KEYPOINTS["right_Knee_inner"],FRONT_KEYPOINTS["right_Knee_outer"],front_image_ratio) )

    straight_measurements_dict["knee_to_ankle_right"].append( eclidian_distance(FRONT_KEYPOINTS["right_Knee_outer"],FRONT_KEYPOINTS["right_ankle_outer"]) )
    straight_measurements_dict["knee_to_ankle_right"].append( eclidian_distance2(FRONT_KEYPOINTS["right_Knee_outer"],FRONT_KEYPOINTS["right_ankle_outer"],front_image_ratio) )
  
    straight_measurements_dict["knee_to_ankle_left"].append( eclidian_distance(FRONT_KEYPOINTS["left_Knee_outer"],FRONT_KEYPOINTS["left_ankle_outer"]) )
    straight_measurements_dict["knee_to_ankle_left"].append( eclidian_distance2(FRONT_KEYPOINTS["left_Knee_outer"],FRONT_KEYPOINTS["left_ankle_outer"],front_image_ratio) )

    straight_measurements_dict["T_stone_to_left_knee"].append( eclidian_distance(FRONT_KEYPOINTS["stone_of_trousers"],FRONT_KEYPOINTS["left_Knee_inner"]) )
    straight_measurements_dict["T_stone_to_left_knee"].append( eclidian_distance2(FRONT_KEYPOINTS["stone_of_trousers"],FRONT_KEYPOINTS["left_Knee_inner"],front_image_ratio) )

    straight_measurements_dict["T_stone_to_right_knee"].append( eclidian_distance(FRONT_KEYPOINTS["stone_of_trousers"],FRONT_KEYPOINTS["right_Knee_inner"]) )
    straight_measurements_dict["T_stone_to_right_knee"].append( eclidian_distance2(FRONT_KEYPOINTS["stone_of_trousers"],FRONT_KEYPOINTS["right_Knee_inner"],front_image_ratio) )
    #front image measurments end  ###############################################################################################################################
    
    # print staight measurments
    print("#"*40)
    for  key in straight_measurements_dict:
        print (key.replace("_"," "))
        measurement_sum = 0
        i = 0
        for measurement in straight_measurements_dict[key]:
            measurement_sum += measurement
            i += 1
            print(measurement,"====>  ",end="")
        straight_measurements_average_dict [key] = measurement_sum/i
        print(straight_measurements_average_dict [key])
        print("#"*40)

    # chest perimeter
    r1 = straight_measurements_average_dict["front_chest"]/2
    r2 = straight_measurements_average_dict["side_chest"]/2
    true_measurments_dict["chest_nipple"].append(rectangle_premiter(r1,r2))
    true_measurments_dict["chest_nipple"].append(ellipse_perimeter(r1,r2))
    # end of chest

    # chest perimeter
    r1 = straight_measurements_average_dict["front_waist"]/2
    r2 = straight_measurements_average_dict["side_waist"]/2
    true_measurments_dict["waist"].append(rectangle_premiter(r1,r2))
    true_measurments_dict["waist"].append(ellipse_perimeter(r1,r2))

    # end of chest

    # chest perimeter
    r1 = straight_measurements_average_dict["front_hip"]/2
    r2 = straight_measurements_average_dict["side_hip"]/2
    true_measurments_dict["hip"].append(rectangle_premiter(r1,r2))
    true_measurments_dict["hip"].append(ellipse_perimeter(r1,r2))
    # end of chest

    #print true measurments
    print("#"*40)
    for  key in true_measurments_dict:
        print (key.replace("_"," "))
        for measurement in true_measurments_dict[key]:
            print(measurement,"====>  ",end="")
        print()
        print("#"*40)


    cv2.imshow("front_image", front_image)   
    cv2.imshow("side_image", side_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
