import cv2
import numpy as np
import math
from keypoint_config import *
from front_keypoints.Rimon2_1 import *

FRONT_KEYPOINTS = {"left_sholder_outer":[0,0], "left_sholder_inner":[0,0], "right_sholder_inner":[0,0], "right_sholder_outer":[0,0],
    "left_elbow_inner":[0,0],"left_sholder_outer":[0,0], "right_elbow_inner":[0,0], "right_elbow_outer":[0,0], "left_chest_arm_meeting":[0,0],
    "right_chest_arm_meeting":[0,0], "left_chest":[0,0], "right_chest":[0,0], "left_waist":[0,0], "right_waist":[0,0], "left_hip":[0,0],
    "right_hip":[0,0], "stone_of_trousers":[0,0],"left_Knee_outer":[0,0], "left_Knee_inner":[0,0], "left_ankle_outer":[0,0], "left_ankle_inner":[0,0],
    "right_Knee_outer":[0,0],"right_Knee_inner":[0,0], "right_ankle_inner":[0,0], "right_ankle_outer":[0,0]}    

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
                # print(contor_label,x+3,y+3)
                cv2.putText(img,str(contor_label),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255))
                i += 1
        return (res, filtered_image, (x+3,y+3), hierarchy)

    return (res, filtered_image, contors, hierarchy)

def eclidian_distance(point1 , point2):
    global HIP_PIXEL, LEFT_HIP_PIXEL, RIGHT_HIP_PIXEL
    (x1, y1) = point1
    (x2, y2) = point2
    result = 0
    mid = (LEFT_HIP_PIXEL[0]+RIGHT_HIP_PIXEL[0]) / 2
    if y1 >= HIP_HIGHT and y2 >= HIP_HIGHT:
        return (math.sqrt(((x1-x2)**2)+((y1-y2)**2)))/LOWER_RATIO
    elif y1 <= HIP_HIGHT and y2 <= HIP_HIGHT:
        return (math.sqrt(((x1-x2)**2)+((y1-y2)**2)))/UPPER_RATIO
    else:
        if y1 > HIP_HIGHT:
            if x1 > mid:
                result += ((math.sqrt(((x1-LEFT_HIP_PIXEL[0])**2)+((y1-LEFT_HIP_PIXEL[1])**2)))/LOWER_RATIO)
            else:
                result += ((math.sqrt(((x1-RIGHT_HIP_PIXEL[0])**2)+((y1-RIGHT_HIP_PIXEL[1])**2)))/LOWER_RATIO)
        else:
            if x1 > mid:
                result += ((math.sqrt(((x1-LEFT_HIP_PIXEL[0])**2)+((y1-LEFT_HIP_PIXEL[1])**2)))/UPPER_RATIO)
            else:
                result += ((math.sqrt(((x1-RIGHT_HIP_PIXEL[0])**2)+((y1-RIGHT_HIP_PIXEL[1])**2)))/UPPER_RATIO)
        if y2 > HIP_HIGHT:
            if x2 > mid:
                result += ((math.sqrt(((LEFT_HIP_PIXEL[0]-x2)**2)+((LEFT_HIP_PIXEL[1]-y2)**2)))/LOWER_RATIO)
            else:
                result += ((math.sqrt(((RIGHT_HIP_PIXEL[0]-x2)**2)+((RIGHT_HIP_PIXEL[1]-y2)**2)))/LOWER_RATIO)
        else:
            if x2 > mid:
                result += ((math.sqrt(((LEFT_HIP_PIXEL[0]-x2)**2)+((LEFT_HIP_PIXEL[1]-y2)**2)))/UPPER_RATIO)
            else:
                result += ((math.sqrt(((RIGHT_HIP_PIXEL[0]-x2)**2)+((RIGHT_HIP_PIXEL[1]-y2)**2)))/UPPER_RATIO)

    return result

def eclidian_distance2(point1 , point2):
    (x1, y1) = point1
    (x2, y2) = point2
    return (math.sqrt(((x1-x2)**2)+((y1-y2)**2)))/RATIO

if __name__ == "__main__":

    front_image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    
    # scaling 
    print("front_image.shape[1]",front_image.shape[1])
    print("front_image.shape[0]",front_image.shape[0])
    #calculate the 50 percent of original dimensions
    width = int(front_image.shape[1] * SCALE_PERCENT / 100)
    height = int(front_image.shape[0] * SCALE_PERCENT / 100)
    dsize = (width, height)
    # resize image
    front_image = cv2.resize(front_image, dsize)
    print("front_image.shape[1]",front_image.shape[1])
    print("front_image.shape[0]",front_image.shape[0])
    # end of scalling


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

    print("sholder = ",eclidian_distance(FRONT_KEYPOINTS["right_sholder_outer"],FRONT_KEYPOINTS["left_sholder_outer"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["right_sholder_outer"],FRONT_KEYPOINTS["left_sholder_outer"]))
    print("waist = ",eclidian_distance(FRONT_KEYPOINTS["right_waist"],FRONT_KEYPOINTS["left_waist"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["right_waist"],FRONT_KEYPOINTS["left_waist"]))
    print("hip = ",eclidian_distance(FRONT_KEYPOINTS["left_hip"],FRONT_KEYPOINTS["right_hip"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["left_hip"],FRONT_KEYPOINTS["right_hip"]))
    print("under arm  = ",eclidian_distance(FRONT_KEYPOINTS["left_chest_arm_meeting"],FRONT_KEYPOINTS["right_chest_arm_meeting"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["left_chest_arm_meeting"],FRONT_KEYPOINTS["right_chest_arm_meeting"]))
    print("hip knee left  = ",eclidian_distance(FRONT_KEYPOINTS["left_hip"],FRONT_KEYPOINTS["left_Knee_outer"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["left_hip"],FRONT_KEYPOINTS["left_Knee_outer"]))
    print("hip knee right  = ",eclidian_distance(FRONT_KEYPOINTS["right_hip"],FRONT_KEYPOINTS["right_Knee_outer"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["right_hip"],FRONT_KEYPOINTS["right_Knee_outer"]))
    print("waist knee left  = ",eclidian_distance(FRONT_KEYPOINTS["left_waist"],FRONT_KEYPOINTS["left_Knee_outer"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["left_waist"],FRONT_KEYPOINTS["left_Knee_outer"]))
    print("waist knee right  = ",eclidian_distance(FRONT_KEYPOINTS["right_waist"],FRONT_KEYPOINTS["right_Knee_outer"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["right_waist"],FRONT_KEYPOINTS["right_Knee_outer"]))
    print("waist hip left  = ",eclidian_distance(FRONT_KEYPOINTS["left_waist"],FRONT_KEYPOINTS["left_hip"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["left_waist"],FRONT_KEYPOINTS["left_hip"]))
    print("waist hip right  = ",eclidian_distance(FRONT_KEYPOINTS["right_waist"],FRONT_KEYPOINTS["right_hip"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["right_waist"],FRONT_KEYPOINTS["right_hip"]))
    print("knee left  = ",eclidian_distance(FRONT_KEYPOINTS["left_Knee_outer"],FRONT_KEYPOINTS["left_Knee_inner"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["left_Knee_outer"],FRONT_KEYPOINTS["left_Knee_inner"]))
    print("knee right  = ",eclidian_distance(FRONT_KEYPOINTS["right_Knee_inner"],FRONT_KEYPOINTS["right_Knee_outer"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["right_Knee_inner"],FRONT_KEYPOINTS["right_Knee_outer"]))
    print("knee to ankle right = ",eclidian_distance(FRONT_KEYPOINTS["right_Knee_outer"],FRONT_KEYPOINTS["right_ankle_outer"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["right_Knee_outer"],FRONT_KEYPOINTS["right_ankle_outer"]))
    print("knee to ankle left  = ",eclidian_distance(FRONT_KEYPOINTS["left_Knee_outer"],FRONT_KEYPOINTS["left_ankle_outer"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["left_Knee_outer"],FRONT_KEYPOINTS["left_ankle_outer"]))
    print("T stone to left knee  = ",eclidian_distance(FRONT_KEYPOINTS["stone_of_trousers"],FRONT_KEYPOINTS["left_Knee_inner"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["stone_of_trousers"],FRONT_KEYPOINTS["left_Knee_inner"]))
    print("T stone to right knee  = ",eclidian_distance(FRONT_KEYPOINTS["stone_of_trousers"],FRONT_KEYPOINTS["right_Knee_inner"]),"-->",eclidian_distance2(FRONT_KEYPOINTS["stone_of_trousers"],FRONT_KEYPOINTS["right_Knee_inner"]))
    
    cv2.imshow("front_image", front_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
