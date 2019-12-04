import cv2
import numpy as np
import math
from keypoint_config import *


UPPER_RATIO = 2.2962
LOWER_RATIO = 2.530864198
HIP_HIGHT = 0
LEFT_HIP_PIXEL = (0, 0)
RIGHT_HIP_PIXEL = (0, 0)

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
    # print("point1 = ", point1,"point2 = ", point2)
    result = 0
    mid = (LEFT_HIP_PIXEL[0]+RIGHT_HIP_PIXEL[0]) / 2
    if y1 >= HIP_HIGHT and y2 >= HIP_HIGHT:
        return (math.sqrt(((x1-x2)**2)+((y1-y2)**2)))/LOWER_RATIO
    elif y1 <= HIP_HIGHT and y2 <= HIP_HIGHT:
        return (math.sqrt(((x1-x2)**2)+((y1-y2)**2)))/UPPER_RATIO
    else:
        if y1 > HIP_HIGHT:
            if x1 >= mid:
                # print("yes x1 >= mid")
                result += ((math.sqrt(((x1-LEFT_HIP_PIXEL[0])**2)+((y1-LEFT_HIP_PIXEL[1])**2)))/LOWER_RATIO)
            else:
                result += ((math.sqrt(((x1-RIGHT_HIP_PIXEL[0])**2)+((y1-RIGHT_HIP_PIXEL[1])**2)))/LOWER_RATIO)
        else:
            if x1 >= mid:
                # print("yes x1 >= mid")
                result += ((math.sqrt(((x1-LEFT_HIP_PIXEL[0])**2)+((y1-LEFT_HIP_PIXEL[1])**2)))/UPPER_RATIO)
            else:
                result += ((math.sqrt(((x1-RIGHT_HIP_PIXEL[0])**2)+((y1-RIGHT_HIP_PIXEL[1])**2)))/UPPER_RATIO)

        if y2 > HIP_HIGHT:
            if x2 >= mid:
                # print("yes x2 >= mid")
                result += ((math.sqrt(((LEFT_HIP_PIXEL[0]-x2)**2)+((LEFT_HIP_PIXEL[1]-y2)**2)))/LOWER_RATIO)
            else:
                result += ((math.sqrt(((RIGHT_HIP_PIXEL[0]-x2)**2)+((RIGHT_HIP_PIXEL[1]-y2)**2)))/LOWER_RATIO)
        else:
            if x2 >= mid:
                # print("yes x2 >= mid")
                result += ((math.sqrt(((LEFT_HIP_PIXEL[0]-x2)**2)+((LEFT_HIP_PIXEL[1]-y2)**2)))/UPPER_RATIO)
            else:
                result += ((math.sqrt(((RIGHT_HIP_PIXEL[0]-x2)**2)+((RIGHT_HIP_PIXEL[1]-y2)**2)))/UPPER_RATIO)

    return result

if __name__ == "__main__":

    body_keypoints = {"left_sholder_outer":[0,0], "left_sholder_inner":[0,0], "right_sholder_inner":[0,0], "right_sholder_outer":[0,0],
    "left_elbow_inner":[0,0],"left_sholder_outer":[0,0], "right_elbow_inner":[0,0], "right_elbow_outer":[0,0], "left_chest_arm_meeting":[0,0],
    "right_chest_arm_meeting":[0,0], "left_chest":[0,0], "right_chest":[0,0], "left_waist":[0,0], "right_waist":[0,0], "left_hip":[0,0],
    "right_hip":[0,0], "stone_of_trousers":[0,0],"left_Knee_outer":[0,0], "left_Knee_inner":[0,0], "left_ankle_outer":[0,0], "left_ankle_inner":[0,0],
    "right_Knee_outer":[0,0],"right_Knee_inner":[0,0], "right_ankle_inner":[0,0], "right_ankle_outer":[0,0]}

    img = cv2.imread("images/Rimon1.1_Wrapped.jpg", cv2.IMREAD_COLOR)
    # scale_percent = 20

    # print("img.shape[1]",img.shape[1])
    # print("img.shape[0]",img.shape[0])
    # #calculate the 50 percent of original dimensions
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)

    # dsize
    # dsize = (width, height)

    # # resize image
    # img = cv2.resize(img, dsize)
    # print("img.shape[1]",img.shape[1])
    # print("img.shape[0]",img.shape[0])
    # cv2.imwrite('new.jpg',img) 

    # # sholder outer
    cv2.circle(img, (83, 147), point_radius, right_sholder_outer_color, -1)
    #right sholder inner
    cv2.circle(img, (109, 136), point_radius, right_sholder_inner_color, -1)
    #left sholder inner
    cv2.circle(img, (153, 133), point_radius, left_sholder_inner_color, -1)
    #left sholder outer 
    cv2.circle(img, (180, 147), point_radius, left_sholder_outer_color, -1)
    #right elbow outer
    cv2.circle(img, (53, 205), point_radius, right_elbow_outer_color, -1)
    #right elbow inner
    cv2.circle(img, (72, 220), point_radius, right_elbow_inner_color, -1)
    #left elbow inner
    cv2.circle(img, (183, 221), point_radius, left_elbow_inner_color, -1)
    #left elbow outer
    cv2.circle(img, (203, 210), point_radius, left_elbow_outer_color, -1)
    #right chest and arm meeting
    cv2.circle(img, (89, 174), point_radius, right_chest_arm_meeting_color, -1)
    #left chest and arm meeting
    cv2.circle(img, (172, 176), point_radius, left_chest_arm_meeting_color, -1)
    #right waist
    cv2.circle(img, (90, 234), point_radius, right_waist_color, -1)
    #left waist
    cv2.circle(img, (168, 234), point_radius, left_waist_color, -1)

 
    #right hip
    cv2.circle(img, (87, 259), point_radius, right_hip_color, -1)
    #left hip
    cv2.circle(img, (169, 260), point_radius, left_hip_color, -1)
    #stone of trousers
    cv2.circle(img, (127, 311), point_radius, stone_of_trousers_color, -1)

    #right Knee outer 
    cv2.circle(img, (88, 363), point_radius, right_Knee_outer_color, -1)
    #right Knee inner
    cv2.circle(img, (116, 365), point_radius, right_Knee_inner_color, -1)
    #right ankle outer
    cv2.circle(img, (88, 434), point_radius, right_ankle_outer_color, -1)
    #right ankle inner
    cv2.circle(img, (110, 437), point_radius, right_ankle_inner_color, -1)

    # #left Knee inner
    cv2.circle(img, (135, 367), point_radius, left_Knee_inner_color, -1)
    #left Knee outer
    cv2.circle(img, (163, 365), point_radius, left_Knee_outer_color, -1)
    #left Knee inner
    cv2.circle(img, (144, 440), point_radius, left_ankle_inner_color, -1)
    #left Knee outer
    cv2.circle(img, (163, 441), point_radius, left_ankle_outer_color, -1)    

    HIP_HIGHT = 260
    RIGHT_HIP_PIXEL =  (87, 259)
    LEFT_HIP_PIXEL = (169, 260)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #hsv hue set value  (color)

    (res,yellow,body_keypoints["left_sholder_outer"],_) = apply_filter_return_countoures(hsv,left_sholder_outer_lower,left_sholder_outer_upper,min_contor_area= 30, img=img,contor_label="LSO")
    (res,yellow,body_keypoints["left_sholder_inner"],_) = apply_filter_return_countoures(hsv,left_sholder_inner_lower,left_sholder_inner_upper,min_contor_area= 30, img=img,contor_label="LSI")
    
    (res,yellow,body_keypoints["right_sholder_inner"],_) = apply_filter_return_countoures(hsv,right_sholder_inner_lower,right_sholder_inner_upper,min_contor_area= 30, img=img,contor_label="RSI")
    (res,yellow,body_keypoints["right_sholder_outer"],_) = apply_filter_return_countoures(hsv,right_sholder_outer_lower,right_sholder_outer_upper,min_contor_area= 30, img=img,contor_label="RSO")
    
    (res,yellow,body_keypoints["left_elbow_inner"],_) = apply_filter_return_countoures(hsv,left_elbow_inner_lower,left_elbow_inner_upper,min_contor_area= 30, img=img,contor_label="LEI")
    (res,yellow,body_keypoints["left_elbow_outer"],_) = apply_filter_return_countoures(hsv,left_elbow_outer_lower,left_elbow_outer_upper,min_contor_area= 30, img=img,contor_label="LEO ")

    (res,yellow,body_keypoints["right_elbow_inner"],_) = apply_filter_return_countoures(hsv,right_elbow_inner_lower,right_elbow_inner_upper,min_contor_area= 30, img=img,contor_label="REI")
    (res,yellow,body_keypoints["right_elbow_outer"],_) = apply_filter_return_countoures(hsv,right_elbow_outer_lower,right_elbow_outer_upper,min_contor_area= 30, img=img,contor_label="REO")
    
    (res,yellow,body_keypoints["left_chest_arm_meeting"],_) = apply_filter_return_countoures(hsv,left_chest_arm_meeting_lower,left_chest_arm_meeting_upper,min_contor_area= 30, img=img,contor_label="LCAM")
    (res,yellow,body_keypoints["right_chest_arm_meeting"],_) = apply_filter_return_countoures(hsv,right_chest_arm_meeting_lower,right_chest_arm_meeting_upper,min_contor_area= 30, img=img,contor_label="RCAM")
    
    # (res,yellow,body_keypoints["left_chest"],_) = apply_filter_return_countoures(hsv,left_chest_lower,left_chest_upper,min_contor_area= 30, img=img,contor_label="LC")
    # (res,yellow,body_keypoints["right_chest"],_) = apply_filter_return_countoures(hsv,right_chest_lower,right_chest_upper,min_contor_area= 30, img=img,contor_label="RC")
    
    (res,yellow,body_keypoints["left_waist"],_) = apply_filter_return_countoures(hsv,left_waist_lower,left_waist_upper,min_contor_area= 30, img=img,contor_label="LW")
    (res,yellow,body_keypoints["right_waist"],_) = apply_filter_return_countoures(hsv,right_waist_lower,right_waist_upper,min_contor_area= 30, img=img,contor_label="RW")
    
    (res,yellow,body_keypoints["left_hip"],_) = apply_filter_return_countoures(hsv,left_hip_lower,left_hip_upper,min_contor_area= 30, img=img,contor_label="LH") 
    (res,yellow,body_keypoints["right_hip"],_) = apply_filter_return_countoures(hsv,right_hip_lower,right_hip_upper,min_contor_area= 30, img=img,contor_label="RH")
   
    (res,yellow,body_keypoints["stone_of_trousers"],_) = apply_filter_return_countoures(hsv,stone_of_trousers_lower,stone_of_trousers_upper,min_contor_area= 30, img=img,contor_label="SOT")
    
    (res,yellow,body_keypoints["left_Knee_outer"],_) = apply_filter_return_countoures(hsv,left_Knee_outer_lower,left_Knee_outer_upper,min_contor_area= 30, img=img,contor_label="LKO")
    (res,yellow,body_keypoints["left_Knee_inner"],_) = apply_filter_return_countoures(hsv,left_Knee_inner_lower,left_Knee_inner_upper,min_contor_area= 30, img=img,contor_label="LKI")
    
    (res,yellow,body_keypoints["left_ankle_outer"],_) = apply_filter_return_countoures(hsv,left_ankle_outer_lower,left_ankle_outer_upper,min_contor_area= 30, img=img,contor_label="LAO")
    (res,yellow,body_keypoints["left_ankle_inner"],_) = apply_filter_return_countoures(hsv,left_ankle_inner_lower,left_ankle_inner_upper,min_contor_area= 30, img=img,contor_label="LAI")

    (res,yellow,body_keypoints["right_Knee_outer"],_) = apply_filter_return_countoures(hsv,right_Knee_outer_lower,right_Knee_outer_upper,min_contor_area= 30, img=img,contor_label="RKO")
    (res,yellow,body_keypoints["right_Knee_inner"],_) = apply_filter_return_countoures(hsv,right_Knee_inner_lower,right_Knee_inner_upper,min_contor_area= 30, img=img,contor_label="RKI")
    
    (res,yellow,body_keypoints["right_ankle_inner"],_) = apply_filter_return_countoures(hsv,right_ankle_inner_lower,right_ankle_inner_upper,min_contor_area= 30, img=img,contor_label="RAI")
    (res,yellow,body_keypoints["right_ankle_outer"],_) = apply_filter_return_countoures(hsv,right_ankle_outer_lower,right_ankle_outer_upper,min_contor_area= 30, img=img,contor_label="RAO")

    # print(body_keypoints("right_sholder_outer"))
    print("sholder = ",eclidian_distance(body_keypoints["right_sholder_outer"],body_keypoints["left_sholder_outer"]))
    print("waist = ",eclidian_distance(body_keypoints["right_waist"],body_keypoints["left_waist"]))
    print("hip = ",eclidian_distance(body_keypoints["left_hip"],body_keypoints["right_hip"]))
    print("under arm  = ",eclidian_distance(body_keypoints["left_chest_arm_meeting"],body_keypoints["right_chest_arm_meeting"]))
    print("hip knee left  = ",eclidian_distance(body_keypoints["left_hip"],body_keypoints["left_Knee_outer"]))
    print("hip knee right  = ",eclidian_distance(body_keypoints["right_hip"],body_keypoints["right_Knee_outer"]))
    print("waist knee left  = ",eclidian_distance(body_keypoints["left_waist"],body_keypoints["left_Knee_outer"]))
    print("waist knee right  = ",eclidian_distance(body_keypoints["right_waist"],body_keypoints["right_Knee_outer"]))
    print("waist hip left  = ",eclidian_distance(body_keypoints["left_waist"],body_keypoints["left_hip"]))
    print("waist hip right  = ",eclidian_distance(body_keypoints["right_waist"],body_keypoints["right_hip"]))
    print("knee left  = ",eclidian_distance(body_keypoints["left_Knee_outer"],body_keypoints["left_Knee_inner"]))
    print("knee right  = ",eclidian_distance(body_keypoints["right_Knee_inner"],body_keypoints["right_Knee_outer"]))
    print("knee to ankle right = ",eclidian_distance(body_keypoints["right_Knee_outer"],body_keypoints["right_ankle_outer"]))
    print("knee to ankle left  = ",eclidian_distance(body_keypoints["left_Knee_outer"],body_keypoints["left_ankle_outer"]))
    print("T stone to left knee  = ",eclidian_distance(body_keypoints["stone_of_trousers"],body_keypoints["left_Knee_inner"]))
    print("T stone to right knee  = ",eclidian_distance(body_keypoints["stone_of_trousers"],body_keypoints["right_Knee_inner"]))
    print("hip to anckle  = ",eclidian_distance(body_keypoints["right_hip"],body_keypoints["right_ankle_outer"]))

    # print("knee to ankle right = ",eclidian_distance((133,521),(132,650)))
    # print("knee to ankle left  = ",eclidian_distance(body_keypoints["left_Knee_outer"],body_keypoints["left_ankle_outer"]))
    # print("T stone to left knee  = ",eclidian_distance(body_keypoints["stone_of_trousers"],body_keypoints["left_Knee_inner"]))
    # print("T stone to right knee  = ",eclidian_distance(body_keypoints["stone_of_trousers"],body_keypoints["right_Knee_inner"]))
    
    cv2.imshow("hsv", hsv)
    cv2.imshow("img", img)
    # cv2.imshow("mask", yellow)
    # cv2.imshow("Result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
