# Grad-Project

  ## how to start
  
  ### 1- shot 2 photos 1 front and one side view.   
  ### 2- open file "front_image_keypoint_selector.py"  
     - edit "SCALE_PERCENT" as desired.  
     - edit "FILE_NAME" write the name you want. **This is bold text**
     - edit "image_name" type imput front image name.  
     - finally change "PERSON_HEIGHT" with the new image-person's hight.  
     
  ### 3- open file "side_image_keypoint_selector.py"  
     - edit "SCALE_PERCENT" as desired.  
     - edit "FILE_NAME" write the name you want.                  ** B **  
     - edit "image_name" type imput front image name.  
     - finally change "PERSON_HEIGHT" with the new image-person's hight.  
  
  ### 4- open file "main.py"  
     - add this line if not exists $ from front_keypoints.<FILE_NAME as nammed in **A** >import *  
     - add this line if not exists $ from side_keypoints.<FILE_NAME as nammed in **B** >import *  

  ### 5- run in terminal   $ python main.py  
