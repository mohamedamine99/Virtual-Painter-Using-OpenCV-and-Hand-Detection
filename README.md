# Virtual-Painter-Using-OpenCV-and-Hand-Detection

![drawing](https://github.com/mohamedamine99/Virtual-Painter-Using-OpenCV-and-Hand-Detection/blob/main/drawing.gif)


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>  
    <li><a href="#software-requirements">Software Requirements</a></li>
      <ul>
        <li><a href="#python-environment">Python environment</a></li>
        <li><a href="#packages">Packages</a></li>
      </ul>
    </li>      
    <li><a href="#software-implementation">Software implementation</a></li>
      <ul>
        <li><a href="#hand-landmark-model">Hand Landmark Model</a></li>
        <li><a href="#python-implementation">Python implementation</a></li>  
      </ul>
    <li><a href="#results">Results</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
       
  </ol>
</details>

## About the project

The use of a physical device for human-computer interaction, such as a mouse or keyboard, hinders natural interface since it creates a significant barrier between the user and the machine.  
However, new sorts of HCI solutions have been developed as a result of the rapid growth of technology and software.  
In this project , I have made use of a robust hand and finger tracking system ,which can efficiently track both hand and hand landmarks features , in order to make a fun Ninja fruit-like game.

## Software Requirements:

### Python environment:

* Python 3.9 
* A python IDE , in my case I used [PyCharm](https://www.jetbrains.com/fr-fr/pycharm/).

### Packages:
* [OpenCV](https://opencv.org/course-opencv-for-beginners/#home) : OpenCV is the world's largest and most popular computer vision library . The library is cross-platform and free for use.
* [MediaPipe](https://google.github.io/mediapipe/) : MediaPipe offers cross-platform, customizable ML solutions for live and streaming media. it will help us detect and track hands and handlandmarks features.
* [Numpy](https://numpy.org/) : introducing support for large, multi-dimensional arrays and matrices, as well as a vast set of high-level mathematical functions to manipulate them.

**NB**: All these packages need to be installed properly.

## Software implementation:
### Hand Landmark Model:
For more details check this [Mediapipe hand tracking documentation](https://google.github.io/mediapipe/solutions/hands).
![image](https://user-images.githubusercontent.com/86969450/135113891-c741aa31-7ef7-4a6b-8967-398a2bc003f8.png)  

Following palm detection over the entire image, the hand landmark model uses regression to accomplish exact keypoint localization of 21 3D hand-knuckle coordinates within the detected hand regions, i.e. direct coordinate prediction.

Concerning the MULTI_HAND_LANDMARKS: 
Collection of detected/tracked hands, where each hand is represented as a list of 21 hand landmarks and each landmark is composed of x, y and z. x and y are normalized to [0.0, 1.0] by the image width and height respectively. z represents the landmark depth with the depth at the wrist being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.

### Python implementation:
Now let's get to our code:  
Let's begin with importing the required packages
 ```py
import cv2 
import time               # useful for calculating the FPS rate
import random             # for spawning "fruits" at random positions and random colours
import mediapipe as mp    # for hand detection and tracking
import math               # for various mathematical calculations
import numpy as np
 ``` 
Now lets get our objects :
 ```py
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(False, 1, 0.8, 0.3)
 ``` 
The  `hands = mp_hands.Hands(False, 1, 0.8, 0.3)` line is used to initialize a MediaPipe Hand object.  
Its arguments are as follows:
* **static_image_mode:** Whether to treat the input images as a batch of staticand possibly unrelated images, or a video stream. 
* **max_num_hands:** Maximum number of hands to detect. 
* **min_detection_confidence:** Minimum confidence value ([0.0, 1.0]) for hand detection to be considered successful. 
* **min_tracking_confidence:** Minimum confidence value ([0.0, 1.0]) for the hand landmarks to be considered tracked successfully. 
  
now , the below variables will be needed to calculate the FPS rate and get other stuff done.
 ```py
prev_frame_time = 0                   # time of the previous frame
curr_frame_time = 0                   # time of the current frame
tips_pts = np.array([[]], np.int32)   # numpy array of coordinates of the fingertips
Draw_pts = np.array([[]], np.int32)
colour = (255, 0, 0)
is_Draw_curr_Frame = False            
is_Draw_prev_Frame = False
 ``` 
 
 The Colors palette is represented by a dictionary of colors, each color itself i represenr=ted with a dictionary that contains infos about the location of every color icon , radius , color, wheter its selected , and a numpy array of arrays to indicate the points colored with that colors in the drawing.
 
  ```py
 Color_Circle = {
    "Blue": {
        "Center": (40, 40),
        "Radius": 40,
        "Color": (255, 0, 0),
        "is Active": False,
        "Drawing": [np.array([[]], np.int32)],
        "Distance": 300},

    "Green": {
        "Center": (40, 140),
        "Radius": 40,
        "Color": (0, 255, 0),
        "is Active": False,
        "Drawing": [np.array([[]], np.int32)],
        "Distance": 300},

    "Red": {
        "Center": (40, 240),
        "Radius": 40,
        "Color": (0, 0, 255),
        "is Active": False,
        "Drawing": [np.array([[]], np.int32)],
        "Distance": 300},

    "Black": {
        "Center": (40, 340),
        "Radius": 40,
        "Color": (0, 0, 0),
        "is Active": False,
        "Drawing": [np.array([[]], np.int32)],
        "Distance": 300},

    "Purple": {
        "Center": (40, 340),
        "Radius": 40,
        "Color": (200, 0, 200),
        "is Active": False,
        "Drawing": [np.array([[]], np.int32)],
        "Distance": 300},
    "Yellow": {
        "Center": (40, 440),
        "Radius": 40,
        "Color": (0, 100, 255),
        "is Active": False,
        "Drawing": [np.array([[]], np.int32)],
        "Distance": 300}

}

 ``` 
 now lets create some useful functions :
 
  ```py
   #this function creates a bounding box around the hand 
   #it takes as an argument the landmarks (0 ,4 , 8 , 12 ,16 represent the finger tips)
 def Bounding_box_coords(lms):             
    b_x1, b_y2, b_x2, b_y2 = (0, 0, 0, 0)

    b_y1 = min(lms[20].y, lms[16].y, lms[12].y, lms[8].y, lms[4].y, lms[0].y)
    b_y1 = int(b_y1 * h)

    b_y2 = max(lms[20].y, lms[16].y, lms[12].y, lms[8].y, lms[4].y, lms[0].y)
    b_y2 = int(b_y2 * h)

    b_x1 = min(lms[20].x, lms[16].x, lms[12].x, lms[8].x, lms[4].x, lms[0].x)
    b_x1 = int(b_x1 * w)

    b_x2 = max(lms[20].x, lms[16].x, lms[12].x, lms[8].x, lms[4].x, lms[0].x)
    b_x2 = int(b_x2 * w)
    # print(b_x1, b_x2)
    return (b_x1, b_y1), (b_x2, b_y2)

# this function calculates the distance between 2d-points
def distance(a, b):                       
    return (int(math.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))))

# this function determines whether the users hand is in draw position 
# a hand is in  draw position when the tip of the index and the tip of the thumb are really close
# however that distance varies with the hand's closeness to the cam so we need to make it normalized with respect to a reference distance 
# in our case the reference distance is between thumb tip and thumb dip
def Is_in_Draw_Position(handlms, w, h):
    thumb_tip_coords = (handlms[4].x * w, handlms[4].y * h)
    index_tip_coords = (handlms[8].x * w, handlms[8].y * h)
    thumb_dip_coords = (handlms[3].x * w, handlms[3].y * h)
    # index_dip_coords = (handlms[7].x * w, handlms[7].y * h)
    ref_d = distance(thumb_tip_coords, thumb_dip_coords)
    if (ref_d == 0):
        pass
    else:
        d = distance(thumb_tip_coords, index_tip_coords)
        final_d = int(d / ref_d)

    cv2.putText(img, str(final_d), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 3)
    if final_d < 1:
        return True
    return False
 ``` 
 Now let's get to the main part of the code:

  ```py
cap = cv2.VideoCapture(0)     # we set our pc webcam as our input
while cap.isOpened():         # while the webcam is opened
    ok, img = cap.read()      # capture images
    if not ok:
        continue
    h, w, _ = img.shape       # get the dimensions of our image 
    
    empty_img = 255 * np.ones((h, w, 3), np.uint8)      # create an empty white image with the size of our frame

    img = cv2.flip(img, 1)                              # the frame is mirrored so we flip it
    for color in Color_Circle:                          # display the color palette
        # print(color)
        cv2.circle(img, Color_Circle[color]["Center"],
                   Color_Circle[color]["Radius"],
                   Color_Circle[color]["Color"], -1)
        cv2.circle(empty_img, Color_Circle[color]["Center"],
                   Color_Circle[color]["Radius"],
                   Color_Circle[color]["Color"], -1)

    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # convert the frame from BGR to RGB in order to process it correctly with mediapipe
    results = hands.process(RGB_img)                  #launch the detection and tracking process on our img and store the results in "results"

    if results.multi_hand_landmarks:                  #if a hand is detected
        for handlm in results.multi_hand_landmarks:
            for id, lm in enumerate(handlm.landmark):
                # print(handlm.landmark)
                lm_pos = (int(lm.x * w), int(lm.y * h))                           # get landmarks positions
                mp_draw.draw_landmarks(img, handlm, mp_hands.HAND_CONNECTIONS)    # draw the landmarks 
                if (id % 4 == 0):                                                 # if a landmark is a fingertip ( 0,4,8,12,16,20)
                    tips_pts = np.append(tips_pts, lm_pos)                        # append fingertips coordinates to tips_pts array 
                    tips_pts = tips_pts.reshape((-1, 1, 2))
                    # print(len(tips_pts))

                    while (len(tips_pts) >= 5):                                 # keep array length constant = 5
                        tips_pts = np.delete(tips_pts, len(tips_pts) - 5, 0)
                if id == 8:                                                     # if we detect the index finger tip
                    cv2.circle(img, lm_pos, 18, (255, 255, 255), -1)
                    for color in Color_Circle:                                  # calculate the distance between the index finger tip and each color in tha palette
                        Color_Circle[color]["Distance"] = distance(lm_pos, Color_Circle[color]["Center"])
                        cv2.line(img, lm_pos, Color_Circle[color]["Center"], Color_Circle[color]["Color"], 3)
                        cv2.putText(img, str(Color_Circle[color]["Distance"]), Color_Circle[color]["Center"],
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
                        if Color_Circle[color]["Distance"] < 35:                # if the index is close enough to a color then this color becomes selected or "active"          
                            for i in Color_Circle:
                                Color_Circle[i]["is Active"] = False             # deactivate unselected colors

                            Color_Circle[color]["is Active"] = True

                        if Color_Circle[color]["is Active"] == True:
                            cv2.circle(empty_img, lm_pos, 18, Color_Circle[color]["Color"], -1)


                            if (Is_in_Draw_Position(handlm.landmark, w, h)):    # if we are in draw position
                                print(Is_in_Draw_Position(handlm.landmark, w, h))
                                is_Draw_curr_Frame = True                       # if we are currently drawing 
                                print(" is_Draw_curr_Frame", is_Draw_curr_Frame, "is_Draw_prev_Frame",
                                      is_Draw_prev_Frame)
                                if (is_Draw_prev_Frame == False) and (is_Draw_curr_Frame == True):   # if we just started a drawing sequence 
                                    Color_Circle[color]["Drawing"].append(np.array([[]], np.int32))  # append drawing coordinates in a numpy array

                                Color_Circle[color]["Drawing"][len(Color_Circle[color]["Drawing"]) - 1] = \
                                    np.append(
                                        Color_Circle[color]["Drawing"][len(Color_Circle[color]["Drawing"]) - 1],
                                        lm_pos)
                                print(Color_Circle[color]["Drawing"])

                                Color_Circle[color]["Drawing"][len(Color_Circle[color]["Drawing"]) - 1] = \
                                    Color_Circle[color]["Drawing"][len(Color_Circle[color]["Drawing"]) - 1].reshape(
                                        (-1, 1, 2))

                            else:
                                print(Is_in_Draw_Position(handlm.landmark, w, h))
                                is_Draw_curr_Frame = False

                            is_Draw_prev_Frame = is_Draw_curr_Frame
                            print(" *** is_Draw_curr_Frame", is_Draw_curr_Frame, "is_Draw_prev_Frame",
                                  is_Draw_prev_Frame)

                            print(len(Color_Circle[color]["Drawing"]))

                Box_corner1, Box_corner2 = Bounding_box_coords(handlm.landmark)  

                cv2.rectangle(img, Box_corner1, Box_corner2, (0, 0, 255), 2)            # draw a bounding box around the hand  
                # print(Box_corner2 , h , w)
                # cv2.circle(img,Box_center,1 ,(255,0,0),2)
                cv2.polylines(img, [tips_pts], False, (255, 0, 255), 2)                 # draw a polygone around the hand  


    for color in Color_Circle:                                                          # display our drawing
        for i in range(0, len(Color_Circle[color]["Drawing"])):
            cv2.polylines(empty_img, [Color_Circle[color]["Drawing"][i]], False, Color_Circle[color]["Color"], 18)


    curr_frame_time = time.time()
    delta_time = curr_frame_time - prev_frame_time
    fps = int(1 / delta_time)
    prev_frame_time = curr_frame_time
    cv2.putText(img, "FPS : " + str(fps), (int(0.01 * w), int(0.2 * h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
    cv2.imshow("final img", img)
    cv2.imshow("empty img", empty_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):         # "q" to quit
        break
    elif cv2.waitKey(1) & 0xFF == ord("c"):       # "c" to clear drawing
        for color in Color_Circle:
            Color_Circle[color]["Drawing"].clear()
            pass

cap.release()
 ``` 
 
   ## Results:
  ![drawing](https://github.com/mohamedamine99/Virtual-Painter-Using-OpenCV-and-Hand-Detection/blob/main/drawing.gif)

  
  ## Conclusion:
In this project, we successfullty detected and tracked a hand and its landmarks ,using the mediapipe module, and were able to extract data in order to create an interactive hand gesture mini-app to draw simple sketches with different colors. Such applications would be extremely useful for futur AR projects.
  
  ### Contact:
* Mail : mohamedamine.benabdeljelil@insat.u-carthage.tn -- mohamedaminebenjalil@yahoo.fr
* Linked-in profile: https://www.linkedin.com/in/mohamed-amine-ben-abdeljelil-86a41a1a9/

### Acknowledgements:
* Google developers for making the [Mediapipe hand tracking module](https://google.github.io/mediapipe/solutions/hands)
* OpenCV team for making the awesome [Opencv Library](https://opencv.org/)
* [NumPy Team](https://numpy.org/gallery/team.html) for making the [Numpy Library](https://numpy.org/about/)
  
