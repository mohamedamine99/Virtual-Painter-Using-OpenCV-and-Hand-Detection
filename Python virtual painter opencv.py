import cv2
import mediapipe as mp
import time
import numpy as np
import math


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(False, 1, 0.8, 0.3)

tips_pts = np.array([[]], np.int32)
Draw_pts = np.array([[]], np.int32)
colour = (255, 0, 0)
prev_frame_time = 0
curr_frame_time = 0

is_Draw_curr_Frame = False
is_Draw_prev_Frame = False

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


def distance(a, b):
    return (int(math.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))))


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


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ok, img = cap.read()
    if not ok:
        continue
    h, w, _ = img.shape
    empty_img = 255 * np.ones((h, w, 3), np.uint8)

    img = cv2.flip(img, 1)
    for color in Color_Circle:
        # print(color)
        cv2.circle(img, Color_Circle[color]["Center"],
                   Color_Circle[color]["Radius"],
                   Color_Circle[color]["Color"], -1)
        cv2.circle(empty_img, Color_Circle[color]["Center"],
                   Color_Circle[color]["Radius"],
                   Color_Circle[color]["Color"], -1)

    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(RGB_img)

    if results.multi_hand_landmarks:
        for handlm in results.multi_hand_landmarks:
            for id, lm in enumerate(handlm.landmark):
                # print(handlm.landmark)
                lm_pos = (int(lm.x * w), int(lm.y * h))
                mp_draw.draw_landmarks(img, handlm, mp_hands.HAND_CONNECTIONS)
                if (id % 4 == 0):
                    tips_pts = np.append(tips_pts, lm_pos)
                    tips_pts = tips_pts.reshape((-1, 1, 2))
                    # print(len(tips_pts))

                    while (len(tips_pts) >= 5):
                        tips_pts = np.delete(tips_pts, len(tips_pts) - 5, 0)
                if id == 8:
                    cv2.circle(img, lm_pos, 18, (255, 255, 255), -1)
                    for color in Color_Circle:
                        Color_Circle[color]["Distance"] = distance(lm_pos, Color_Circle[color]["Center"])
                        cv2.line(img, lm_pos, Color_Circle[color]["Center"], Color_Circle[color]["Color"], 3)
                        cv2.putText(img, str(Color_Circle[color]["Distance"]), Color_Circle[color]["Center"],
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
                        if Color_Circle[color]["Distance"] < 35:
                            for i in Color_Circle:
                                Color_Circle[i]["is Active"] = False

                            Color_Circle[color]["is Active"] = True

                        if Color_Circle[color]["is Active"] == True:
                            cv2.circle(empty_img, lm_pos, 18, Color_Circle[color]["Color"], -1)


                            if (Is_in_Draw_Position(handlm.landmark, w, h)):
                                print(Is_in_Draw_Position(handlm.landmark, w, h))
                                is_Draw_curr_Frame = True
                                print(" is_Draw_curr_Frame", is_Draw_curr_Frame, "is_Draw_prev_Frame",
                                      is_Draw_prev_Frame)
                                if (is_Draw_prev_Frame == False) and (is_Draw_curr_Frame == True):
                                    Color_Circle[color]["Drawing"].append(np.array([[]], np.int32))

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

                cv2.rectangle(img, Box_corner1, Box_corner2, (0, 0, 255), 2)
                # print(Box_corner2 , h , w)
                # cv2.circle(img,Box_center,1 ,(255,0,0),2)
                cv2.polylines(img, [tips_pts], False, (255, 0, 255), 2)


    for color in Color_Circle:
        for i in range(0, len(Color_Circle[color]["Drawing"])):
            cv2.polylines(empty_img, [Color_Circle[color]["Drawing"][i]], False, Color_Circle[color]["Color"], 18)


    curr_frame_time = time.time()
    delta_time = curr_frame_time - prev_frame_time
    fps = int(1 / delta_time)
    prev_frame_time = curr_frame_time
    cv2.putText(img, "FPS : " + str(fps), (int(0.01 * w), int(0.2 * h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
    cv2.imshow("final img", img)
    cv2.imshow("empty img", empty_img)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
    elif cv2.waitKey(5) & 0xFF == ord("c"):
        for color in Color_Circle:
            Color_Circle[color]["Drawing"].clear()
            pass

cap.release()

