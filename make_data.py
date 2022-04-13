from cProfile import label
from unittest import result
import cv2
import mediapipe as mp
import pandas as pd

# doc anh tu web cam e

cap = cv2.VideoCapture(0)

# khoi tao thu vien mediapipe 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode = True, max_num_hands = 1, min_detection_confidence=0.5)


lm_list = []
label = "comeback"
number_of_frames = 600



# ghi nhan thong so khung xuong 
def make_landmark_timestep(results):
    print(results.landmark)
    c_lm = []
    for id, lm in enumerate(results.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)

    return c_lm


def draw_landmark_on_image(mp_drawing, hand_landmarks, img):
    mp_drawing.draw_landmarks(
        img,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS)

    for id, lm in enumerate(hand_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy),5, (0,0,0),cv2.FILLED)
        
    return img 


# with mp_hands.Hands(static_image_mode = True, max_num_hand = 1, min_detection_confidence=0.5) as hands:
while len(lm_list) <= number_of_frames:
    # nhan dien popse
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks:
         
        for hand_landmarks in results.multi_hand_landmarks:
            # ghi nhan thong so khung xuong
            lm = make_landmark_timestep(hand_landmarks)
            lm_list.append(lm)
            # ve khung xuong len anh 
            img = draw_landmark_on_image(mp_drawing, hand_landmarks, frame)



    if ret :
        cv2.imshow("image",frame)
        if  cv2.waitKey(1) == ord('q'):
            break

# write vao file
df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
cap.release()
cv2.destroyAllWindows()