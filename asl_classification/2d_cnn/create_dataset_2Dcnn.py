import os
import mediapipe as mp
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = "../../asl_dataset"

directories = [dir for dir in os.listdir(DATA_DIR) if not dir.startswith(".")]

data = []
labels = []

for dir_ in tqdm(directories):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(
                #     img_rgb,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style()
                # )
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                x_min = min(x_)
                y_min = min(y_)

                for i in range(len(x_)):
                    data_aux.append([x_[i] - x_min, y_[i] - y_min])

            data.append(data_aux)
            labels.append(dir_)

f = open("data_2Dcnn.pickle", "wb")
pickle.dump({"data": data, "labels": labels}, f)
f.close()
