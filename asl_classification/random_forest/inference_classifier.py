# test classifier
import cv2
import mediapipe as mp
import os
import pickle
import numpy as np

read_model = "./model.p"
model_dict = pickle.load(open(read_model, "rb"))
model = model_dict["model"]


labels_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
}

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:
    ret, frame = cap.read()

    H, W, _ = frame.shape

    data_aux = []
    x_ = []
    y_ = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W) - 150
        y1 = int(min(y_) * H) - 50

        x2 = int(max(x_) * W) + 50
        y2 = int(max(y_) * H) + 50

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        cv2.putText(
            frame,  # Image/frame on which to draw the text
            predicted_character,  # Text string to be displayed
            (x1, y1 - 10),  # Position of the text (bottom-left corner)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            1.3,  # Font scale factor
            (0, 0, 0),  # Text color (in BGR format)
            3,  # Thickness of the text
            cv2.LINE_AA,  # Type of line for the text border
        )

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
