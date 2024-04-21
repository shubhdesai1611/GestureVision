import numpy as np
import cv2
import pickle
import mediapipe as mp
import tensorflow as tf
import time
import re
from fast_autocomplete import AutoComplete
import subprocess
from gtts import gTTS

# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

next_word_model = tf.keras.models.load_model("./next_word_model.h5")

tokenizer = {}

with open("tokenizer.pickle", "rb") as tokens:
    tokenizer = pickle.load(tokens)


def predict_next_word(text):
    token_text = tokenizer.texts_to_sequences([text])[0]
    # padding
    padded_token_text = pad_sequences([token_text], maxlen=56, padding="pre")
    # predict
    pos = np.argmax(next_word_model.predict(padded_token_text))

    for word, index in tokenizer.word_index.items():
        if index == pos:
            return word


def word_suggestion():
    words = ""
    with open("dataset.txt", "r") as file:
        words = file.read()

    words = re.split(r"\s+", words)
    words = {
        word: {}
        for word in words
        if not (word.startswith("(") or word.endswith(")") or word.endswith("?"))
    }
    autocomplete = AutoComplete(words=words)
    return autocomplete


autocomplete = word_suggestion()


def get_suggested_word(word):
    auto = ""
    try:
        auto = autocomplete.search(word=word, max_cost=3, size=1)[0][0]
    except:
        return ""
    return auto


def text_to_speech(mytext):
    if len(mytext) > 0:
        language = "en"

        myobj = gTTS(text=mytext, lang=language, slow=False)

        myobj.save("textToSpeech.mp3")

        # Playing the converted file
        subprocess.Popen(["afplay", "textToSpeech.mp3"])


cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4)

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
    26: "Next",
    27: "Space",
    28: "Current Suggested Word Accepted",
    29: "Next Suggested Word Accepted",
    30: "BackSpace",
    31: "Speak",
}

# Load the CNN model
model = tf.keras.models.load_model("./2d_cnn_model.h5")

word = [""]

next = True  # True - Expecting sign, False - Expecting next sign
previous_word = ""


def textOnFrame(frame, text, x1, y1, color):
    cv2.putText(
        frame,
        text,
        (x1, y1),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        color,
        3,
        cv2.LINE_AA,
    )


box_top_left = (100, 100)
box_bottom_right = (1000, 800)

text_for_next_sign = "Expecting Next as input"


while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    # frame = cv2.flip(frame, 1)

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

                x_.append(x)
                y_.append(y)

            for i in range(len(x_)):
                data_aux.append([x_[i] - min(x_), y_[i] - min(y_)])

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        if len(data_aux) < 42:
            data_aux.extend([[0, 0]] * (42 - len(data_aux)))

        if len(data_aux) == 42:
            data_aux_reshaped = np.asarray(data_aux).reshape((42, 2, 1))

            # Make prediction
            prediction = model.predict(np.expand_dims(data_aux_reshaped, axis=0))
            predicted_class = np.argmax(prediction)
            predicted_character = labels_dict[predicted_class]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            textOnFrame(frame, predicted_character, x1, y1 - 10, (0, 0, 0))

            if next:
                textOnFrame(
                    frame,
                    "Expecting Sign as input",
                    95,
                    95,
                    (255, 0, 0),
                )
            else:
                textOnFrame(
                    frame,
                    text_for_next_sign,
                    95,
                    95,
                    (255, 0, 0),
                )

            if (
                min(x_) * W > box_top_left[0]
                and min(y_) * H > box_top_left[1]
                and max(x_) * W < box_bottom_right[0]
                and max(y_) * H < box_bottom_right[1]
            ):
                if not next:
                    if predicted_character == "Next":
                        text_for_next_sign = "Expecting Next as input"
                        next = True
                        time.sleep(0.5)
                    else:
                        text_for_next_sign = (
                            "Please provide next (folder 26 sign) as input"
                        )

                if next and predicted_character != "Next":
                    if predicted_character == "BackSpace":
                        if len(word[-1]) > 0:
                            word[-1] = word[-1][:-1]
                    elif predicted_character == "Current Suggested Word Accepted":
                        if len(word[-1]) > 0:
                            word[-1] = get_suggested_word(word[-1])
                            previous_word = word[-1]
                            word.append("")
                    elif predicted_character == "Next Suggested Word Accepted":
                        if len(previous_word) > 0:
                            word[-1] = predict_next_word(previous_word)
                            previous_word = word[-1]
                            word.append("")
                    elif predicted_character == "Speak":
                        speech = " ".join(word)
                        text_to_speech(speech)
                    elif predicted_character == "Space":
                        if len(word[0]) == 0:
                            pass
                        else:
                            previous_word = word[-1]
                            word.append("")
                    else:
                        word[-1] += predicted_character
                        # word += predicted_character
                        # predict_next_word("good")
                    next = False

    # Draw the bounding box on the frame
    cv2.rectangle(frame, box_top_left, box_bottom_right, (255, 0, 0), 2)

    # Display the word being formed
    if len(word[-1]) > 0:
        print(word[-1])
        textOnFrame(
            frame,
            "Current Word Suggestion: " + get_suggested_word(word[-1]),
            50,
            950,
            (255, 0, 0),
        )

    if len(previous_word) > 0:
        textOnFrame(
            frame,
            "Next Word Suggestion: " + predict_next_word(previous_word),
            50,
            1000,
            (0, 255, 0),
        )
    textOnFrame(frame, "Current Sentence: " + " ".join(word), 50, 900, (50, 50, 0))

    cv2.imshow("frame", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
