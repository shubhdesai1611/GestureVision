import os

import cv2

local_dir = os.path.dirname(__file__)
DATA_DIR = "../../asl_dataset"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 32
dataset_size = 500

cap = cv2.VideoCapture(1)


def prinitingOnFrame(frame, text):
    cv2.putText(
        frame,
        text,
        (1000, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )


box_top_left = (100, 100)
box_bottom_right = (1000, 1000)

for j in range(31, number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    while True:
        ret, frame = cap.read()
        prinitingOnFrame(frame, 'Ready? Press "a"')
        cv2.rectangle(frame, box_top_left, box_bottom_right, (255, 0, 0), 2)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(25)
        if key == ord("a"):
            print("Collecting data for class {}".format(j))
            counter = 0
            while counter < dataset_size:
                ret, frame = cap.read()
                prinitingOnFrame(frame, f"Captured {counter} of {dataset_size} images")
                cv2.rectangle(frame, box_top_left, box_bottom_right, (255, 0, 0), 2)
                cropped_frame = frame[
                    box_top_left[1] : box_bottom_right[1],
                    box_top_left[0] : box_bottom_right[0],
                ]
                cv2.imwrite(
                    os.path.join(DATA_DIR, str(j), "{}.jpg".format(counter)),
                    cropped_frame,
                )
                counter += 1
                cv2.imshow("frame", frame)
                cv2.waitKey(25)
            ret, frame = cap.read()
            prinitingOnFrame(frame, f"Data collected for class:{j}")
            cv2.imshow("frame", frame)
            cv2.waitKey(3000)
            break

cap.release()
cv2.destroyAllWindows()
