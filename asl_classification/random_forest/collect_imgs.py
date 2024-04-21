import os

import cv2

DATA_DIR = "../../asl_dataset"

number_of_classes = 31
dataset_size = 100

cap = cv2.VideoCapture(1)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print("Collecting data for class {}".format(j))

    while True:
        ret, frame = cap.read()
        cv2.putText(
            frame,  # Image/frame on which to draw the text
            'Ready? Press "Q" ! :)',  # Text string to be displayed
            (100, 50),  # Position of the text (bottom-left corner)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            1.3,  # Font scale factor
            (0, 255, 0),  # Text color (in BGR format)
            3,  # Thickness of the text
            cv2.LINE_AA,  # Type of line for the text border
        )
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) == ord("q"):
            break
        if cv2.waitKey(25) == ord("a"):
            cap.release()
            cv2.destroyAllWindows()

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), "{}.jpg".format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
