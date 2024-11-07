import cv2
import numpy as np
import matplotlib as plt

cap = cv2.VideoCapture(0)  # 0 for webcam
face_cascade = cv2.CascadeClassifier("haarcascade.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


def show_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # histogram equalization
    # basically normalizes the brightness and increases contrast
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    eyes = eye_cascade.detectMultiScale(gray)

    for ex, ey, ew, eh in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


while True:
    ret, frame = cap.read()

    if frame is None:
        break

    # do stuff with frame
    show_face(frame)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
