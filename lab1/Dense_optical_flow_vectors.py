import numpy as np
import cv2 as cv

cap = cv.VideoCapture("vtest.avi")

ret, frame1 = cap.read()
if not ret:
    raise RuntimeError("Nie można wczytać pierwszej klatki")

prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

step = 16  # odstęp między wektorami

while True:
    ret, frame2 = cap.read()
    if not ret:
        print("No frames grabbed!")
        break

    next_img = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(
        prvs,
        next_img,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    # obraz do rysowania
    vis = frame2.copy()

    h, w = next_img.shape

    for y in range(0, h, step):
        for x in range(0, w, step):

            fx, fy = flow[y, x]

            end_point = (
                int(x + fx),
                int(y + fy)
            )

            cv.arrowedLine(
                vis,
                (x, y),
                end_point,
                (0, 255, 0),  # zielone strzałki
                1,
                tipLength=0.3
            )

    cv.imshow("Optical Flow Vectors", vis)

    key = cv.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break

    prvs = next_img

cap.release()
cv.destroyAllWindows()
