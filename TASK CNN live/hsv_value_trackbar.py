import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def nothing(x):
    pass

cv2.namedWindow('HSV Calibration')
cv2.createTrackbar("L - H", "HSV Calibration",   0, 179, nothing)
cv2.createTrackbar("L - S", "HSV Calibration",   0, 255, nothing)
cv2.createTrackbar("L - V", "HSV Calibration",   0, 255, nothing)
cv2.createTrackbar("U - H", "HSV Calibration", 179, 179, nothing)
cv2.createTrackbar("U - S", "HSV Calibration", 255, 255, nothing)
cv2.createTrackbar("U - V", "HSV Calibration", 255, 255, nothing)

print("adjust trackbars until your pen tip is white in the mask")
print("s = save and quit,  enter = quit without saving")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "HSV Calibration")
    l_s = cv2.getTrackbarPos("L - S", "HSV Calibration")
    l_v = cv2.getTrackbarPos("L - V", "HSV Calibration")
    u_h = cv2.getTrackbarPos("U - H", "HSV Calibration")
    u_s = cv2.getTrackbarPos("U - S", "HSV Calibration")
    u_v = cv2.getTrackbarPos("U - V", "HSV Calibration")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower, upper)

    cv2.imshow('HSV Calibration', frame)
    cv2.imshow('Mask', mask)

    key = cv2.waitKey(5) & 0xFF
    if key == 13:
        print("quit without saving")
        break
    if key == ord('s'):
        np.save('hsv_value', [[l_h, l_s, l_v], [u_h, u_s, u_v]])
        print(f"saved  lower={[l_h, l_s, l_v]}  upper={[u_h, u_s, u_v]}")
        break

cap.release()
cv2.destroyAllWindows()