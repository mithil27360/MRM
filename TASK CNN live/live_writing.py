import cv2
import numpy as np
import torch
from modelmps import SimpleCNN

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load('simple_mnist.pth', map_location=device))
model.eval()

hsv_vals = np.load('hsv_value.npy')
lower = np.array(hsv_vals[0], dtype=np.uint8)
upper = np.array(hsv_vals[1], dtype=np.uint8)


def predict(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    retval, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None, 0.0

    x, y, w, h = cv2.boundingRect(coords)
    crop = thresh[y:y+h, x:x+w]

    side = max(w, h)
    pad = side // 4
    side = side + pad * 2

    square = np.zeros((side, side), dtype=np.uint8)
    oy = (side - h) // 2
    ox = (side - w) // 2
    square[oy:oy+h, ox:ox+w] = crop

    resized = cv2.resize(square, (28, 28))

    tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor = tensor / 255.0
    tensor = (tensor - 0.1307) / 0.3081
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = int(probs.argmax(dim=1).item())
        conf = probs[0][pred].item() * 100

    return pred, conf


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

canvas = None
prev_x, prev_y = 0, 0
is_tracking = False
label = ""
noise_kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, noise_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, noise_kernel)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_finger = False

    if contours:
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) > 600:
            x, y, w, h = cv2.boundingRect(biggest)
            cx = x + w // 2
            cy = y + h // 2

            if not (w < 10 or h < 10 or w > 200 or h > 200):
                found_finger = True
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

                if is_tracking:
                    dist = ((cx - prev_x)**2 + (cy - prev_y)**2) ** 0.5
                    if dist > 3:
                        cv2.line(canvas, (prev_x, prev_y), (cx, cy), (255, 255, 255), 12)

                prev_x, prev_y = cx, cy
                is_tracking = True

    if not found_finger:
        is_tracking = False
        prev_x, prev_y = 0, 0

    display = cv2.add(frame, canvas)

    if label:
        cv2.putText(display, label, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3)

    cv2.putText(display, "p=predict  c=clear  enter=quit", (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow("Air Writing", display)

    key = cv2.waitKey(5) & 0xFF

    if key == 13:
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
        label = ""
        prev_x, prev_y = 0, 0
        is_tracking = False
    elif key == ord('p'):
        pred, conf = predict(canvas)
        if pred is not None:
            label = f"digit: {pred}  ({conf:.1f}%)"
            print(f"predicted: {pred}  confidence: {conf:.1f}%")
        else:
            label = "draw something first"

cap.release()
cv2.destroyAllWindows()