import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)  # change index if needed

def focus_metric(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()   # higher = sharper


def capture_frame():
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to grab frame")
    return frame


def move_focus(step):
    """
    step > 0  -> move camera up
    step < 0  -> move camera down
    """
    print(f"Moving focus by {step}")
    time.sleep(0.2)  # simulate motor time

def autofocus(max_iters=30, step_size=1):
    direction = 1  # 1 = up, -1 = down

    frame = capture_frame()
    best_score = focus_metric(frame)
    print("Start focus score:", best_score)

    for i in range(max_iters):
        move_focus(direction * step_size)
        frame = capture_frame()
        score = focus_metric(frame)

        print(f"Iter {i}: score = {score:.2f}")

        if score > best_score:
            best_score = score
        else:
            # Went the wrong way → reverse and reduce step
            direction *= -1
            step_size = max(1, step_size // 2)
            move_focus(direction * step_size)

        if step_size == 1:
            print("Focus converged.")
            break

    print("Final focus score:", best_score)

autofocus()
cap.release()
cv2.destroyAllWindows()
