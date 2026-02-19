# import cv2
# import numpy as np

# def focus_score_laplacian(gray: np.ndarray) -> float:
#     lap = cv2.Laplacian(gray, cv2.CV_64F)
#     return float(lap.var())

# cap = cv2.VideoCapture(1)  # might be 1,2,... depending on your system

# while True:
#     ok, frame = cap.read()
#     if not ok:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     h, w = gray.shape
#     # center ROI (adjust as needed)
#     roi = gray[h//4: 3*h//4, w//4: 3*w//4]

#     score = focus_score_laplacian(roi)

#     cv2.putText(frame, f"Focus score: {score:.1f}", (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#     cv2.imshow("Amscope Live", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np
from collections import deque

def tenengrad_score(gray: np.ndarray) -> float:
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag2 = gx*gx + gy*gy
    return float(mag2.mean())

def tiled_median_score(gray_roi: np.ndarray, tiles=4) -> float:
    h, w = gray_roi.shape
    th, tw = h // tiles, w // tiles
    scores = []
    for r in range(tiles):
        for c in range(tiles):
            tile = gray_roi[r*th:(r+1)*th, c*tw:(c+1)*tw]
            scores.append(tenengrad_score(tile))
    return float(np.median(scores))

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

window = deque(maxlen=10)  # temporal average across frames

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # central ROI
    roi = gray[h//3: 2*h//3, w//3: 2*w//3]

    score = tiled_median_score(roi, tiles=4)
    window.append(score)
    score_avg = sum(window) / len(window)

    cv2.putText(frame, f"Focus score (avg): {score_avg:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Live", frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
