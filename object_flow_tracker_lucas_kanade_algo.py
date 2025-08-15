import cv2
import numpy as np

video_path = r"video path"
cap = cv2.VideoCapture(video_path)

lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=200, qualityLevel=0.2, minDistance=5, blockSize=7)

ret, old_frame = cap.read()
if not ret:
    print("Error: Could not read the video file.")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, (0 ,0 ,255), -1)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    output = cv2.add(frame, mask)
    cv2.imshow("Optical Flow - Lucas-Kanade", output)

    if cv2.waitKey(60) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
