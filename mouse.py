import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui

####################################
wCam, hCam = 640, 480
frameR = 128  # Frame Reduction
alpha = 0.2  # Smoothening factor for cursor movement
desired_fps = 30  # Target frames per second
####################################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
cap.set(cv2.CAP_PROP_FPS, desired_fps)  # Set the camera FPS if supported

detector = htm.handDetector(maxHands=1)

wScr, hScr = pyautogui.size()
print(wScr, hScr)

while True:
    start_time = time.time()  # Start time of the loop

    # 1. Find hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the wrist coordinates
    if len(lmList) != 0:
        x_wrist, y_wrist = lmList[0][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only Index Finger: Moving Mode
        x_mapped = np.interp(x_wrist, (frameR, wCam - frameR), (0, wScr))
        y_mapped = np.interp(y_wrist, (frameR, hCam - frameR), (0, hScr))

        # 6. Smoothen Values
        clocX = plocX * (1 - alpha) + x_mapped * alpha
        clocY = plocY * (1 - alpha) + y_mapped * alpha

        # 7. Move Mouse
        pyautogui.moveTo(wScr - clocX, clocY)
        cv2.circle(img, (int(x_wrist), int(y_wrist)), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY

        # 8. Clicking Mode
        if fingers[1] == 0:
            pyautogui.click()

    # 11. Frame rate calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # 12. Display
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Frame rate limiting
    elapsed_time = time.time() - start_time
    time_to_wait = max(0, (1 / desired_fps) - elapsed_time)
    time.sleep(time_to_wait)

cv2.destroyAllWindows()
cap.release()
