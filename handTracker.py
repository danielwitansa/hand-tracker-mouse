import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(4, 600)

sensitivity = 2
click = 30

class kalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kalman.processNoiseCov = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        ) * 0.03

    def predict(self):
        return self.kalman.predict()

    def correct(self, coord):
        return self.kalman.correct(coord)
    
class handTracker:
    def __init__(self):
        self.filter = kalmanFilter()
        self.mpHands = mp.solutions.hands
        self.hands = mp.solutions.hands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def calcDist(self, x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance

    def handTracking(self, img):
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if id == 4: # thumb
                        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                        cxt = cx
                        cyt = cy
                    
                    if id == 8: # index
                        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                        cxi = cx
                        cyi = cy

                        print("x:", cxi,"| y:", cyi)
                    
                if (cxi*1.6*sensitivity <= 1366) and (cyi*1.2*sensitivity <= 768):
                    predicted = self.filter.predict()
                    corrected = self.filter.correct(np.array([[np.float32(cxi)], [np.float32(cyi)]]))
                    x = corrected[0]*1.6*sensitivity
                    y = corrected[1]*1.2*sensitivity
                    pyautogui.moveTo(x, y)

                    print("kx:", corrected[0], "| ky:", corrected[1])

                    distit = self.calcDist(cxi, cyi, cxt, cyt)

                    if distit < click:
                        pyautogui.click()

                else:
                    continue
            
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

airMouse = handTracker()

while True:
    _, img = cap.read()

    img = airMouse.handTracking(img)
    cv2.imshow('Air Mouse', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
