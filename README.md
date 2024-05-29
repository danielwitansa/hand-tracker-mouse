# Hand Tracking Mouse Control

This repository contains a Python application that uses `OpenCV` and `MediaPipe` for hand tracking and `pyautogui` to control mouse movements and clicks. The application also incorporates a Kalman filter for smooth mouse movement.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/danielwitansa/hand-tracker-mouse.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Code configuration :
   ```python
   sensitivity = 2 # Mouse sensitivity gain
   click = 30  # Mouse click sensitivity
   hvid = 800  # Video display horizontal
   vvid = 600  # Video display vertical
   hres = 1920  # Monitor horizontal pixels
   vres = 1080  # Monitor vertical pixels
   ```

## Usage

To run the application, use the following command:
```bash
python handTracker.py
```

## Code Explanation

The code in this repository is divided into two main classes: `kalmanFilter` and `handTracker`.

- The `kalmanFilter` class is used to predict and correct the mouse coordinates for smoother movement.
```python
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
```
- The `handTracker` class is responsible for processing the hand landmarks obtained from the MediaPipe Hands solution. It calculates the distance between the thumb and index finger and performs a click action if the distance is less than a certain threshold.
```python
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
                    
                if (cxi*xval*sensitivity <= hres) and (cyi*yval*sensitivity <= vres):
                    predicted = self.filter.predict()
                    corrected = self.filter.correct(np.array([[np.float32(cxi)], [np.float32(cyi)]]))
                    x = corrected[0]*xval*sensitivity
                    y = corrected[1]*yval*sensitivity
                    pyautogui.moveTo(x, y)

                    print("kx:", corrected[0], "| ky:", corrected[1])

                    distit = self.calcDist(cxi, cyi, cxt, cyt)

                    if distit < click:
                        pyautogui.click()

                else:
                    continue
            
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
```
The application uses the webcam feed to track the hand movements and control the mouse accordingly. The mouse movements are smoothed out using a Kalman filter.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
