import mediapipe as mp
import cv2
import time


class handDetector():
    def __init__(self , mode = False , maxHands = 2 , detectionCon = 0.5 , trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode = self.mode,
                                        max_num_hands = self.maxHands,
                                        min_detection_confidence =self.detectionCon,
                                        min_tracking_confidence = self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.tipIds = [4,8,12,16,20]

 
    def findHands(self , img, draw = True):
        imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img , handlms , self.mpHands.HAND_CONNECTIONS) 

        return img
    
    def findPosition(self , img , handnum = 0 , draw = True):

        self.lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handnum]

            for id , lm in enumerate(myhand.landmark):
                h , w, c = img.shape
                cx , cy = int(lm.x * w) , int(lm.y * h)
                #print(id , cx , cy)
                self.lmlist.append([id , cx , cy])
                if draw:
                    cv2.circle(img , (cx , cy) , 5 , (0,0,255) , cv2.FILLED)

        return self.lmlist
    
    def fingersUp(self):
        fingers = []

        #thumb
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1,5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error : Could not open video stream.")
        return 
    
    detector = handDetector()
    while True:
        success , img = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break
        
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        img = cv2.flip(img , 1)
        cv2.putText(img , str(int(fps)), (10,70) , cv2.FONT_HERSHEY_PLAIN , 3 , (255,0,255),3)
        cv2.imshow("Image",img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



