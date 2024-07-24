import os
import numpy as np
import HandTrackingModule as htm
import cv2

folder_path = "header"

my_list = os.listdir(folder_path)
print(my_list)

overlay_list = []
for image_path in my_list:
    image = cv2.imread(f'{folder_path}//{image_path}')
    overlay_list.append(image)
print(len(overlay_list))

header = overlay_list[0]
drawColor = (255,0 ,255)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon = 0.85)
xp , yp  = 0 ,0 
imgCanvas = np.zeros((720 , 1280 , 3 ),np.uint8)
while True:

    ## Import Image
    success , img = cap.read()
    img = cv2.flip(img , 1)

    ## Find Hand landmarks
    img = detector.findHands(img)
    lmlist = detector.findPosition(img , draw = False)

    if len(lmlist) != 0:
        #print(lmlist)

        # tip of index and middle finger
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        ## Check which fingers are up
        fingers = detector.fingersUp()
        ##print(fingers)

        ## If selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp , yp = 0 , 0 
            print("Selection Mode")
            ## Checking for the click
            if y1<125:
                if 0<x1<200:
                    header = overlay_list[0]
                    drawColor = (0,0,255)
                elif 330<x1<530:
                    header = overlay_list[1]
                    drawColor = (255 , 0 ,0)
                elif 660<x1<860:
                    header = overlay_list[2]
                    drawColor = (0,0,0)
                elif 990<x1<1200:
                    header = overlay_list[3]
                    drawColor = (255,255,255)

            cv2.rectangle(img , (x1,y1-15) , (x2,y2+25) , drawColor, cv2.FILLED)

        ## Drawing Mode
        if fingers[1] and fingers[2] == False:
            cv2.circle(img , (x1,y1) , 15 , drawColor , cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp , yp = x1 , y1

            cv2.line(img , (xp , yp), (x1,y1) , drawColor , 15 )
            cv2.line(imgCanvas , (xp , yp), (x1,y1) , drawColor , 15 )

            if drawColor == (0,0,0):                ### Increase the size of eraser
                cv2.line(img , (xp , yp), (x1,y1) , drawColor , 60 )
                cv2.line(imgCanvas , (xp , yp), (x1,y1) , drawColor , 60 )
            else:
                cv2.line(img , (xp , yp), (x1,y1) , drawColor , 15 )
                cv2.line(imgCanvas , (xp , yp), (x1,y1) , drawColor , 15 )

            xp , yp = x1 , y1
    
    imgGray = cv2.cvtColor(imgCanvas , cv2.COLOR_BGR2GRAY)
    _ , imgInv = cv2.threshold(imgGray , 20 , 255 , cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv , cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img , imgInv)
    img = cv2.bitwise_or(img , imgCanvas)

    ## setting up header
    img[0:125,0:1280] = header
    
    cv2.imshow("Virtual Calculator",img)
    cv2.imshow("Image Canvas",imgCanvas)
    cv2.imshow("Inv",imgInv)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()