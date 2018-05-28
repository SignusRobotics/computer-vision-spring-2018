import cv2
import numpy as np 

#Open camera: 1 is "front cam" on my laptop and 0 is on the back. 
cap = cv2.VideoCapture(1)

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('./outputtestSIGN.avi',fourcc, 29.0, size, 3)

fgbg= cv2.createBackgroundSubtractorMOG2()

counter = 0
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        fgmask = fgbg.apply(frame)
        out.write(fgmask)
        out.write(frame) 

        cv2.imshow('frame', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()