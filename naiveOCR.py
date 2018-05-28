import cv2
import numpy as np

frame = cv2.imread('black.png',0)
cv2.imshow("Black image with the tracked path", frame)
cv2.waitKey(0)
frame = cv2.GaussianBlur(frame,(5,5),0)
sign = frame

whiteMask = np.array([255, 255, 255], dtype = "uint8")

# blur the mask to help remove noise, then apply the
# mask to the frame
cv2.GaussianBlur(whiteMask, (3, 3), 0)

erosion = cv2.erode(sign, whiteMask,iterations = 10)

cv2.imshow("Processed image", erosion)
cv2.waitKey(0)