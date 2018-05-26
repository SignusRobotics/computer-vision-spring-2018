import cv2
import numpy as np

images = [""]

for im in images:
    im = cv2.imread("data/test/0/IMG_1239.JPG")
    
    im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)

    skin_ycrcb_mint = np.array((0, 133, 77))
    skin_ycrcb_maxt = np.array((255, 173, 127))
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    cv2.imshow("second image", skin_ycrcb) # Second image

    imHeight,imWidth = im.shape[:2] 

    xMin = imWidth
    xMax = 0
    yMin = imHeight
    yMax = 0

    (_, contours, _) = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(im, contours, i, (255, 0, 0), 1)
            
            x,y,width,height = cv2.boundingRect(c)

            if x < xMin:
                xMin = x
            if x + width > xMax:
                xMax = x + width
            if y < yMin:
                yMin = y
            if y + height > yMax:
                yMax = y + height        

    lineThickness = 1
    cv2.line(im, (xMin, yMin), (xMin, yMax), (0,0,255), lineThickness)
    cv2.line(im, (xMin, yMin), (xMax, yMin), (0,0,255), lineThickness)
    cv2.line(im, (xMax, yMin), (xMax, yMax), (0,0,255), lineThickness)
    cv2.line(im, (xMax, yMax), (xMin, yMax), (0,0,255), lineThickness)

    cv2.imshow("final image", im)  # Final image

    cv2.waitKey(0)