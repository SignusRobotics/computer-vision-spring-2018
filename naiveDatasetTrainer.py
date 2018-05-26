import cv2
import numpy as np


# Based on https://stackoverflow.com/a/20679579
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

# Based on https://stackoverflow.com/a/20679579
def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False


#frame = cv2.imread("data/1/1_02.jpg")
frame = cv2.imread("data/5/5_02.jpg")

im = frame
im_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

skin_ycrcb_mint = np.array((0, 133, 77))
skin_ycrcb_maxt = np.array((255, 173, 127))
skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
cv2.imshow("second image", skin_ycrcb) # Second image



xMin = 100
xMax = 0
yMin = 100
yMax = 0


(_, contours, _) = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area > 1000:
        cv2.drawContours(im, contours, i, (255, 0, 0), 3)

        for c in contours:
            x,y,w,h = cv2.boundingRect(c)

            if x < xMin:
                xMin = x
            if x + w > xMax:
                xMax = x + w
            if y < yMin:
                yMin = y
            if y + h > yMax:
                yMax = y + h        


lineThickness = 1
cv2.line(im, (xMin, yMin), (xMin, yMax), (0,0,255), lineThickness)
cv2.line(im, (xMin, yMin), (xMax, yMin), (0,0,255), lineThickness)
cv2.line(im, (xMax, yMin), (xMax, yMax), (0,0,255), lineThickness)
cv2.line(im, (xMax, yMax), (xMin, yMax), (0,0,255), lineThickness)

cv2.imshow("final image", im)         # Final image

cv2.waitKey(0)
"""
#Blur the image
blur = cv2.blur(frame,(3,3))

#Convert to HSV color space
hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

#Create a binary image with where white will be skin colors and rest is black
mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))

kernel_square = np.ones((11,11),np.uint8)
kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
res = cv2.bitwise_and(frame, frame, mask= mask2)

dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
filtered = cv2.medianBlur(dilation2,5)
kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
median = cv2.medianBlur(dilation2,5)
ret,thresh = cv2.threshold(median,127,255,0)

_, contours, _= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   

im = res

# x,y,w,h = cv2.boundingRect(contours[0])
# img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

params = cv2.SimpleBlobDetector_Params()
                
# Change thresholds 
params.minThreshold = 0
params.maxThreshold = 100

# Filter by 
params.filterByColor = True
params.blobColor = 255

# Filter by Area. 
params.filterByArea = True
params.minArea = 5

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.1

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect keypoints in blob.
keypoints = detector.detect(im)

# Draw the keypoints to a image.
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

xMin = 100
xMax = 0
yMin = 100
yMax = 0


for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    cv2.drawContours(frame,[hull],-1,(255,255,255),2)

    if x < xMin:
        xMin = x
    if x + w > xMax:
        xMax = x + w
    if y < yMin:
        yMin = y
    if y + h > yMax:
        yMax = y + h
    
# for pt in keypoints:
#     x, y = pt.pt
    
#     if x - pt.size/2 < xMin:
#         xMin = int(x - pt.size/2 - 0)
#     if x + pt.size/2 > xMax:
#         xMax = int(x + pt.size/2 + 0)
#     if y - pt.size/2 < yMin:
#         yMin = int(y - pt.size/2 - 0)
#     if y + pt.size/2 > yMax:
#         yMax = int(y + pt.size/2)
        

print("xMin: ",xMin)
print("xMax: ",xMax)
print("yMin: ",yMin)
print("xMax: ",yMax) 

lineThickness = 2
cv2.line(frame, (xMin, yMin), (xMin, yMax), (0,255,0), lineThickness)
cv2.line(frame, (xMin, yMin), (xMax, yMin), (0,255,0), lineThickness)
cv2.line(frame, (xMax, yMin), (xMax, yMax), (0,255,0), lineThickness)
cv2.line(frame, (xMax, yMax), (xMin, yMax), (0,255,0), lineThickness)


# cv2.imshow('bitwise and', im_with_keypoints)
cv2.imshow('bounding box', frame)
cv2.waitKey(0)

"""