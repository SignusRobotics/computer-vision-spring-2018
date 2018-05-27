import cv2
import numpy as np
import glob

# Based on Darknet - https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py 
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

set_name = "training"
annotations = "data/annotations/" + set_name + "/"

files = glob.glob("data/" + set_name + "/*/*.JPG")

set_info_file = open(set_name + ".txt","w")
firstFile = True
debug = False

for file in files:
    filename = file.split("/")[3].split(".")[0]
    cls_id = file.split("/")[2]
    
    hs = open(annotations + cls_id + "/" + filename + ".txt","w")  

    # Based on https://stackoverflow.com/questions/14752006/computer-vision-masking-a-human-hand/14756351#14756351
    # Face Segmentation Using Skin-Color Map in Videophone Applications
    # Douglas Chai, Student Member, IEEE, and King N. Ngan, Senior Member, IEEE
    # https://www.ee.cuhk.edu.hk/~knngan/TCSVT_v9_n4_p551-564.pdf
    
    im = cv2.imread(file)
    im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)

    skin_ycrcb_mint = np.array((0, 133, 77))
    skin_ycrcb_maxt = np.array((255, 173, 127))
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    
    if(debug):
        cv2.imshow("skin_ycrcb", skin_ycrcb) # Second image

    imageHeight,imageWidth = im.shape[:2] 

    minX = imageWidth
    maxX = 0
    minY = imageHeight
    maxY = 0

    (_, contours, _) = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 1000:

            if(debug):
                cv2.drawContours(im, contours, i, (255, 0, 0), 1)
            
            x,y,width,height = cv2.boundingRect(c)

            if x < minX:
                minX = x
            if x + width > maxX:
                maxX = x + width
            if y < minY:
                minY = y
            if y + height > maxY:
                maxY = y + height        

    if(debug):
        lineThickness = 1
        cv2.line(im, (minX, minY), (minX, maxY), (0,0,255), lineThickness)
        cv2.line(im, (minX, minY), (maxX, minY), (0,0,255), lineThickness)
        cv2.line(im, (maxX, minY), (maxX, maxY), (0,0,255), lineThickness)
        cv2.line(im, (maxX, maxY), (minX, maxY), (0,0,255), lineThickness)

    b = (float(minX), float(maxX), float(minY), float(maxY))
    bb = convert((imageWidth,imageHeight), b)

    # [category number] [object center in X] [object center in Y] [object width in X] [object width in Y]
    print("%s %f %f %f %f" %(cls_id, bb[0], bb[1], bb[2], bb[3]))

    hs.write("%s %f %f %f %f" %(cls_id, bb[0], bb[1], bb[2], bb[3]))
    hs.close()

    if(not firstFile):  
        set_info_file.write("\n")    

    set_info_file.write("data/obj/%s/%s/%s.jpg" %(set_name, cls_id, filename))

    if(firstFile):
        firstFile = False

    if(debug):
        cv2.imshow("final image", im)  # Final image
        cv2.waitKey(0)

set_info_file.close()