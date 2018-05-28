from darkflow.net.build import TFNet
import cv2
import numpy as np

debug = False

options = {"model": "cfg/sign1_9/tiny-yolo-sign.cfg", "load": "bin/vekterSign1_9/tiny-yolo-sign_64000.weights", "threshold": 0.2}

tfnet = TFNet(options)

counter = 0 
while counter <1: 
    print (counter)
    filename = "./Input/Z.avi"
    
    cap = cv2.VideoCapture(filename)

    # Create a new image of same size as video: 
    h = 480 #cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = 854 #cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(h)
    print(w)
    
    image = np.zeros((h,w, 1),dtype= "uint8")
    
    #initializing output video for boxes:     
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
    out = cv2.VideoWriter('./output.avi',fourcc, 29.0, size, 3) # False)  # 'False' for 1-ch instead of 3-ch for color   
    fgbg= cv2.createBackgroundSubtractorMOG2()

    # Check if camera opened successfully
    if cap.isOpened() == False: 
        print("Error opening video stream or file")
    
    # List for storing coordinates of hand movement
    lineTracer = []
    
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Object detection
            result = tfnet.return_predict(frame)

            if debug:
                print (result)

            # Take the object detected with highest confidence
            for det in sorted(result, key=lambda x: x['confidence'], reverse=True)[:1]:
                
                if det['confidence'] > 0.3: 
                    x = det['topleft']['x']
                    y = det['topleft']['y'] 
                    x_max = det['bottomright']['x']
                    y_max = det['bottomright']['y']

                    x_center = ((x_max - x)/2) + x
                    y_center = ((y_max - y)/2) + y 

                    lineTracer.append((x_center, y_center))

                    img = cv2.rectangle(frame,(x,y),(x_max,y_max),(0,255,0),2)    

                    if debug:
                        print(det)

                    text = "%s: %s"%(det['label'], str(round(det['confidence'],3)))
                    cv2.putText(img, text, (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
                    
            lastFrame = frame
            
            fgmask = fgbg.apply(frame)
            out.write(fgmask)
            out.write(frame) 
        else: 
            break
        counter += 1

    first = True
    prevX = 0
    prevY = 0

    # iterate the list of coordinates and draw lines between each of the points
    for x, y in lineTracer:
        print("X: %d, Y: %d" %(x, y))
        
        if first:
            first = False
            prevX = x
            prevY = y
            continue
        
        cv2.line(lastFrame,(prevX,prevY), (x, y), (255,255,255), 1)
        cv2.line(image,(prevX,prevY), (x, y), (255,255,255), 20)
        
        # The next line will be drawn with the current point as the start
        prevX = x
        prevY = y

    # Save the line drawn over the last frame where a object was detected
    cv2.imwrite("outline.png", lastFrame)
    # Save the isolated lines "drawn" by the detected sign
    black = cv2.imwrite("black.png", image)

    # Cleanup
    cap.release()
    out.release()    
    cv2.destroyAllWindows()