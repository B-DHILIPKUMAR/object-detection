import numpy as np
import imutils
import cv2
import time

modl = "MobileNetSSD_deploy.caffemodel"
prototxt = "MobileNetSSD_deploy.prototxt.txt"
confThresh = 0.2
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "mobile"]

COLORS = np.random.uniform(0,255,size=(len(CLASSES),3))

net = cv2.dnn.readNetFromCaffe(prototxt, modl)

m_cam=cv2.VideoCapture(0)
time.sleep(2.0)
while True:
    _,frame=m_cam.read()
    frame=imutils.resize(frame,width=1000)
    (h,w)=frame.shape[:2]
    blobresize=cv2.resize(frame,(300,300))
    blobimg=cv2.dnn.blobFromImage(blobresize,0.007843,(300,300),127.5)
    net.setInput(blobimg)
    detection=net.forward()
    shape=detection.shape[2]
    for i in np.arange(0,shape):
        confidence=detection[0,0,i,2]
        if confidence>confThresh:
            id=int(detection[0,0,i,1])
            box=detection[0, 0, i, 3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")
            label="{}: {:.2f}%".format(CLASSES[id],confidence*100)
            cv2.rectangle(frame,(startX,startY),(endX,endY),COLORS[id],2)
            if startY-15>15:
                y=startY-15
            else:
                startY+15
            cv2.putText(frame,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[id],2)
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1)
    if key==27:
        break
m_cam.release()
cv2.destroyAllWindows()



