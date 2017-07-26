import cv2
import os
import numpy as np
from PIL import Image

cap = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recoginer = cv2.face.createLBPHFaceRecognizer()
recoginer.load("recognizer\\trainningData.yml")
name = ['Ryanair', 'wanglei']
id = 0
while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    # Our operations on the frame come hereq
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
         minNeighbors=7,
         minSize=(30, 30),
        flags=cv2.CV_FEATURE_PARAMS_HAAR
    )
    # print("Found {0} faces!".format(len(faces)))
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        id, conf = recoginer.predict(gray[y:y + h, x:x + w])
        print(conf)
        if conf <= 40:
            cv2.putText(img, name[id-1] , (x ,y+h) ,cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        m = (int)(x + w / 2)
        n = (int)(y + h / 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.rectangle(img, (m, n - 10), (m, n + 10), (255, 255, 255), 2)
        cv2.rectangle(img, (m - 10, n), (m + 10, n), (255, 255, 255), 2)
        # print((int)(x + w / 2), y + h / 2)
    # Display the resulting frame
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

 # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()