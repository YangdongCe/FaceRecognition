import cv2
import os
import numpy as np
from PIL import Image


def recognition(img):
    print("success")
    return

def video():
    cap = cv2.VideoCapture(0)
    facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    id =  input('enter your id')
    sampleNum=0
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
        print("Found {0} faces!".format(len(faces)))
        if len(faces) > 0:
            recognition(img)
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            sampleNum = sampleNum + 1
            m = (int)(x + w / 2)
            n = (int)(y + h / 2)
            cv2.imwrite("img/User."+str(id)+"."+str(sampleNum)+".jpg" ,gray[y:y+h ,x:x+w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.rectangle(img, (m, n - 10), (m, n + 10), (255, 255, 255), 2)
            cv2.rectangle(img, (m - 10, n), (m + 10, n), (255, 255, 255), 2)
            print((int)(x + w / 2), y + h / 2)

        # Display the resulting frame
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if(sampleNum>20):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()




def getImagesWithID(path):
    imagePaths = [os.path.join(path ,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        ID = int(imagePath.split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("tranining" ,faceNp)
        cv2.waitKey(10)
    return IDs ,faces

def recogntions():
    recoginer = cv2.face.createLBPHFaceRecognizer()
    path = 'img'
    getImagesWithID(path)
    IDs, faces = getImagesWithID(path)
    recoginer.train(faces, np.array(IDs))
    recoginer.save('recognizer/trainningData.yml')
    cv2.destroyAllWindows()

if __name__ == '__main__':
     # video()
     recogntions()
