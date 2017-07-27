import cv2
import os
import numpy as np
from PIL import Image


def photoTest(path):
    name = ['NOpreson','Ryanair', 'wanglei', 'fanbingbin']
    imgTest = cv2.imread(path)
    facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recoginer = cv2.face.createLBPHFaceRecognizer()
    recoginer.load("recognizer\\trainningData.yml")
    gray = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = facecascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CV_FEATURE_PARAMS_HAAR
    )
    id = 0
    num = 0
    print("Found {0} faces!".format(len(faces)))
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        id, conf = recoginer.predict(gray[y:y + h, x:x + w])
        print(conf)
        if conf <= 50:
            print(name[id])
            num = id
        m = (int)(x + w / 2)
        n = (int)(y + h / 2)
        cv2.rectangle(imgTest, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.rectangle(imgTest, (m, n - 10), (m, n + 10), (255, 255, 255), 2)
        cv2.rectangle(imgTest, (m - 10, n), (m + 10, n), (255, 255, 255), 2)
    cv2.imshow(name[id], imgTest)
    cv2.waitKey(0)


def getImageface(path, id):
    # Create the haar
    facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Read the image
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    number = len(imagePaths)
    num=0
    for imagepath in imagePaths:
        num = num + 1
        gray = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
        # Detect faces in the image
        faces = facecascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CV_FEATURE_PARAMS_HAAR
        )
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.imwrite("img/User." + str(id) + "." + str(num) + ".jpg", gray[y:y + h, x:x + w])

def videoTest():
    cap = cv2.VideoCapture(0)
    facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recoginer = cv2.face.createLBPHFaceRecognizer()
    recoginer.load("recognizer\\trainningData.yml")
    name = ['Ryanair', 'wanglei']
    id = 0
    while (True):
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
                cv2.putText(img, name[id - 1], (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
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


def video(id):
    cap = cv2.VideoCapture(0)
    facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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
        if(sampleNum>70):
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

def trainDatas():
    recoginer = cv2.face.createLBPHFaceRecognizer()
    path = 'img'
    getImagesWithID(path)
    IDs, faces = getImagesWithID(path)
    recoginer.train(faces, np.array(IDs))
    recoginer.save('recognizer/trainningData.yml')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    while(1):
        num = input('1: take photos  2: train  3：Test import: ')
        if num == '1':
            id = input('enter your id : ')
            video(id)
        elif num == '2':
            trainDatas()
        elif num == '3':
            videoTest()
        else:
            break

