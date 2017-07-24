import cv2
import numpy as np
def recognition(img):
    print("success")
    return

def video():
    cap = cv2.VideoCapture(0)
    facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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
        if len(faces)>0:
            recognition(img)
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            m = (int)(x + w / 2)
            n = (int)(y + h / 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.rectangle(img, (m, n - 10), (m, n + 10), (255, 255, 255), 2)
            cv2.rectangle(img, (m - 10, n), (m + 10, n), (255, 255, 255), 2)
            print((int)(x + w / 2), y + h / 2)
        # Display the resulting frame
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    video()
