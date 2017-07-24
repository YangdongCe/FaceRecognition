import cv2
# Create the haar
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the image

img = cv2.imread("D:\images.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces in the image
faces = facecascade.detectMultiScale(
   gray,
   scaleFactor=1.2,
   minNeighbors=5,
   minSize=(30, 30),
   flags=cv2.CV_FEATURE_PARAMS_HAAR
)
print ("Found {0} faces!".format(len(faces)))
# Draw a rectangle around the faces
for (x, y, w, h) in faces:
   m = (int)(x+w/2)
   n = (int)(y+h/2)
   cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
   cv2.rectangle(img, (m, n-10), (m, n+10), (255, 255, 255), 2)
   cv2.rectangle(img, (m-10, n), (m + 10, n), (255, 255, 255), 2)
   print((int)(x+w/2), y+h/2)

cv2.imshow("Faces found", img)
cv2.waitKey(0)