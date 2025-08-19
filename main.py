import cv2
import numpy as np
import face_recognition


imgElon = face_recognition.load_image_file('data/elon face.jpeg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
faceElonLoc = face_recognition.face_locations(imgElon)[0]
elonEncodeface = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceElonLoc[1],faceElonLoc[3]),(faceElonLoc[0],faceElonLoc[2]),(255,255,0),4)


imgTest = face_recognition.load_image_file('data/ELON TEST.jpeg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
faceTestLoc = face_recognition.face_locations(imgTest)[0]
testEncodeface = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceTestLoc[1],faceTestLoc[3]),(faceTestLoc[0],faceTestLoc[2]),(255,255,0),4)

areTheySame = face_recognition.compare_faces([elonEncodeface],testEncodeface,0.5)
print(areTheySame)

cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)