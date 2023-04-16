import cv2
import numpy as np
import face_recognition

#step 1: taking input
imgElon = face_recognition.load_image_file('ImagesBasic/elon musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB) #converting BGR to RGB

imgTest = face_recognition.load_image_file('ImagesBasic\\bill gates.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB) #converting BGR to RGB


#step 2: identify the face
faceLoc = face_recognition.face_locations(imgElon)[0] #show the coordinates of first detected face
encodeElon = face_recognition.face_encodings(imgElon)[0] #128 measurements of faces (distance b/w eyes, nose length, etc)
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#step 3: checking if encodings match?
results = face_recognition.compare_faces([encodeElon],encodeTest) #compare encodeTest with known list of Encodings, returns list of boolean vals
faceDis = face_recognition.face_distance([encodeElon],encodeTest) #finds distance b/w the images and returns lists of distances (if multiple imgs)
print(results,faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2) 

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)

