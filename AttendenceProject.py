import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# step 1: fetching images

path = 'ImagesAttendence'
images = []                             #to store images
classNames = []                         #to store names
myList = os.listdir(path)   # returns list of all file/directories in path   
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])      #os.path.splitext splits path to path & extention, we are storing part before . to list
print(classNames)



# step 2: find encodings

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0] # store the encoding of first detected face 
        encodeList.append(encode)   # storing the encoding into the list
    return(encodeList)

def markAttendance(name,regisNo):
    with open('attendence.csv','r+') as f: # 'with ensures file is closed properly after block is executed
    # open returns a file object, and it need to be closed after execution using close(), hence we used with to close it automatically
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()  #returns date time object needs to be converted into string
            dtString = now.strftime('%H:%M:%S') # string format time
            f.writelines(f'\n{name},{regisNo},{dtString}')



encodeListKnown = findEncodings(images) # making a list of known faces (faces to match with)
print('Encoding Complete')


cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()   #reads the frame of stream and return boolean val & frame data (as numpy array)
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)


# step 3: find matches b/w encodings 
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame): #zip combines objects from both lists(iterables) into a single iterable
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace) # will give a list showing comparision with all list value
        #print(faceDis)
        matchIndex = np.argmin(faceDis) # returns index of minimum value in array

        if matches[matchIndex]:
            #name = classNames[matchIndex].upper()
            stuDetail = classNames[matchIndex].split('_')
            name = stuDetail[0].upper()
            if len(stuDetail)>1 :
                  regisNo = stuDetail[1]
            else:
                regisNo = 'Registration Number Not Exist'
            #print(regisNo)

            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 #as location is calculate in scaled downed image
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name,regisNo)
        else:
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 #as location is calculate in scaled downed image
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,'Intruder',(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)