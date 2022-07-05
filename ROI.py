import cv2 as cv
import os

DIR = 'dataset'


face_cascade = cv.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
filenames = os.listdir(DIR) 

for filename in filenames:
    try:
        img = cv.imread(DIR+'/'+filename)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.5,minNeighbors = 4)
        for (x,y,w,h) in faces:
            roi = img[y:y+h,x:x+w]
            cv.imwrite(DIR+'/'+filename,roi)
        cv.imshow('frame',roi)
        key = cv.waitKey()
        
        
        if key%256 == 27:
            break
    except:            
        break