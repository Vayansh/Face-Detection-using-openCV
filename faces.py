import cv2 as cv

cam = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier('cascades\data\haarcascade_frontalface_alt.xml')

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv.flip(frame,1)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.5,minNeighbors = 4)
    for (x,y,w,h) in faces:
        pt = (x,y)
        pt3 = (x+w,y+h)    
        frame = cv.rectangle(frame,pt,pt3,(255,0,0),2)
        # cv.imshow('frame',frame)
    cv.imshow('frame',frame)
        
    key = cv.waitKey(1)
    if key == ord('q') or key%256 == 27:
        break
    
cam.release()    
cv.destroyAllWindows()    
