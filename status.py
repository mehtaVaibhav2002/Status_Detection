import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound('wakeUp.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
eyeLeft = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
eyeRight = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl=['Close','Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
vid = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_DUPLEX
outcome_Right = [99]
outcome_Left = [99]
cnt=0
scr=0
scr1=2

#An Infinite loop to get the video frome from the webcam.
while(True):
    ret, frame = vid.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = eyeLeft.detectMultiScale(gray)
    right_eye =  eyeRight.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    #This particular loop will detect the face from the video frame
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    #This loop will predict the state of left eye
    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        cnt=cnt+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        predict_y = model.predict(l_eye)
        outcome_Left = np.argmax(predict_y, axis=1)
        if(outcome_Left[0]==1):
            lbl='Open'   
        if(outcome_Left[0]==0):
            lbl='Closed'
        break

    #This loop will predict the state of right eye
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        cnt = cnt+1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye/255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        predict_x = model.predict(r_eye)
        outcome_Right = np.argmax(predict_x, axis=1)
        if(outcome_Right[0] == 1):
            lbl = 'Open'
        if(outcome_Right[0] == 0):
            lbl = 'Closed'
        break

    #The score variable will keep the track of the state of eyes
    if(outcome_Right[0]==0 and outcome_Left[0]==0):
        scr=scr+1   
        cv2.putText(frame,"Closed",(10,height-30), font, 1,(244,164,96),1,cv2.LINE_AA)
    else:
        scr=scr-1
        cv2.putText(frame,"Open",(10,height-30), font, 1,(30,114,255),1,cv2.LINE_AA)
    
        
    if(scr<0):
        scr=0   
    cv2.putText(frame,'scr:'+str(scr),(100,height-30), font, 1,(127,255,212),1,cv2.LINE_AA)
    if(scr>15): #This will beep the alarm as the person is in a drowsy state
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  
            pass
        if(scr1<16):
            scr1= scr1+2
        else:
            scr1=scr1-2
            if(scr1<2):
                scr1=2
        cv2.rectangle(frame,(0,0),(width,height),(153,50,204),scr1) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()