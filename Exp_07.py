import serial
import struct
from time import sleep
import numpy as np
import cv2

oldAngle = 90 
#Start the serial port to communicate with arduino
bytesize=8
parity='N'
stopbits=1
timeout=0.5
data = serial.Serial('COM3', baudrate=9600, timeout=timeout)
data.write(str(oldAngle).encode())


def faceCamDif(x,w, oldAngle):
    faceCenterX=(x+(w/2))
    diffCentoFace = (resolutionX/2)-faceCenterX
    Screenwidth = 47.5 #width of screen in cm 
    targetDistance = 100 #target distance in cm
    targetDistancePixM = (resolutionX/Screenwidth)*targetDistance
    angleReq = np.arcsin([diffCentoFace/(4042)])*(180/np.pi)
    print("Face center X = ", faceCenterX)
    print("Angle Required = ", angleReq)
    movementReq = oldAngle+angleReq
    return movementReq


#Initialise the haar cascade facial recognition
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Start the videocapture stream
cap = cv2.VideoCapture(0)
resolutionX = 1920
resolutionY = 1080
cap.set(3, resolutionX);       #sets camera resolution x
cap.set(4, resolutionY);       #sets camera resolution y

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #Convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Facial tracking happens
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        print("x = ", x)
        print("y = ", y)
        print("w = ", w)
        print("h = ", h)
        cv2.rectangle(frame,(x,y), (x+w,y+h),(255,0,0),2)
        i=faceCamDif(x,w,oldAngle)
        oldAngle = i[0]
        print("Destination Angle = ",i[0])
        data.write(str(i[0]).encode())
    #     print("Servo angle write = ", str(i).encode())
        sleep(0.2)

    cv2.imshow('img',frame)


    #This quits the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
