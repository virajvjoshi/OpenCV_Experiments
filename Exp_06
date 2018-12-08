import numpy as np
import cv2
from matplotlib import pyplot as plt
import serial
import struct
from time import sleep

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

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    # This bit creates the outlinesoa
    fgmask = fgbg.apply(frame)
    fgmask = cv2.Canny(fgmask,190,250)
    fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE, kernel)
    # Display the resulting frame
    cv2.imshow("window", fgmask)

    #This creates the disappearing colour blur
    fgmask = fgbg.apply(frame,learningRate = 0) #this applies the background removal (there are several that can be used)
    #ret,thresh = cv2.threshold(fgmask,255,255,255)  #applies a threshold to a BW image, the countour drawings needs this
    #frame = cv2.multiply(frame,frame, scale = 0.1)
    alpha = -2
    beta = 1
    frame = cv2.convertScaleAbs(frame,frame,alpha,beta)
    image, contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #This finds the contours in the thresholded image
    image2 = cv2.drawContours(image, contours, -1, (0,255,0), 0) #This draws the contours
    image = cv2.GaussianBlur(image,(51,51),0)
    frame = cv2.blur(frame,(10,10))
    res = cv2.bitwise_and(frame,frame, mask= image)
    #res2 = cv2.bitwise_and(res,image)
    #hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV) # This converts the resultant image to HSV colour space

    #cv2.imshow("window", image) # this bit shows the end result
    #cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('frame',res)  #changed this line too (from gray)
    # cv2.imshow('original feed',frame)
    # cv2.imshow('contour', image2)
    #cv2.imshow('frame2',res2)
    #cv2.imshow('thing',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()

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
