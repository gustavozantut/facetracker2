#pigpio module for servo instead of RPi.GPIO in Raspberry pi avoids jittering.
import cv2
import numpy as np 
import pickle
import pigpio
from time import sleep
from numpy import interp
import argparse
from threading import Thread

def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        # Grab the image size and initialize dimensions
        dim = None
        (h, w) = image.shape[:2]

        # Return original image if no need to resize
        if width is None and height is None:
            return image

        # We are resizing height if width is none
        if width is None:
            # Calculate the ratio of the height and construct the dimensions
            r = height / float(h)
            dim = (int(w * r), height)
        # We are resizing width if height is none
        else:
            # Calculate the ratio of the 0idth and construct the dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # Return the resized image
        return cv2.resize(image, dim, interpolation=inter)

cap = cv2.VideoCapture('rtsp://192.168.0.23:8553/unicast')

args = argparse.ArgumentParser()
args.add_argument('-t', '--trained', default='n')
args = args.parse_args()

if args.trained == 'y':
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("trainer.yml")
	with open('labels', 'rb') as f:
		dicti = pickle.load(f)
		f.close()

panServo = 2
tiltServo = 3

panPos = 1250
tiltPos = 1600

name1 = ""

servo = pigpio.pi()
servo.set_servo_pulsewidth(panServo, panPos)
servo.set_servo_pulsewidth(tiltServo, tiltPos)

minMov = 10
maxMov = 25

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def movePanTilt(x, y, w, h):
	global panPos
	global tiltPos
	cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
	if int(x+(w/2)) > 360:
		panPos = int(panPos - interp(int(x+(w/2)), (360, 640), (minMov, maxMov)))
	elif int(x+(w/2)) < 280:
		panPos = int(panPos + interp(int(x+(w/2)), (280, 0), (minMov, maxMov)))
	
	if int(y+(h/2)) > 280:
		tiltPos = int(tiltPos + interp(int(y+(h/2)), (280, 480), (minMov, maxMov)))
	elif int(y+(h/2)) < 200:
		tiltPos = int(tiltPos - interp(int(y+(h/2)), (200, 0), (minMov, maxMov)))
	
	if not panPos > 2500 or not panPos < 500:
		servo.set_servo_pulsewidth(panServo, panPos)
	
	if not tiltPos > 2500 or tiltPos < 500:
		servo.set_servo_pulsewidth(tiltServo, tiltPos)

while True:
	ret, frame = cap.read()
	frame = maintain_aspect_ratio_resize(frame)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
	
	for (x, y, w, h) in faces:
		if args.trained == "y":
			roiGray = gray[y:y+h, x:x+w]
			id_, conf = recognizer.predict(roiGray)
			conf = int(conf)
			for name, value in dicti.items():
				if value == id_:
					name1 = name
					print(name, conf)

			if conf < 105:
				cv2.putText(frame, name1 + str(140-conf), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0 ,255), 2,cv2.LINE_AA)
				movePanTilt(x, y, w, h)

		else:
			movePanTilt(x, y, w, h)

	cv2.imshow('frame', frame)
	key = cv2.waitKey(1)


	if key == ord("q"):
		break
	

cv2.destroyAllWindows()