# import the necessary packages
#from pyimagesearch.motion_detection import SingleMotionDetector
import argparse
import datetime
import pickle
import threading
import time
from threading import Thread
from time import sleep
import cv2
import imutils
import numpy as np
import pigpio
from flask import Flask, Response, render_template
from imutils.video import VideoStream
from numpy import interp


#------------------------------------FUNCTIONS----------------------------------------------------------------#

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

#-------------------------------------------------------------------------------------------------------------#


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()

vs = cv2.VideoCapture('rtsp://192.168.0.23:8553/unicast')
time.sleep(2.0)

panServo = 2
tiltServo = 3

panPos = 1250
tiltPos = 1600
print("move inicial")

name1 = ""

servo = pigpio.pi()
servo.set_servo_pulsewidth(panServo, panPos)
servo.set_servo_pulsewidth(tiltServo, tiltPos)

minMov = 10
maxMov = 25

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print("cafe cascade:", faceCascade)
@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")
def detect_face(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock
	# initialize the face detector and the total number of frames
	# read thus far

    # initialize the total number of frames
	# read thus far
	total = 0

    # loop over frames from the video stream
	while True:
		print("inicio loop funcao")
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		status, frame = vs.read()
		frame = maintain_aspect_ratio_resize(frame)
		print("leu e redimensionou frame")
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		print("gray")
		faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
		print("faces")
		print(faces)

		for (x, y, w, h) in faces:
			print("for")
			if args.trained == "y":
				roiGray = gray[y:y+h, x:x+w]
				id_, conf = recognizer.predict(roiGray)
				conf = int(conf)
				for name, value in dicti.items():
					if value == id_:
						name1 = name
						print(name, conf)
				if conf < 105:
					print("classifing")
					cv2.putText(frame, name1 + str(140-conf), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0 ,255), 2,cv2.LINE_AA)
					movePanTilt(x, y, w, h)
					print("mexeu")
			else:
				movePanTilt(x, y, w, h)
				print("mexeu")

	with lock:
		outputFrame = frame.copy()
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-p", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	ap.add_argument('-t', '--trained', type=str, default="n",
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	if args["trained"] == 'y':
		recognizer = cv2.face.LBPHFaceRecognizer_create()
		recognizer.read("trainer.yml")
		with open('labels', 'rb') as f:
			dicti = pickle.load(f)
			f.close()

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_face, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
	
# release the video stream pointer

