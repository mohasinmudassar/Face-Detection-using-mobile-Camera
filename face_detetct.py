# pip install requests numpy opencv-python
# pip install opencv-contrib-python

import requests
import cv2
import numpy as np

url = 'http://192.168.10.3:8080/shot.jpg'
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
	r = requests.get(url)
	img_arr = np.array(bytearray(r.content), dtype=np.uint8)
	img = cv2.imdecode(img_arr, -1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray, 1.04, 6)

	if len(faces) == 0:
		print('no faces detected')
	else:
		for (x,y,w,h) in faces:
			detected = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

	cv2.imshow('IPWebcam', detected)
	if cv2.waitKey(1) & 0xFF == 27:
		break

cv2.destroyAllWindows()