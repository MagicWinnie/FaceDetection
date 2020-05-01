# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
from mss import mss
import numpy as np
import argparse
import imutils
import dlib
import cv2

if __name__ == '__main__':
    def nothing(*arg):
        pass
		


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--camera", required=True,
	help="1 if webcam or 2 if screen record")
args = vars(ap.parse_args())
a = int(args["camera"])
if a == 1:
	cap = cv2.VideoCapture(0)
else:
	cv2.namedWindow( "output" )
	cv2.namedWindow( "settings" )
	cv2.createTrackbar('top', 'settings', 160, 1080, nothing)
	cv2.createTrackbar('left', 'settings', 160, 1920, nothing)
	cv2.createTrackbar('width', 'settings', 200, 1920, nothing)
	cv2.createTrackbar('height', 'settings', 200, 1080, nothing)
mon = {'top': 160, 'left': 160, 'width': 200, 'height': 200}
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'C:\Users\magic\Documents\facial-landmarks\facial-landmarks\shape_predictor_68_face_landmarks.dat')

while True:
	if a == 1:
		ret, image = cap.read()
	else:
		with mss() as sct:
			image = np.array(sct.grab(mon))
		top_img = cv2.getTrackbarPos('top', 'settings')
		left_img = cv2.getTrackbarPos('left', 'settings')
		width_img = cv2.getTrackbarPos('width', 'settings')
		height_img = cv2.getTrackbarPos('height', 'settings')
		if (top_img == 0 and height_img == 0):
			cv2.setTrackbarPos('height', 'settings', 10)
			height_img = 10
		if (width_img == 0 and left_img == 0):
			cv2.setTrackbarPos('width', 'settings', 10)
			width_img = 10
		mon['top'] = top_img
		mon['left'] = left_img
		mon['width'] = width_img
		mon['height'] = height_img
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)
	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		for (x, y) in shape:
		    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	cv2.imshow("output", image)
	if cv2.waitKey(1) == 27:
		break
if a == 1:
	cap.release()
cv2.destroyAllWindows()
