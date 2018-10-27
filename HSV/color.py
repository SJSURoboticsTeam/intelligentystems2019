import cv2
import numpy as np

cap = cv2.VideoCapture(0)


while True:
	_, frame = cap.read()

	#gb = cv2.GaussianBlur(frame, (3, 3), 0)

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Gaussian Blur and Bilateral Filter
	# hsv2 = cv2.bilateralFilter(hsv, 15, 75, 75)
	hsv3 = cv2.GaussianBlur(hsv, (9,9), 0)

	
	lower_green = np.array([30, 42, 75])
	upper_green = np.array([89, 255, 255])

	mask = cv2.inRange(hsv, lower_green, upper_green)
	res = cv2.bitwise_and(frame, frame, mask = mask)

	cv2.imshow('frame', frame)
	cv2.imshow('mask', mask)
	cv2.imshow('res', res)

	params = cv2.SimpleBlobDetector_Params()
	params.minThreshold = 0
	params.maxThreshold = 256

	params.filterByArea = True
	params.minArea = 200

	params.filterByCircularity = True 
	params.minCircularity = 0.1

	params.filterByConvexity = True
	params.minConvexity = 0.5

	params.filterByInertia = True
	params.minInertiaRatio = 0.5

	detector = cv2.SimpleBlobDetector_create(params)

	reversemask = 255 - mask 
	keypoints = detector.detect(reversemask)

	
	#configured for 1280 x 720

	try:
		xval = keypoints[0].pt[0]
		if xval < 400:
			print("left")
		elif xval >= 400 and xval <= 800:
			print("straight")
		elif xval > 800 and xval <1200:
			print("right")
		#yval = keypoints[0].pt[1]
		diameter = keypoints[0].size
		#print (xval, ", ", yval, ", ", diameter)


		#cv2.circle(frame,(int(xval), int(yval)), 25, (255, 0,0 ), -1)
	except IndexError:
		print("none")

	'''_, countours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	try:
		blob = max(countours, key=lambda el:cv2.contourArea(el))
		M = cv2.moments(blob)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		print(center)
		print()
	except (ValueError, ZeroDivisionError) as e:
		print("err")
	'''

	cv2.imshow('frame', frame)
	cv2.imshow('mask', mask)
	cv2.imshow('res', res)
	

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break
cv2.destroyAllWindows()
cap.release()