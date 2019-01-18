import cv2
import numpy as np
from matplotlib import pyplot as plt



while True:
	img = cv2.imread('quantt.png')
	hsv2 = cv2.GaussianBlur(img, (15,15), 0)

	hsv3 = cv2.GaussianBlur(img, (9,9), 0)

	lower_green = np.array([30, 42, 75])
	upper_green = np.array([89, 255, 255])


	mask2 = cv2.inRange(hsv3, lower_green, upper_green)
	mask3 = cv2.inRange(hsv2, lower_green, upper_green)
	mask1 = cv2.inRange(img, lower_green, upper_green)

	cv2.imshow('frame', img)
	cv2.imshow('gauss', hsv3)
	cv2.imshow('mask', mask1)
	cv2.imshow('mask guass', mask2)
	cv2.imshow('mask guass 2', mask3)
	

	

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break
cv2.destroyAllWindows()
cap.release()