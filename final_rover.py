import requests
import urllib
import cv2
import numpy as np
import time

class drive_data:
  def __init__(self): 
    """
    List of drive mode enums:
    DM_DEBUG = 0
    DM_CRAB = 1
    DM_SPIN = 2
    DM_DRIVE = 3
    """
    # code taken from https://github.com/SJSURobotics2019/missioncontrol2019/blob/drive/src/modules/Drive/joystick_new.js
    self.mode = 2
    self.T_MAX = 20
    self.AXIS_X = 0.0
    self.AXIS_Y = 0.0
    self.YAW = 0
    self.THROTTLE = 0.0
    self.BRAKES = 0
    # on the physical joystick the button acts as a dead man's switch
    self.mast_position = 0
    self.TRIGGER = 0
    self.REVERSE = 0
    # in drive mode, this determines which wheel is the rear wheel
    # this may be changed depending on the position of the RealSense camera on the rover
    self.wheel_A = 0
    self.wheel_B = 0



# Getting the distance of the tennis ball
def getDistance(int pixelWidth):
	return 4400 / pixelWidth
def stopRover():
	d_data = drive_data()
	d_data.YAW = 0
	d_data.THROTTLE = 0
	d_data_str = "mode=%i&AXIS_X=%f&AXIS_Y=%f&YAW=%f&THROTTLE=%f&BRAKES=%r&MAST_POSITION=%f&TRIGGER=%r&REVERSE=%r&wheel_A=%i&wheel_B=%i&wheel_C=%i" % (
	d_data.mode, d_data.T_MAX, d_data.AXIS_X, d_data.AXIS_Y, d_data.YAW, d_data.THROTTLE, d_data.BRAKES, 
	drive_data.MAST_POSITION, d_data.TRIGGER, d_data.REVERSE, d_data.wheel_A, d_data.wheel_B, d_data.wheel_C)
	requests.post("http://" + esp32_ipaddr + ":80/handle_update?" + d_data_str)

def initializeRover():
	#Initialize Spin
	d_data = drive_data()
	d_data_str = "mode=%i&AXIS_X=%f&AXIS_Y=%f&YAW=%f&THROTTLE=%f&BRAKES=%r&MAST_POSITION=%f&TRIGGER=%r&REVERSE=%r&wheel_A=%i&wheel_B=%i&wheel_C=%i" % (
	d_data.mode, d_data.T_MAX, d_data.AXIS_X, d_data.AXIS_Y, d_data.YAW, d_data.THROTTLE, d_data.BRAKES, 
	drive_data.MAST_POSITION, d_data.TRIGGER, d_data.REVERSE, d_data.wheel_A, d_data.wheel_B, d_data.wheel_C)
	requests.post("http://" + esp32_ipaddr + ":80/handle_update?" + d_data_str)
	

	# MIGHT NEED TO BE UPDATED DURING PHYSICAL TESTING - '3' SYMBOLIZES TIME IN BETWEEN ROVER CHANGING MODES AND STARTING SPIN
	time.sleep(3)
	d_data = drive_data()
	d_data.YAW = -.1
	d_data.THROTTLE = .1
	d_data_str = "mode=%i&AXIS_X=%f&AXIS_Y=%f&YAW=%f&THROTTLE=%f&BRAKES=%r&MAST_POSITION=%f&TRIGGER=%r&REVERSE=%r&wheel_A=%i&wheel_B=%i&wheel_C=%i" % (
	d_data.mode, d_data.T_MAX, d_data.AXIS_X, d_data.AXIS_Y, d_data.YAW, d_data.THROTTLE, d_data.BRAKES, 
	drive_data.MAST_POSITION, d_data.TRIGGER, d_data.REVERSE, d_data.wheel_A, d_data.wheel_B, d_data.wheel_C)
	requests.post("http://" + esp32_ipaddr + ":80/handle_update?" + d_data_str)

def deepLearningModel(frame):
	labelsPath = os.path.sep.join("yolo", "obj2.names"])
	LABELS = open(labelsPath).read().strip().split("\n")
	 
	# initialize a list of colors to represent each possible class label
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join("yolo", "yolov3.weights"])
	configPath = os.path.sep.join("yolo", "yolov3.cfg"])
	 
	# load our YOLO object detector trained on COCO dataset (80 classes)
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# load our input image and grab its spatial dimensions
	image = frame
	(H, W) = image.shape[:2]
	 
	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	 
	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	 
	# show timing information on YOLO
	# print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > .1:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, .1, .1)
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			if LABELS[classIDs[i]] == 'sports ball':
				

				return True
		return False


def execute():
	start_time = time.time()

	# For getting image data from pi camera
	camera_stream_ipaddr = '169.254.43.9:8080'

	esp32_ipaddr = '192.168.4.1'

	file = open('distancedata.txt', 'w')


	# Lower and Upper thresholds for green tennis ball HSV range
	lower_green = np.array([29,84, 6])
	upper_green = np.array([64, 255, 255])

	params = cv2.SimpleBlobDetector_Params()
	params.fitlerByColor = False

	params.minThreshold = 0
	params.maxThreshold = 255

	params.filterByArea = True
	params.minArea = 100

	params.filterByCircularity = True
	params.minCircularity = .3

	params.filterByInertia = True
	params.minInertiaRatio = .3

	params.filterByConvexity = True
	params.minConvexity = .1

	detector = cv2.SimpleBlobDetector_create(params)

	initializeRover()


	isCenter = False
	isFound = False
	isLeft = True

	while True:
		resp = urllib.urlopen(url)
		frameCounter += 1
		frame = np.asarray(bytearray(resp.read()), dtype="uint8")
		frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
		width = frame.shape[1]
		# MIGHT NEED TO BE UPDATED DURING PHYSICAL TESTING - '90' SYMBOLIZES TIME IN BETWEEN ROVER CHANGING MODES AND STARTING SPIN
		if start_time - time.time() < 90 and isFound == False:
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			hsv = cv2.bilateralFilter(hsv, 9,30,30)

			#additional filters here

			width = hsv.shape[1]
			 
			# This will only show pixels (in white) within threshold
			mask = cv2.inRange(hsv, lower_green, upper_green)

			reversemask = 255 - mask
			keypoints = detector.detect(reversemask)

			if isFound == True and len(keypoints) == 0:
				stopRover()


			if len(keypoints) > 0:
				diameter = int(keypoints[0].size)
				x = (int(keypoints[0].pt[0]) - diameter / 2)
				y = (int(keypoints[0].pt[1]) - diameter / 2)
				file.write('Completed')			


				# if tennis ball is in middle of the frame
				if x >= .45 * width and x <= .55 * width and frameCounter % 10 == 0:
					

					if isCenter == True:
						ts = time.time() * 1000000000
						ts_str = "%.20f" % ts
						distance = getDistance(diameter)
						file.write(str(ts) + " distance: " + str(getDistance(diameter)))
					else:
						isTennisBall = deepLearningModel(frame)
						if isTennisBall == True:
							isCenter = True

					
				elif x > .55 * width and direction == 'left':
					d_data = drive_data()
					d_data.mode = 3
					d_data.YAW = .1
					d_data.THROTTLE = .1
					d_data_str = "mode=%i&AXIS_X=%f&AXIS_Y=%f&YAW=%f&THROTTLE=%f&BRAKES=%r&MAST_POSITION=%f&TRIGGER=%r&REVERSE=%r&wheel_A=%i&wheel_B=%i&wheel_C=%i" % (
					d_data.mode, d_data.T_MAX, d_data.AXIS_X, d_data.AXIS_Y, d_data.YAW, d_data.THROTTLE, d_data.BRAKES, 
					drive_data.MAST_POSITION, d_data.TRIGGER, d_data.REVERSE, d_data.wheel_A, d_data.wheel_B, d_data.wheel_C)
					requests.post("http://" + esp32_ipaddr + ":80/handle_update?" + d_data_str)
					isLeft = False
				elif x < .45 * width and direction === 'right':
					d_data = drive_data()
					d_data.mode = 3
					d_data.YAW = -.1
					d_data.THROTTLE = .1
					d_data_str = "mode=%i&AXIS_X=%f&AXIS_Y=%f&YAW=%f&THROTTLE=%f&BRAKES=%r&MAST_POSITION=%f&TRIGGER=%r&REVERSE=%r&wheel_A=%i&wheel_B=%i&wheel_C=%i" % (
					d_data.mode, d_data.T_MAX, d_data.AXIS_X, d_data.AXIS_Y, d_data.YAW, d_data.THROTTLE, d_data.BRAKES, 
					drive_data.MAST_POSITION, d_data.TRIGGER, d_data.REVERSE, d_data.wheel_A, d_data.wheel_B, d_data.wheel_C)
					requests.post("http://" + esp32_ipaddr + ":80/handle_update?" + d_data_str)

					isLeft = True
		else: 
			labelsPath = os.path.sep.join("yolo", "obj2.names")
			LABELS = open(labelsPath).read().strip().split("\n")
			 
			# initialize a list of colors to represent each possible class label
			np.random.seed(42)
			COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

			# derive the paths to the YOLO weights and model configuration
			weightsPath = os.path.sep.join("yolo", "yolov3.weights")
			configPath = os.path.sep.join("yolo", "yolov3.cfg")
			 
			# load our YOLO object detector trained on COCO dataset (80 classes)
			net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

			# load our input image and grab its spatial dimensions
			image = frame
			(H, W) = image.shape[:2]
			 
			# determine only the *output* layer names that we need from YOLO
			ln = net.getLayerNames()
			ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
			 
			# construct a blob from the input image and then perform a forward
			# pass of the YOLO object detector, giving us our bounding boxes and
			# associated probabilities
			blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			net.setInput(blob)
			start = time.time()
			layerOutputs = net.forward(ln)
			# end = time.time()
			 
			# show timing information on YOLO
			# print("[INFO] YOLO took {:.6f} seconds".format(end - start))

			# initialize our lists of detected bounding boxes, confidences, and
			# class IDs, respectively
			boxes = []
			confidences = []
			classIDs = []

			# loop over each of the layer outputs
			for output in layerOutputs:
				# loop over each of the detections
				for detection in output:
					# extract the class ID and confidence (i.e., probability) of
					# the current object detection
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					# filter out weak predictions by ensuring the detected
					# probability is greater than the minimum probability
					if confidence > args["confidence"]:
						# scale the bounding box coordinates back relative to the
						# size of the image, keeping in mind that YOLO actually
						# returns the center (x, y)-coordinates of the bounding
						# box followed by the boxes' width and height
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						# use the center (x, y)-coordinates to derive the top and
						# and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						# update our list of bounding box coordinates, confidences,
						# and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)
			# apply non-maxima suppression to suppress weak, overlapping bounding
			# boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
			# ensure at least one detection exists
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					# extract the bounding box coordinates
					if LABELS[classIDs[i]] == 'sports ball':
						(x, y) = (boxes[i][0], boxes[i][1])
						(w, h) = (boxes[i][2], boxes[i][3])

						if x >= .45 * width and x <= .55 * width:
							ts = time.time() * 1000000000
							ts_str = "%.20f" % ts
							distance = getDistance(diameter)
							file.write(str(ts) + " distance: " + str(getDistance(diameter)))
							
						elif x > .55 * width and direction == 'left':
							d_data = drive_data()
							d_data.mode = 3
							d_data.YAW = .1
							d_data.THROTTLE = .1
							d_data_str = "mode=%i&AXIS_X=%f&AXIS_Y=%f&YAW=%f&THROTTLE=%f&BRAKES=%r&MAST_POSITION=%f&TRIGGER=%r&REVERSE=%r&wheel_A=%i&wheel_B=%i&wheel_C=%i" % (
							d_data.mode, d_data.T_MAX, d_data.AXIS_X, d_data.AXIS_Y, d_data.YAW, d_data.THROTTLE, d_data.BRAKES, 
							drive_data.MAST_POSITION, d_data.TRIGGER, d_data.REVERSE, d_data.wheel_A, d_data.wheel_B, d_data.wheel_C)
							requests.post("http://" + esp32_ipaddr + ":80/handle_update?" + d_data_str)
							isLeft = False
						elif x < .45 * width and direction == 'right':
							d_data = drive_data()
							d_data.mode = 3
							d_data.YAW = -.1
							d_data.THROTTLE = .1
							d_data_str = "mode=%i&AXIS_X=%f&AXIS_Y=%f&YAW=%f&THROTTLE=%f&BRAKES=%r&MAST_POSITION=%f&TRIGGER=%r&REVERSE=%r&wheel_A=%i&wheel_B=%i&wheel_C=%i" % (
							d_data.mode, d_data.T_MAX, d_data.AXIS_X, d_data.AXIS_Y, d_data.YAW, d_data.THROTTLE, d_data.BRAKES, 
							drive_data.MAST_POSITION, d_data.TRIGGER, d_data.REVERSE, d_data.wheel_A, d_data.wheel_B, d_data.wheel_C)
							requests.post("http://" + esp32_ipaddr + ":80/handle_update?" + d_data_str)
							isLeft = True










