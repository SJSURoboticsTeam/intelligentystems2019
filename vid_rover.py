# Read until video is completed
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MiniBatchKMeans
import requests
import urllib

cap = cv2.VideoCapture('part3.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outputsimpletest1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
frameCounter = 0
url = "http://192.168.43.84:50000/picture/1/current/"

# http://192.168.43.84:5000/cv_cmd?dir=straight
ip_addr = '192.168.43.84:5000'
while(True):
  # Capture frame-by-frame
  #ret, frame = cap.read()
  #resp = urllib.urlretrieve('https://nuts.com/images/auto/510x340/assets/08587fcd582716a6.jpg', "hello.jpg")
  resp = urllib.urlopen(url)
  frameCounter += 1
  frame = np.asarray(bytearray(resp.read()), dtype="uint8")
  frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
  
  #if ret == True:
    #frameCounter += 1
  #print(frameCounter)

  #Display the resulting frame
  # img = cv2.imread('Frame',frame)





  (h, w) = frame.shape[:2]

  # image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
  # image = image.reshape((image.shape[0] * image.shape[1], 3))
  # clt = MiniBatchKMeans(n_clusters = 4)
  # labels = clt.fit_predict(image)
  # quant = clt.cluster_centers_.astype("uint8")[labels]
  # quant = quant.reshape((h, w, 3))
  # quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

  # n = 32

  # # read image and convert to gray
  
  # img = cv2.resize(frame, (0,0), fx=.2, fy=.2)
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  # (h, w) = img.shape[:2]

  # img =np.reshape(img, (img.shape[0]* img.shape[1], 3))
  # clt = MiniBatchKMeans(n_clusters=n)
  # labels = clt.fit_predict(img)
  # quant = clt.cluster_centers_.astype("uint8")[labels]

  # quant = np.reshape(quant, (h,w,3))
  # img = np.reshape(img, (h,w,3))

  # quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
  # img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
  
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


  hsv2 = cv2.bilateralFilter(hsv,9,75,75)

  hsv3 =  cv2.medianBlur(hsv2,15)
  hsv4 = cv2.GaussianBlur(frame, (9,9), 0)

  


  # lower_green = np.array([29, 58, 0])
  # lower_green = np.array([29,84, 6])
  # upper_green = np.array([64, 255, 255])
  lower_green = np.array([32, 96, 86])
  upper_green = np.array([64, 255, 230])

  mask = cv2.inRange(hsv3, lower_green, upper_green)
  # res = cv2.bitwise_and(img, img, mask = mask)

  params = cv2.SimpleBlobDetector_Params()
  params.minThreshold = 1

  params.filterByArea = True
  params.minArea = 700

  params.filterByCircularity = True 
  params.minCircularity = .05

  params.filterByConvexity = True
  params.minConvexity = .05

  params.filterByInertia = .05

  detector = cv2.SimpleBlobDetector_create(params)

  reversemask = 255 - mask 
  keypoints = detector.detect(reversemask)





  try:
    # xval = keypoints[0].pt[0]
    # if xval < 400:
    #   print("left")
    # elif xval >= 400 and xval <= 800:
    #   print("straight")
    # elif xval > 800 and xval <1200:
    #   print("right")
    # yval = keypoints[0].pt[1]
    # diameter = keypoints[0].size
    #dimensions = hsv.shape

# height, width, number of channels in image
    # height = img.shape[0]
    width = hsv4.shape[1]
    

    counter = 0
    for x in keypoints:
      if counter == 0 :
        diameter = int(x.size) 
        xval = (int(x.pt[0])   - (diameter / 2)) 
        yval = (int(x.pt[1]) - (diameter / 2)) 
        cv2.rectangle(hsv4, (xval, yval), (xval + diameter, yval + diameter) , (0, 0, 255), 2)
        cv2.putText(hsv4,'Tennis Ball',(xval+diameter+10,yval+diameter),0,0.3,(0,255,0))


        if diameter >= 700:
          print("stop")
          if frameCounter % 10 == 0:
            res = requests.post('http://' + ip_addr + '/?cv_cmd?dir=stop')
        if xval >= .45 * width and xval <= .55 * width:
          print("straight")
          if frameCounter % 10 == 0:
            res = requests.post('http://' + ip_addr + '/?cv_cmd?dir=straight')
        if xval < .45 * width:
          print("left")
          if frameCounter % 10 == 0:
            res = requests.post('http://' + ip_addr + '/?cv_cmd?dir=left')
        if xval >= .55 * width: 
          print("right")
          if frameCounter % 10 == 0:
            res = requests.post('http://' + ip_addr + '/?cv_cmd?dir=right')

        counter += 1
        
    # diameter = int(keypoints[0].size)
    # xval = int(keypoints[0].pt[0] - (diameter / 2))
    # yval = int(keypoints[0].pt[1] - (diameter / 2))
    # cv2.rectangle(hsv4, (xval, yval), (xval + diameter, yval + diameter) , (0, 0, 255), 2)
    # cv2.putText(hsv4,'Tennis Ball',(xval+diameter+10,yval+diameter),0,0.3,(0,255,0))
      

      

    # out.write(hsv4)



    # if len(keypoints) > 1:
    #   print("More than one blob");
    # else:
    
    #   cv2.circle(img,(int(xval), int(yval)), 25, (255, 0,0 ), -1)
    #print (xval, ", ", yval, ", ", diameter)


    
  except IndexError:
    print("none")




  #cv2.imshow('res4', hsv3)
  cv2.imshow('res', hsv4)
  # print("hi")
  cv2.imshow('res3', mask)




  # Press Q on keyboard to  exit
  if cv2.waitKey(25) & 0xFF == ord('q'):
    break
 
 
 
# When everything done, release the video capture object
cap.release()
out.release()