from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/cars.mp4")  # For Video

model = YOLO("../Yolo-Weights/yolov8n.pt")  # we are using here yolo model 8 with nano version

# Classnames provide by the yolo model and by this id they identify the object

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask.png")   # This is for masking the image with video as we need to dectect object in a
                                # specific area

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)    #This package we have imported from github
                        # ^^^ This is used for tracking the object and identify that the object is same in each
                        # and every frame

limits = [400, 297, 673, 297]   # This is used to draw red line that specifies that from this line we need to
                                # count the car
totalCount = []          # this variable is used for storing the id of vehicles

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)     # here we are merging the mask img with video

    #imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    #img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)   # putting our video in yolo model and getting some results

    detections = np.empty((0, 5))   # used to create a empty array of size 5 for storing dir x1,y1,y2,x2,conf

    for r in results:      # for fetching each results
        boxes = r.boxes    #  for bounding boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]   #  for getting the coordinates of x1,y1,x2,y2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)  # used to rectancle these corrdinates
            w, h = x2 - x1, y2 - y1    # height and width of boxes

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100    # we are getting conf in many decimals thats why
                                                            # we here use ceil func and here we bring it to 2
                                                            # decimal places
            # Class Name
            cls = int(box.cls[0])    # to get the class id
            currentClass = classNames[cls] # geeting the class name

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)

                currentArray = np.array([x1, y1, x2, y2, conf]) # storing these datas
                detections = np.vstack((detections, currentArray)) # for appending in numpy

    resultsTracker = tracker.update(detections)    # to update the frame

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)   # making a line in road
    for result in resultsTracker:    # vehicles in the line
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)   # making these value in integer
        print(result)
        w, h = x2 - x1, y2 - y1    #width and height
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))   # for making sq in vehicle
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)            #for putting id

        cx, cy = x1 + w // 2, y1 + h // 2    # for the location of cetre in vehicles
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)   # for making circle in vehicle

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:   # if vehicle crosses that red line
            if totalCount.count(id) == 0:    # if that vehicle id is not present in totalCount variable
                totalCount.append(id)       # append that id in totalCount
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)  # draw line

    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))    #put the totalCount in video



    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
