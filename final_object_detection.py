import cv2
import numpy as np
import time

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(r"MobileNetSSD_deploy.prototxt.txt", r"MobileNetSSD_deploy.caffemodel")

# initialize the video stream, allow the camera sensor to warm up,
# and initialize the FPS counter
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

# Variable to track whether a person has been detected
person_detected = False

while True:
    # grab the frame from the video stream and resize it
    ret, frame = cap.read()
    if not ret:
        print("Error reading the frame")
        break

    # adjust the size of the window
    frame = cv2.resize(frame, (900, 700))

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # Flag to check if a person has been detected in the current frame
    person_detected_in_frame = False

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections and focus only on 'person' class
        if confidence > 0.2 and int(detections[0, 0, i, 1]) == 15:
            # Set the flag to True if a person is detected
            person_detected_in_frame = True

            # extract the (x, y)-coordinates of the bounding box for the person
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
  
            # draw the prediction on the frame
            label = "Person: {:.2f}%".format(confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[15], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[15], 2)

    # Print the message if a person is detected in the frame
    if person_detected_in_frame and not person_detected:
        print("SE HA DETECTADO UNA PERSONA")
        person_detected = True
    elif not person_detected_in_frame:
        # Reset the detection status if no person is detected in the current frame
        person_detected = False

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()