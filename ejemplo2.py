import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

class PersonDetectionApp:
    def __init__(self, root, window_title):
        self.root = root
        self.root.title(window_title)

        # Sección para el video
        self.video_frame = tk.Frame(root)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.cap = cv2.VideoCapture(0)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Sección para controles e indicadores
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Indicadores para el dron
        self.label_title = tk.Label(self.control_frame, text="DRON PARA VIDEOVIGILANCIA", font=("Helvetica", 16))
        self.label_title.pack()

        self.led_takeoff = tk.Label(self.control_frame, text="Dron en Tierra", bg="red", fg="white", width=20)
        self.led_takeoff.pack()

        self.led_landing = tk.Label(self.control_frame, text="Dron no ha llegado", bg="red", fg="white", width=20)
        self.led_landing.pack()

        # Botones de control
        self.btn_start_detection = tk.Button(self.control_frame, text="Iniciar Detección", command=self.start_detection)
        self.btn_start_detection.pack()

        self.btn_stop_detection = tk.Button(self.control_frame, text="Detener Detección", command=self.stop_detection, state=tk.DISABLED)
        self.btn_stop_detection.pack()

        # Indicador para detección de personas
        self.led_person_detection = tk.Label(self.control_frame, text="No se ha detectado una persona", bg="red", fg="white", width=40)
        self.led_person_detection.pack()

        self.person_detected = False
        self.detector_running = False

        # Load the model
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(r"MobileNetSSD_deploy.prototxt.txt", r"MobileNetSSD_deploy.caffemodel")

        # initialize the list of class labels MobileNet SSD was trained to
        # detect, then generate a set of bounding box colors for each class
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

    def start_detection(self):
        # Set takeoff LED to green
        self.led_takeoff.config(text="Dron en Aire", bg="green")

        self.person_detected = False
        self.detector_running = True
        self.btn_start_detection.config(state=tk.DISABLED)
        self.btn_stop_detection.config(state=tk.NORMAL)
        self.update()

    def stop_detection(self):
        # Set landing LED to green
        self.led_landing.config(text="Dron ha llegado", bg="green")

        self.detector_running = False
        self.btn_start_detection.config(state=tk.NORMAL)
        self.btn_stop_detection.config(state=tk.DISABLED)

    def update(self):
        if not self.detector_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Error al leer el frame")
            return

        # Resize frame for display
        frame = cv2.resize(frame, (400, 300))  # Adjust the size as needed

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and predictions
        self.net.setInput(blob)
        detections = self.net.forward()

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
                cv2.rectangle(frame, (startX, startY), (endX, endY), self.COLORS[15], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[15], 2)

        # Print the message if a person is detected in the frame
        if person_detected_in_frame and not self.person_detected:
            print("SE HA DETECTADO UNA PERSONA")
            self.person_detected = True
            # Set person detection LED to green
            self.led_person_detection.config(text="Se ha detectado una persona", bg="green")
        elif not person_detected_in_frame:
            # Reset the detection status if no person is detected in the current frame
            self.person_detected = False
            # Set person detection LED to red
            self.led_person_detection.config(text="No se ha detectado una persona", bg="red")

        # Convert the frame to RGB format for displaying with Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = ImageTk.PhotoImage(image=img)

        # Update the label with the new image
        self.video_label.img = img
        self.video_label.configure(image=img)

        # Schedule the next update
        self.root.after(10, self.update)

    def run(self):
        self.root.mainloop()

# Main code
root = tk.Tk()
app = PersonDetectionApp(root, "Person Detection App")
app.run()
