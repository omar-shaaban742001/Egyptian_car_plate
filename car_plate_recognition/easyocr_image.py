from ultralytics import YOLO
import cv2 as cv
import matplotlib.pyplot as plt
import easyocr

# Initialize trained yolo 
yolo = YOLO('F:/programming/computer vision nanodegree/projects/car_plate_recognition/model/best (2).pt')

# Read image and perform object detection
img_path = 'F:/programming/computer vision nanodegree/datasets/egyptians car plates/EALPR Vechicles dataset/Vehicles/0025.jpg'

# Perform object detection and crop the detected car plate
result = yolo.predict(img_path)
x1,y1,x2,y2 = result[0].boxes.xyxy[0]
x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

img = cv.imread(img_path)
crop_img = img[y1:y2, x1:x2]
crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2RGB)

# OCR using EasyOCR
reader = easyocr.Reader(['ar'])
text = reader.readtext(crop_img)
print(text)

# Display the cropped car plate
plt.imshow(crop_img)
plt.show()
