from ultralytics import YOLO
import cv2 as cv
import matplotlib.pyplot as plt

yolo = YOLO('F:/programming/computer vision nanodegree/projects/car_plate_recognition/model/best (2).pt')

img_path = 'F:/programming/computer vision nanodegree/datasets/egyptians car plates/EALPR Vechicles dataset/Vehicles/0179.jpg'

result = yolo.predict(img_path)
x1,y1,x2,y2 = result[0].boxes.xyxy[0]

img = cv.imread(img_path)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

cv.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)) , (255,0,0),3)
cv.putText(img, 'Car Plate', (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')  # Turn off axis
plt.show()
