from ultralytics import YOLO
import cv2 as cv
import matplotlib.pyplot as plt
import easyocr
import pandas as pd 
from datetime import datetime

yolo = YOLO('F:/programming/computer vision nanodegree/projects/car_plate_recognition/model/best (2).pt')

vid_path = 'F:/programming/computer vision nanodegree/projects/car_plate_recognition/video/Cairo Streets full with cars #egypt #cairo #shorts.mp4'

reader = easyocr.Reader(['ar'])
df = pd.DataFrame(columns=['Name', 'Car_Plate', 'Time'])
# results = yolo.predict(vid_path, save=True)

# Excel sheet 
file_path = 'F:/programming/computer vision nanodegree/projects/car_plate_recognition/output/car_plate_sheet.xlsx'

cap = cv.VideoCapture(vid_path)
car_id = 0
while True:
    ret, frame  = cap.read()
    
    if not ret:
        break
    
    results = yolo.predict(frame)

    for result in results:
        boxes = result.boxes 
        
        for box in boxes:
            # Bound boxes coordinates
            if box.conf > .5:
                x1,y1,x2,y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # crop the image
                cropped_plate = frame[y1:y2, x1:x2]
                
                # read the text from the cropped image
                texts = reader.readtext(cropped_plate)
                car_id +=1
                
                for text in texts:
                    # add the text to the dataframe
                    new_data = {'Name': car_id, 'Car_Plate': text[1], 'Time': datetime.now()}
                    # Convert the new data to a DataFrame
                    new_row = pd.DataFrame([new_data])

                    # Append the new row using pd.concat
                    df = pd.concat([df, new_row], ignore_index=True)
                    
    # Break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Save data into excel sheet 
df.to_excel('F:/programming/computer vision nanodegree/projects/car_plate_recognition/output/car_plate_sheet.xlsx', index=False)

