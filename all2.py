import cv2
import os
import requests
import json
import pytesseract
import re
import numpy as np
from roboflow import Roboflow

rf = Roboflow(api_key="eO7L6hTB5Z8Sr6gokA8A")
project = rf.workspace().project("ingredient-classification")
model = project.version(7).model

server_url = "http://49.50.167.12:8080/save-english"

headers = {
    "Content-Type": "application/json"
}

date_pattern = r'\d{4}-\d{2}-\d{2}'





cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Camera not found")
    exit()

time_num = 0
image_num = 0
extracted = None

while cam.isOpened():
    
    status, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    contrast = 1.25
    brightness = 50
    frame[:,:,2] = np.clip(contrast * frame[:,:,2] + brightness, 0, 255)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    if not status:
        break
        
    cv2.imshow('cam', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('a'):
        cv2.imwrite('image.jpg', frame)
        
        prediction = model.predict("image.jpg", confidence = 40, overlap = 30).json()
        
        time_num = 0
        extracted_name = None

        if prediction is None:
            os.remove("image.jpg")
            time_num = 0
            print("skipping this frame")
            continue
        
        for pred in prediction['predictions']:
            extracted_name = pred['class']
            print(extracted_name)
            
        exp_date = pytesseract.image_to_string('image.jpg')
        print("before match" + exp_date)
        match = re.search(date_pattern, exp_date)
        
        if match:
            exp_date = match.group(0)
            print("after match" + exp_date)
        else:
            print("no exp_date found")
            exp_date = ""

        snd_hdr = {
            "ingredientName" : extracted_name,
            "durationAt" : exp_date
            }
        
        print(snd_hdr)
        snd_data = json.dumps(snd_hdr)
        
        print("Sending ingredient info to server")
        print(snd_data)
        response = requests.post(server_url, headers=headers, data=snd_data)
        print("response: ", response)
        print("response.text: ", response.text)
        key = None

    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
        continue



cam.release()
cv2.destroyAllWindows()
