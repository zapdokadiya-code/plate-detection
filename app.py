import easyocr
import cv2
import pandas as pd
import os
import numpy as np
from datetime import datetime

# -----------------------------
# Setup folders
# -----------------------------
if not os.path.exists("plates"):
    os.makedirs("plates")

csv_file = "plate_log.csv"

if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["Plate Number", "Timestamp", "Image Path", "Confidence"])
    df.to_csv(csv_file, index=False)

# -----------------------------
# Load OCR + Cascade
# -----------------------------
print("Initializing EasyOCR...")
reader = easyocr.Reader(['en', 'hi'], gpu=False)

cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
print(f"Loading cascade from: {cascade_path}")
plate_cascade = cv2.CascadeClassifier(cascade_path)

if plate_cascade.empty():
    print("Error loading cascade classifier. Please ensure OpenCV is correctly installed.")
    exit(1)

detected_plates = set()

print("✅ Starting webcam....")
print("❌ Press 'q' on the keyboard to exit")

# -----------------------------
# LIVE LOOP
# -----------------------------
# Initialize VideoCapture (0 is usually the built-in webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in plates:
        plate_img = frame[y:y+h, x:x+w]

        result = reader.readtext(plate_img)

        if result:
            # Drawing a rectangle around the detected plate
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            
            # Extracted plate text
            plate_text = result[0][-2]
            plate_text = str(plate_text).upper().replace(" ", "")
            confidence = result[0][-1]

            cv2.putText(frame, f"{plate_text} ({confidence:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            if plate_text not in detected_plates and len(plate_text) >= 6:
                detected_plates.add(plate_text)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Ensure the 'plates' directory exists before saving (handled by setup loop above)
                img_path = f"plates/{plate_text}_{int(datetime.now().timestamp())}.jpg"
                
                cv2.imwrite(img_path, plate_img)

                new_row = pd.DataFrame([{
                    "Plate Number": plate_text,
                    "Timestamp": timestamp,
                    "Image Path": img_path,
                    "Confidence": confidence
                }])

                new_row.to_csv(csv_file, mode='a', header=False, index=False)

                print(f"🚗 Detected: {plate_text} (Conf: {confidence:.2f})")

    # Display the live feed
    cv2.imshow('License Plate Detection', frame)

    # Exit condition: pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Application finished.")
