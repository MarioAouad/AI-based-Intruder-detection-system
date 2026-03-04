"""
calibrate_camera.py
-------------------
Phase 2: Live Webcam Distance Calibration
"""

import cv2
from ultralytics import YOLO

print("Loading YOLOv8-Small...")
model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(0)

print("\n" + "="*50)
print(" CALIBRATION INSTRUCTIONS:")
print(" 1. Grab a tape measure.")
print(" 2. Stand EXACTLY 100cm (1 meter) away from your laptop camera.")
print(" 3. Write down the 'Pixel Width' number!")
print("="*50 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam.")
        break

    # Your fix implemented here: classes=
    results = model.track(frame, classes=0, conf=0.70, persist=True, verbose=False)

    # THE BULLETPROOF METHOD: Loop through results to avoid the "list" error entirely
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            
            # Grab the coordinates of the first person detected
            box = r.boxes
            
            # Unpack the 4 coordinates safely
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Calculate the pixel width
            pixel_width = x2 - x1

            # Draw the bounding box and the number
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Pixel Width: {pixel_width} px"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Phase 2: Camera Calibration", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()