import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO('runs/detect/train4/weights/best.pt')  # Use your trained model weights here

# Start capturing video from the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera is opened properly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to continuously capture frames
while True:
    # Capture a single frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run inference on the current frame
    results = model.predict(source=frame, imgsz=640, conf=0.1)  # Adjust the confidence threshold as needed

    # Since results is a list, we access the first element (inference for the current frame)
    result = results[0]

    # Draw bounding boxes and labels on the frame
    frame_with_boxes = result.plot()  # This adds the boxes to the frame

    # Show the frame with detections (bounding boxes, labels, etc.)
    cv2.imshow("Object Detection", frame_with_boxes)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
