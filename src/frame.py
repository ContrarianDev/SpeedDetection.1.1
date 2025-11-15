import cv2

cap = cv2.VideoCapture(
    "paste your updated video URL here"
     )  # Use camera source if needed
ret, frame = cap.read()
if ret:
    cv2.imwrite("frame.jpg", frame)  # Save frame as an image
cap.release()

