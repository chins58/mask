import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained mask detection model
mask_model = load_model('mask_detector.model')  # Assuming you have a trained mask detection model

# Labels for mask detection
LABELS = ['Mask', 'No Mask']

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess the face ROI for mask classification
        face_roi_resized = cv2.resize(face_roi, (224, 224))  # Assuming the mask model expects 224x224 input
        face_roi_normalized = face_roi_resized / 255.0  # Normalize pixel values
        face_roi_expanded = np.expand_dims(face_roi_normalized, axis=0)  # Expand dimensions for the model

        # Predict if the face is wearing a mask or not
        mask_prediction = mask_model.predict(face_roi_expanded)
        label = LABELS[np.argmax(mask_prediction)]  # Get the label (Mask or No Mask)

        # Set color for the bounding box (green for mask, red for no mask)
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
        
        # Draw the bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Put the label (Mask/No Mask) on the frame
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame
    cv2.imshow('Mask Detection', frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
