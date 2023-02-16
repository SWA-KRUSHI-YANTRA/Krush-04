import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('cotton_detection_model.h5')

# Set up the video capture device
cap = cv2.VideoCapture(0)

# Loop over the frames
while True:
    # Capture the frame
    ret, frame = cap.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (299, 299))
    frame = np.expand_dims(frame, axis=0) / 255.0

    # Make a prediction
    prediction = model.predict(frame)

    # Draw the result on the frame
    if prediction > 0.5:
        cv2.putText(frame, 'Cotton detected', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'No cotton detected', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('frame', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close the windows
cap.release()
cv2.destroyAllWindows()
