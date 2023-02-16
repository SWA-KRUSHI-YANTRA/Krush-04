import cv2
import numpy as np
import time
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path_or_model='path/to/cotton_weights.pt')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to Torch tensor
    img = torch.from_numpy(frame).to('cuda').float()

    # Normalize pixel values
    img /= 255.0

    # Inference on the model
    tic = time.time()
    results = model(img, size=640)
    toc = time.time()

    # Process the detected objects
    boxes = results.xyxy[0].detach().cpu().numpy()
    confidences = boxes[:, 4]
    class_ids = boxes[:, 5].astype(int)
    boxes = boxes[:, :4]

    # Apply non-maximum suppression to remove redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw boxes and labels on the frame
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(model.names), 3))
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = model.names[class_ids[i]]
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}",
                        (int(x), int(y) - 5), font, 1, color, 2)

    # Display the resulting frame
    cv2.imshow("Cotton Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
