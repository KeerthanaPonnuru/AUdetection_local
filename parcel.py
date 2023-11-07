import cv2
import numpy as np

# Load YOLOv8 model and configuration
net = cv2.dnn.readNet('yolov8.weights', 'yolov8.cfg')  # Replace with your model files

# Load class names (if you have a file with class names)
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Load the image you want to detect parcels in
image = cv2.imread('parcel_image.jpg')  # Replace with your image file

# Prepare the image for YOLOv8
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get output layer names
output_layer_names = net.getUnconnectedOutLayersNames()

# Run inference
detections = net.forward(output_layer_names)

# Process detections
for detection in detections:
    for obj in detection:
        scores = obj[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:  # You can adjust the confidence threshold
            center_x = int(obj[0] * image.shape[1])
            center_y = int(obj[1] * image.shape[0])
            width = int(obj[2] * image.shape[1])
            height = int(obj[3] * image.shape[0])

            # Calculate bounding box coordinates
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            # Draw bounding box and label
            color = (0, 255, 0)  # Green color for the bounding box
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
            label = f'{classes[class_id]}: {confidence:.2f}'
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the result
cv2.imshow('Parcel Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
