import cv2

# Load the pre-trained model for object detection
net = cv2.dnn.readNet('yolov2.weights', 'yolov2.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Get the names of the output layersq
output_layer_names = net.getUnconnectedOutLayersNames()

# Open real-time camera feed
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()
    
    if not ret:
        break

    # Preprocess frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform forward pass
    outputs = net.forward(output_layer_names)

    # Process detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]

            if confidence > 0.5:
                # Get coordinates for drawing
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
                cv2.putText(frame, f'{classes[class_id]} {confidence:.2f}', (center_x - w // 2, center_y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
