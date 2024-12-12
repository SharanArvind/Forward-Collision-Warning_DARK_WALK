import cv2
import numpy as np

# Load the pre-trained model for object detection
net = cv2.dnn.readNet('yolov2.weights', 'yolov2.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Get the names of the output layers
output_layer_names = net.getUnconnectedOutLayersNames()

# Open real-time camera feed
cap = cv2.VideoCapture(0)

# Safety threshold settings
DEFAULT_CRITICAL_THRESHOLD = 0.2  # Percentage of frame area for critical warning
DEFAULT_WARNING_THRESHOLD = 0.1   # Percentage of frame area for general warning
CENTER_PRIORITY_ZONE = 0.4  # Fraction of frame width/height for center zone

# Priority weighting for object categories
HIGH_PRIORITY_CLASSES = {'person', 'bicycle', 'car', 'motorbike', 'bus', 'truck'}

# Object tracking data
object_positions = {}  # To store previous positions of detected objects

def calculate_dynamic_thresholds(frame_area, density):
    """Dynamically scale thresholds based on frame area and object density."""
    scaling_factor = 1 + density * 0.5
    critical_threshold = DEFAULT_CRITICAL_THRESHOLD * frame_area / scaling_factor
    warning_threshold = DEFAULT_WARNING_THRESHOLD * frame_area / scaling_factor
    return critical_threshold, warning_threshold

def draw_heatmap(frame, zones, risk_levels):
    """Overlay risk heatmap on the frame."""
    overlay = frame.copy()
    alpha = 0.6  # Transparency level
    colors = {
        "low": (255, 255, 0),  # Cyan
        "medium": (0, 255, 255),  # Yellow
        "high": (0, 0, 255)  # Red
    }
    
    for zone_coords, risk_level_key in zones.items():
        # Ensure zone_coords is a tuple of two points
        if isinstance(zone_coords, tuple) and len(zone_coords) == 2:
            top_left, bottom_right = zone_coords
            color = colors.get(risk_levels[risk_level_key], (255, 255, 255))  # Default to white if key not found
            cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        else:
            print(f"Invalid zone_coords format: {zone_coords}")
    
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    # Preprocess frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform forward pass
    outputs = net.forward(output_layer_names)

    # Initialize zone risks
    zones = {
        "center": ((int(frame_width * 0.3), int(frame_height * 0.3)),
                   (int(frame_width * 0.7), int(frame_height * 0.7))),
        "left": ((0, 0), (int(frame_width * 0.3), frame_height)),
        "right": ((int(frame_width * 0.7), 0), (frame_width, frame_height))
    }
    risk_levels = {zone: "low" for zone in zones}

    # Calculate object density
    object_count = sum(len(output) for output in outputs)
    density = object_count / (frame_width * frame_height)

    # Get dynamic thresholds
    critical_threshold, warning_threshold = calculate_dynamic_thresholds(frame_area, density)

    # Process detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]

            if confidence > 0.5:
                # Get bounding box coordinates
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)

                bbox_area = w * h

                # Update object tracking
                obj_id = f"{center_x}-{center_y}"  # Basic ID from position
                if obj_id in object_positions:
                    prev_x, prev_y = object_positions[obj_id]
                    dx, dy = center_x - prev_x, center_y - prev_y
                    motion_direction = "towards" if dx**2 + dy**2 < bbox_area else "away"
                else:
                    motion_direction = "static"
                object_positions[obj_id] = (center_x, center_y)

                # Determine zone and risk
                object_name = classes[class_id]
                zone = "center" if (zones["center"][0][0] < center_x < zones["center"][1][0] and
                                    zones["center"][0][1] < center_y < zones["center"][1][1]) else (
                        "left" if center_x < zones["center"][0][0] else "right")

                if bbox_area > critical_threshold and object_name in HIGH_PRIORITY_CLASSES:
                    risk_levels[zone] = "high"
                    warning_message = f'CRITICAL: {object_name.upper()} IN {zone.upper()} ZONE!'
                elif bbox_area > warning_threshold:
                    risk_levels[zone] = "medium"
                    warning_message = f'WARNING: {object_name} detected in {zone} zone'
                else:
                    warning_message = None

                # Display warnings
                if warning_message:
                    cv2.putText(frame, warning_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print(warning_message)

                # Draw bounding box and label
                cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2),
                              (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
                cv2.putText(frame, f'{object_name} {confidence:.2f}',
                            (center_x - w // 2, center_y - h // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Overlay heatmap
    frame = draw_heatmap(frame, zones, risk_levels)

    # Display the frame
    cv2.imshow('Advanced Object Detection', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
