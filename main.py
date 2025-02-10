import cv2
import numpy as np
import tensorflow as tf

# Step 1: Load the pre-trained model
print("Loading the pre-trained model...")
model_path = "ssd_mobilenet_v2_coco_2018_03_29/saved_model"
model = tf.saved_model.load(model_path)

# Step 2: Initialize the webcam capture
print("Starting the webcam...")
cap = cv2.VideoCapture(0)

# Step 3: Define the class labels
print("Defining the class labels...")
class_labels = ["person"]

# Step 4: Initialize the variables for counting people
print("Initializing the variables...")
highest_count = 0  # highest count of people in current cycle
count = 0  # current count
frame_count = 0

# Step 5: Start the video capture loop
print("Starting the video capture loop...")
while True:
    frame_count += 1
    # Step 6: Read the frame from the webcam
    ret, frame = cap.read()

    # Step 7: Preprocess the frame for the model
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Step 8: Perform object detection
    detections = model.signatures["serving_default"](input_tensor)

    # Step 9: Process the detections
    num_detections = int(detections.pop("num_detections"))
    detections = {
        key: value[0, :num_detections].numpy() for key, value in detections.items()
    }
    detections["num_detections"] = num_detections

    # Step 10: Draw bounding boxes around detected people
    for i in range(num_detections):
        class_id = int(detections["detection_classes"][i])
        if class_id == 1:  # Class label for 'person'
            score = detections["detection_scores"][i]
            if score > 0.3:  # Set a threshold for detection confidence
                bbox = detections["detection_boxes"][i]
                ymin, xmin, ymax, xmax = bbox
                h, w, _ = frame.shape
                xmin = int(xmin * w)
                xmax = int(xmax * w)
                ymin = int(ymin * h)
                ymax = int(ymax * h)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                count += 1

    # Step 11: Display the frame
    cv2.imshow("Object Detection", frame)

    # Step 12: Count the number of people
    if (
        count > highest_count
    ):  # only update the count if the number of people has increased (within the current cycle)
        highest_count = count
        print(f"Number of people detected: {highest_count}")
        
        # Write the value of highest_count to a text file
        with open("./shared_data.txt", "w") as file:
            file.write(str(highest_count))
    if frame_count == 150:  # how often we want to reset our count
        highest_count = 0
        frame_count = 0

    # Reset the count
    count = 0

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) == ord("q"):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
