from ultralytics import YOLOE
import cv2

# Load YOLOE model
model = YOLOE("yoloe-26n-seg.pt")

# Objects to recognize
objects = [
    "person",
    "cell phone",
    "laptop",
    "computer",
    "keyboard",
    "mouse",
    "monitor",
    "book",
    "calendar",
    "pen",
    "pencil",
    "bottle",
    "cup",
    "watch",
    "headphones",
    "backpack",
    "chair",
    "table",
    "remote control",
    "wallet",
    "glasses",
    "keys",
    "television",
    "car",
    "dog",
    "cat"
]

# Set object vocabulary
model.set_classes(objects)

# Start webcam
camera = cv2.VideoCapture(0)

print("AI Vision - Smart Object Recognition")
print("Show an object to the camera.")
print("Press Q to close.")

while camera.isOpened():

    success, frame = camera.read()

    if not success:
        print("Could not read camera.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Object detection
    results = model.predict(
        frame,
        conf=0.50,
        verbose=False
    )

    # Draw only confident detections
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow(
        "AI Vision - Smart Object Recognition",
        annotated_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()