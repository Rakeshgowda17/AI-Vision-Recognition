import cv2
import mediapipe as mp
import math

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)


# Calculate distance between two points
def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


# Detect gesture
def detect_gesture(lm):

    # Finger states
    index_open = lm[8].y < lm[6].y
    middle_open = lm[12].y < lm[10].y
    ring_open = lm[16].y < lm[14].y
    pinky_open = lm[20].y < lm[18].y

    # Thumb position
    thumb_up = lm[4].y < lm[3].y
    thumb_down = lm[4].y > lm[3].y

    # Count fingers
    fingers = sum([
        index_open,
        middle_open,
        ring_open,
        pinky_open
    ])

    # --------------------------------
    # OPEN HAND
    # --------------------------------
    if fingers == 4 and thumb_up:
        return "OPEN HAND"

    # --------------------------------
    # PEACE ✌️
    # --------------------------------
    if index_open and middle_open and not ring_open and not pinky_open:
        return "PEACE"

    # --------------------------------
    # POINT ☝️
    # --------------------------------
    if index_open and not middle_open and not ring_open and not pinky_open:
        return "POINT"

    # --------------------------------
    # FIST ✊
    # --------------------------------
    if fingers == 0:
        return "FIST"

    # --------------------------------
    # THUMBS UP 👍
    # --------------------------------
    if thumb_up and fingers == 0:
        return "THUMBS UP"

    # --------------------------------
    # THUMBS DOWN 👎
    # --------------------------------
    if thumb_down and fingers == 0:
        return "THUMBS DOWN"

    return "UNKNOWN"


# Start MediaPipe
with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    print("Camera started!")
    print("Show different hand gestures.")
    print("Press Q to close.")

    while camera.isOpened():

        success, frame = camera.read()

        if not success:
            print("Could not read camera.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:

            hand_landmarks = result.hand_landmarks[0]

            # Detect gesture
            gesture = detect_gesture(hand_landmarks)

            # Frame size
            h, w, _ = frame.shape

            # Draw landmarks
            for landmark in hand_landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Display gesture
            cv2.putText(
                frame,
                f"GESTURE: {gesture}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

        # Display window
        cv2.imshow(
            "AI Vision - Gesture Recognition",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()