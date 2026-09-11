import cv2
import math
import mediapipe as mp
from ultralytics import YOLOE


# =========================================================
# AI VISION RECOGNITION - FINAL APPLICATION
# =========================================================


# -----------------------------
# MEDIAPIPE HAND SETUP
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_model = "hand_landmarker.task"

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=hand_model
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)


# -----------------------------
# YOLOE OBJECT SETUP
# -----------------------------

object_model = YOLOE("yoloe-26n-seg.pt")

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

object_model.set_classes(objects)


# -----------------------------
# FINGER COUNT
# -----------------------------

def count_fingers(lm):

    count = 0

    if lm[8].y < lm[6].y:
        count += 1

    if lm[12].y < lm[10].y:
        count += 1

    if lm[16].y < lm[14].y:
        count += 1

    if lm[20].y < lm[18].y:
        count += 1

    if lm[4].x < lm[3].x:
        count += 1

    return count


# -----------------------------
# GESTURE RECOGNITION
# -----------------------------

def detect_gesture(lm):

    index_open = lm[8].y < lm[6].y
    middle_open = lm[12].y < lm[10].y
    ring_open = lm[16].y < lm[14].y
    pinky_open = lm[20].y < lm[18].y

    thumb_up = lm[4].y < lm[3].y
    thumb_down = lm[4].y > lm[3].y

    fingers = count_fingers(lm)

    # Open hand
    if fingers == 5:
        return "OPEN HAND"

    # Peace
    if (
        index_open
        and middle_open
        and not ring_open
        and not pinky_open
    ):
        return "PEACE"

    # Point
    if (
        index_open
        and not middle_open
        and not ring_open
        and not pinky_open
    ):
        return "POINT"

    # Thumbs up
    if thumb_up and fingers == 0:
        return "THUMBS UP"

    # Thumbs down
    if thumb_down and fingers == 0:
        return "THUMBS DOWN"

    # Fist
    if fingers == 0:
        return "FIST"

    return "UNKNOWN"


# =========================================================
# START CAMERA
# =========================================================

with HandLandmarker.create_from_options(hand_options) as hand_landmarker:

    camera = cv2.VideoCapture(0)

    print("")
    print("====================================")
    print("   AI VISION RECOGNITION")
    print("====================================")
    print("Hand + Gesture + Object Recognition")
    print("Press Q to close")
    print("")

    while camera.isOpened():

        success, frame = camera.read()

        if not success:
            print("Camera could not be read.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # =================================================
        # OBJECT DETECTION
        # =================================================

        object_results = object_model.predict(
            frame,
            conf=0.50,
            verbose=False
        )

        # Draw YOLO detections
        output = object_results[0].plot()


        # =================================================
        # HAND DETECTION
        # =================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        hand_result = hand_landmarker.detect(mp_image)


        if hand_result.hand_landmarks:

            hand = hand_result.hand_landmarks[0]

            # Count fingers
            fingers = count_fingers(hand)

            # Detect gesture
            gesture = detect_gesture(hand)

            h, w, _ = output.shape


            # Draw hand landmarks
            for landmark in hand:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    output,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # Draw hand connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]

            for start, end in connections:

                x1 = int(hand[start].x * w)
                y1 = int(hand[start].y * h)

                x2 = int(hand[end].x * w)
                y2 = int(hand[end].y * h)

                cv2.line(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


            # =================================================
            # INFORMATION PANEL
            # =================================================

            cv2.putText(
                output,
                f"GESTURE: {gesture}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                f"FINGERS: {fingers}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        # =================================================
        # TITLE
        # =================================================

        cv2.putText(
            output,
            "AI VISION RECOGNITION",
            (20, output.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # Show final application
        cv2.imshow(
            "AI Vision Recognition - Final",
            output
        )


        # Q = quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    camera.release()
    cv2.destroyAllWindows()

    print("AI Vision Recognition stopped.")