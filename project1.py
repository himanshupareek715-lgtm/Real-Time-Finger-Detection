import math
import cv2
import mediapipe as mp

from collections import Counter

from mediapipe.tasks import python
from mediapipe.tasks.python import vision



# -----------------------------
# MediaPipe setup
# -----------------------------

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,

    # Detection stability
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# -----------------------------
# Calculate angle
# -----------------------------

def calculate_angle(a, b, c):

    ba = (
        a.x - b.x,
        a.y - b.y
    )

    bc = (
        c.x - b.x,
        c.y - b.y
    )

    dot_product = (
        ba[0] * bc[0]
        +
        ba[1] * bc[1]
    )

    magnitude_ba = math.sqrt(
        ba[0] ** 2 +
        ba[1] ** 2
    )

    magnitude_bc = math.sqrt(
        bc[0] ** 2 +
        bc[1] ** 2
    )

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    value = dot_product / (
        magnitude_ba * magnitude_bc
    )

    # Prevent math domain errors
    value = max(-1, min(1, value))

    angle = math.acos(value)

    return math.degrees(angle)


# -----------------------------
# Count fingers
# -----------------------------

def count_fingers(hand):

    count = 0

    fingers = [
        (5, 6, 7, 8),       # Index
        (9, 10, 11, 12),    # Middle
        (13, 14, 15, 16),   # Ring
        (17, 18, 19, 20)    # Pinky
    ]

    for mcp, pip, dip, tip in fingers:

        pip_angle = calculate_angle(
            hand[mcp],
            hand[pip],
            hand[dip]
        )

        dip_angle = calculate_angle(
            hand[pip],
            hand[dip],
            hand[tip]
        )

        if pip_angle > 160 and dip_angle > 160:
            count += 1

    # Thumb

    thumb_angle = calculate_angle(
        hand[1],
        hand[2],
        hand[4]
    )

    if thumb_angle > 150:
        count += 1

    return count


# -----------------------------
# Webcam
# -----------------------------

cap = cv2.VideoCapture(0)

count_history = []

frame_timestamp = 0


# -----------------------------
# Main loop
# -----------------------------

while True:

    success, img = cap.read()

    if not success:
        break

    # Mirror webcam
    img = cv2.flip(img, 1)

    # Increase timestamp
    frame_timestamp += 33

    # Convert image for MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img
    )

    # Detect hand in VIDEO mode
    result = detector.detect_for_video(
        mp_image,
        frame_timestamp
    )

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        h, w, _ = img.shape


        # -----------------------------
        # Handedness
        # -----------------------------

        handedness = result.handedness[0][0].category_name

        # Because webcam is mirrored,
        # swap MediaPipe's label
        if handedness == "Left":
            handedness = "Right"
        else:
            handedness = "Left"


        # -----------------------------
        # Finger count
        # -----------------------------

        count = count_fingers(hand)

        count_history.append(count)

        if len(count_history) > 7:
            count_history.pop(0)

        stable_count = Counter(
            count_history
        ).most_common(1)[0][0]


        # -----------------------------
        # Draw landmarks
        # -----------------------------

        for landmark in hand:

            x = int(
                landmark.x * w
            )

            y = int(
                landmark.y * h
            )

            cv2.circle(
                img,
                (x, y),
                6,
                (0, 255, 0),
                -1
            )


        # -----------------------------
        # Hand connections
        # -----------------------------

        connections = [

            # Thumb
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),

            # Index
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),

            # Middle
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),

            # Ring
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),

            # Pinky
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),

            # Palm
            (5, 9),
            (9, 13),
            (13, 17)
        ]


        for start, end in connections:

            x1 = int(
                hand[start].x * w
            )

            y1 = int(
                hand[start].y * h
            )

            x2 = int(
                hand[end].x * w
            )

            y2 = int(
                hand[end].y * h
            )

            cv2.line(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


        # -----------------------------
        # Information box
        # -----------------------------

        cv2.rectangle(
            img,
            (10, 10),
            (280, 125),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            img,
            f"Fingers: {stable_count}",
            (25, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"Hand: {handedness}",
            (25, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # -----------------------------
    # Display
    # -----------------------------

    cv2.imshow(
        "Real-Time Finger Detection",
        img
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------

cap.release()

cv2.destroyAllWindows()