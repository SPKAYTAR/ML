from Webcam import OPEN_CAM  # Opens webcam
from JointDetection import IDENTIFY_JOINTS, CREATE_POSE_MODEL, CREATE_DRAW
import cv2  # Display frames
import mediapipe as mp  # Landmark connections

cam = OPEN_CAM()
model = CREATE_POSE_MODEL()
mpDraw = CREATE_DRAW()

while True:
    success, frame = cam.read()  # Capture webcam frame

    if not success:
        break

    # Detect hands, face, and body
    joints = IDENTIFY_JOINTS(model, frame)

    # Draw left hand
    if joints.left_hand_landmarks is not None:
        mpDraw.draw_landmarks(
            frame,
            joints.left_hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS
        )

    # Draw right hand
    if joints.right_hand_landmarks is not None:
        mpDraw.draw_landmarks(
            frame,
            joints.right_hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS
        )

    # Draw selected face landmarks only
    if joints.face_landmarks is not None:
        selected_face = [10, 152, 1, 61, 291, 70, 300]

        for index in selected_face:
            landmark = joints.face_landmarks.landmark[index]

            # Convert normalized coordinates into pixel positions
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            # Draw a small circle at each selected face landmark
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Display webcam output
    cv2.imshow("Webcam", frame)

    # Exit when q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()  # Release webcam
model.close()  # Release MediaPipe resources
cv2.destroyAllWindows()  # Close display window