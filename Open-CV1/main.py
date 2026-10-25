from Webcam import OPEN_CAM               # Webcam setup
from JointDetection import IDENTIFY_JOINTS , CREATE_POSE_MODEL,CREATE_DRAW # Joint detection
import cv2
import mediapipe as mp

cam = OPEN_CAM()  # Start webcam
pose = CREATE_POSE_MODEL()
mpDraw = CREATE_DRAW()
while True:
    success, frame = cam.read()  # Capture current frame

    if not success:
        break

    joints = IDENTIFY_JOINTS(pose,frame)  # Process frame with MediaPipe
    if joints.pose_landmarks is not None:
       mpDraw.draw_landmarks(
            frame,
            joints.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS
        )
    cv2.imshow("Webcam", frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
pose.close()
cv2.destroyAllWindows()
