import cv2
import mediapipe as mp

def CREATE_POSE_MODEL():
    return mp.solutions.pose.Pose(
    static_image_mode=False,  # Continuous webcam video
    model_complexity=1,       # Medium power
    smooth_landmarks=True     # Reduce coordinate jitter
    )

def IDENTIFY_JOINTS(pose,frame):
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #format to RGB for mediapipe
    joint_coordinates= pose.process(frameRGB)
    
    return joint_coordinates

#drawing_utils object
def CREATE_DRAW():
    return mp.solutions.drawing_utils
