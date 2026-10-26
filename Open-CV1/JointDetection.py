import cv2
import mediapipe as mp

def CREATE_POSE_MODEL():
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def IDENTIFY_JOINTS(pose,frame):
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #format to RGB for mediapipe
    joint_coordinates= pose.process(frameRGB)
    
    return joint_coordinates

#drawing_utils object
def CREATE_DRAW():
    return mp.solutions.drawing_utils
