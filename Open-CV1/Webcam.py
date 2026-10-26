import cv2  # OpenCV: webcam capture and image processing

#create webcam resolution 640x480
def OPEN_CAM():
    width = 640
    height = 480

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Open webcam

    # Configure camera
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, 30)

    return cam  # Return webcam object
