import math
import cv2 as cv
import mediapipe as mp
import numpy as np


from real_time_control.robot_simulation import Simulation

cap = cv.VideoCapture(0)

#Load drawing untility and pose untility
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

#Start simulation
simulation = Simulation()




def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle



while True:
    success, img = cap.read()

    #Convert color system to fit into mediapipe (OpenCv support BGR-format, MediaPipe support RGB-format)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    #print(results.pose_landmarks)
    if(results.pose_landmarks):
        landmarks = results.pose_landmarks.landmark

        #X-Y Plane
        l_shoulder_XY = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_hip_XY = [landmarks[mpPose.PoseLandmark.LEFT_HIP.value].x, landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y]
        l_elbow_XY = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist_XY = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y]

        #Y-Z Plane
        l_shoulder_YZ = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].z]
        l_hip_YZ = [landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y, landmarks[mpPose.PoseLandmark.LEFT_HIP.value].z]
        l_elbow_YZ = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].z]
        l_wrist_YZ = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].z]


        #calculate angles according to movement borders of joints
        l_elbow_rad = math.radians(-(180 - float(calculate_angle(l_shoulder_XY, l_elbow_XY, l_wrist_XY))))
        l_shoulder_rad = math.radians(float(calculate_angle(l_hip_XY, l_shoulder_XY, l_elbow_XY)))
        l_shoulder_pitch_rad = math.radians(180 - float(calculate_angle(l_hip_YZ, l_shoulder_YZ, l_elbow_YZ)) - 119)
        print("\n l_elbow_rad " , l_elbow_rad)
        print("\n l_shoulder_roll_rad "  , l_shoulder_rad)
        print("\n l_shoulder_pitch_rad " , l_shoulder_pitch_rad)


        #send angles to simulation
        if(l_elbow_rad > -1.5620 and l_elbow_rad < -0.0087):
            simulation.pepper.setAngles("LElbowRoll", l_elbow_rad, 1)
        if (l_shoulder_rad > 0.0087 and l_shoulder_rad < 1.5620):
            simulation.pepper.setAngles("LShoulderRoll", l_shoulder_rad, 1)
        if ( l_shoulder_pitch_rad > -2.0857 and  l_shoulder_pitch_rad < 2.0857):
            simulation.pepper.setAngles("LShoulderPitch",  l_shoulder_pitch_rad, 1)

        #Print angles (degrees)
        cv.putText(img, str(l_elbow_rad),
                    tuple(np.multiply(l_elbow_XY, [640, 480]).astype(int)),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

        cv.putText(img, str(l_shoulder_rad),
                   tuple(np.multiply(l_shoulder_YZ, [640, 480]).astype(int)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)



    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cv.imshow('WebCam pose', img)
    cv.waitKey(1)

simulation.closeSimulation()

