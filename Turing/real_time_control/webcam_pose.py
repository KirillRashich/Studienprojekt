import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2 as cv
import mediapipe as mp
import numpy as np
from tools.data_preprocessor import data_preprocessor
import pandas as pd


from robot_simulation import Simulation

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

        # DataFrame for data collection
        col = []
        for i in range(0, 32):
            col.extend([str(i) + ':x', str(i) + ':y', str(i) + ':z', str(i) + ':visibility'])
        data = pd.DataFrame([], columns=col)  # data_joint_positions

        # Reshape Data
        curr_landmark = []
        for i in range(0, 32):
            curr_landmark.extend([landmarks[i].x, landmarks[i].y, landmarks[i].z, landmarks[i].visibility])

        data.loc[len(data.index)] = curr_landmark

        #X-Y Plane
        #l_shoulder_XY = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
        #l_hip_XY = [landmarks[mpPose.PoseLandmark.LEFT_HIP.value].x, landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y]
        #l_elbow_XY = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
        #l_wrist_XY = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y]

        #Y-Z Plane
        #l_shoulder_YZ = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].z]
        #l_hip_YZ = [landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y, landmarks[mpPose.PoseLandmark.LEFT_HIP.value].z]
        #l_elbow_YZ = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].z]
        #l_wrist_YZ = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].z]


        #calculate angles according to movement borders of joints
        #l_elbow_rad = math.radians(-(180 - float(calculate_angle(l_shoulder_XY, l_elbow_XY, l_wrist_XY))))
        #l_shoulder_rad = math.radians(float(calculate_angle(l_hip_XY, l_shoulder_XY, l_elbow_XY)))
        #l_shoulder_pitch_rad = math.radians(180 - float(calculate_angle(l_hip_YZ, l_shoulder_YZ, l_elbow_YZ)) - 119)
        #print("\n l_elbow_rad " , l_elbow_rad)
        #print("\n l_shoulder_roll_rad "  , l_shoulder_rad)
        #print("\n l_shoulder_pitch_rad " , l_shoulder_pitch_rad)

        preprocessor = data_preprocessor()
        keypoints = preprocessor.mediapipe_to_openpose(data)
        angles = preprocessor.openpose_keypoints_to_angles(keypoints)

        for x in range(angles.shape[0]):
            simulation.pepper.setAngles("LShoulderPitch", angles['LShoulder'][x][0], 1)
            simulation.pepper.setAngles("LShoulderRoll", angles['LShoulder'][x][1], 1)
            simulation.pepper.setAngles("RShoulderPitch", angles['RShoulder'][x][0], 1)
            simulation.pepper.setAngles("RShoulderRoll", angles['RShoulder'][x][1], 1)
            simulation.pepper.setAngles("LElbowYaw", angles['LElbow'][x][0], 1)
            simulation.pepper.setAngles("LElbowRoll", angles['LElbow'][x][1], 1)
            simulation.pepper.setAngles("RElbowYaw", angles['RElbow'][x][0], 1)
            simulation.pepper.setAngles("RElbowRoll", angles['RElbow'][x][1], 1)


        #send angles to simulation
        #simulation.pepper.setAngles("LElbowRoll", l_elbow_rad, 1)
        #simulation.pepper.setAngles("LShoulderRoll", l_shoulder_rad, 1)
        #simulation.pepper.setAngles("LShoulderPitch",  l_shoulder_pitch_rad, 1)

        #Print angles (degrees)
        #cv.putText(img, str(l_elbow_rad),
        #            tuple(np.multiply(l_elbow_XY, [640, 480]).astype(int)),
        #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

        #cv.putText(img, str(l_shoulder_rad),
        #           tuple(np.multiply(l_shoulder_YZ, [640, 480]).astype(int)),
        #           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)



    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cv.imshow('WebCam pose', img)
    cv.waitKey(1)

simulation.closeSimulation()

