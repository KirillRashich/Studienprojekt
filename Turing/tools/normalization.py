from tools.data_preprocessor import data_preprocessor
from pathlib import Path
import numpy as np
import pandas as pd


class Normalize:
    """                  MIN ANGLE     MAX ANGLE
    "LShoulderPitch" = [  -2.0857   ,    2.0857 ]
    "LShoulderRoll"  = [   0.0087   ,    1.5620 ]
    "RShoulderPitch" = [  -2.0857   ,    2.0857 ]
    "RShoulderRoll"  = [  -1.5620   ,   -0.0087 ]
    "LElbowYaw"      = [  -2.0857   ,    2.0857 ]
    "LElbowRoll"     = [  -1.5620   ,   -0.0087 ]
    "RElbowYaw"      = [  -2.0857   ,    2.0857 ]
    "RElbowRoll"     = [   0.0087  ,     1.5620 ]
    """
    L_SHOULDER_PITCH_MIN = -2.0857
    L_SHOULDER_PITCH_MAX =  2.0857
    L_SHOULDER_ROLL_MIN  =  0.0087
    L_SHOULDER_ROLL_MAX  =  1.5620
    R_SHOULDER_PITCH_MIN = -2.0857
    R_SHOULDER_PITCH_MAX =  2.0857
    R_SHOULDER_ROLL_MIN  = -1.5620
    R_SHOULDER_ROLL_MAX  = -0.0087
    L_ELBOW_YAW_MIN      = -2.0857
    L_ELBOW_YAW_MAX      =  2.0857
    L_ELBOW_ROLL_MIN     = -1.5620
    L_ELBOW_ROLL_MAX     = -0.0087
    R_ELBOW_YAW_MIN      = -2.0857
    R_ELBOW_YAW_MAX      =  2.0857
    R_ELBOW_ROLL_MIN     =  0.0087
    R_ELBOW_ROLL_MAX     =  1.5620

    def set_in_range(self, data_frame):

        for row in range(data_frame.shape[0]):
            if(data_frame.iloc[row,0] > self.L_SHOULDER_PITCH_MAX):
                data_frame.iloc[row,0] = self.L_SHOULDER_PITCH_MAX
            if(data_frame.iloc[row,0]<self.L_SHOULDER_PITCH_MIN):
                data_frame.iloc[row,0]= self.L_SHOULDER_PITCH_MIN

            if (data_frame.iloc[row, 1] > self.L_SHOULDER_ROLL_MAX):
                data_frame.iloc[row, 1] = self.L_SHOULDER_ROLL_MAX
            if (data_frame.iloc[row, 1] < self.L_SHOULDER_ROLL_MIN):
                data_frame.iloc[row, 1] = self.L_SHOULDER_ROLL_MIN

            if (data_frame.iloc[row, 2] > self.R_SHOULDER_PITCH_MAX):
                data_frame.iloc[row, 2] = self.R_SHOULDER_PITCH_MAX
            if (data_frame.iloc[row, 2] < self.R_SHOULDER_PITCH_MIN):
                data_frame.iloc[row, 2] = self.R_SHOULDER_PITCH_MIN

            if (data_frame.iloc[row, 3] > self.R_SHOULDER_ROLL_MAX):
                data_frame.iloc[row, 3] = self.R_SHOULDER_ROLL_MAX
            if (data_frame.iloc[row, 3] < self.R_SHOULDER_ROLL_MIN):
                data_frame.iloc[row, 3] = self.R_SHOULDER_ROLL_MIN

            if (data_frame.iloc[row, 4] > self.L_ELBOW_YAW_MAX):
                data_frame.iloc[row, 4] = self.L_ELBOW_YAW_MAX
            if (data_frame.iloc[row, 4] < self.L_ELBOW_YAW_MIN):
                data_frame.iloc[row, 4] = self.L_ELBOW_YAW_MIN

            if (data_frame.iloc[row, 5] > self.L_ELBOW_ROLL_MAX):
                data_frame.iloc[row, 5] = self.L_ELBOW_ROLL_MAX
            if (data_frame.iloc[row, 5] < self.L_ELBOW_ROLL_MIN):
                data_frame.iloc[row, 5] = self.L_ELBOW_ROLL_MIN

            if (data_frame.iloc[row, 6] > self.R_ELBOW_YAW_MAX):
                data_frame.iloc[row, 6] = self.R_ELBOW_YAW_MAX
            if (data_frame.iloc[row, 6] < self.R_ELBOW_YAW_MIN):
                data_frame.iloc[row, 6] = self.R_ELBOW_YAW_MIN

            if (data_frame.iloc[row, 7] > self.R_ELBOW_ROLL_MAX):
                data_frame.iloc[row, 7] = self.R_ELBOW_ROLL_MAX
            if (data_frame.iloc[row, 7] < self.R_ELBOW_ROLL_MIN):
                data_frame.iloc[row, 7] = self.R_ELBOW_ROLL_MIN
        return data_frame


    def normalization(self, data_frame):
        data_frame = self.set_in_range(data_frame)

        for row in range(data_frame.shape[0]):
            data_frame.iloc[row, 0] = (data_frame.iloc[row, 0] - self.L_SHOULDER_PITCH_MIN )/ (self.L_SHOULDER_PITCH_MAX - self.L_SHOULDER_PITCH_MIN)
            data_frame.iloc[row, 1] = (data_frame.iloc[row, 1] - self.L_SHOULDER_ROLL_MIN)  / (self.L_SHOULDER_ROLL_MAX - self.L_SHOULDER_ROLL_MIN)
            data_frame.iloc[row, 2] = (data_frame.iloc[row, 2] - self.R_SHOULDER_PITCH_MIN) / ( self.R_SHOULDER_PITCH_MAX - self.R_SHOULDER_PITCH_MIN)
            data_frame.iloc[row, 3] = (data_frame.iloc[row, 3] - self.R_SHOULDER_ROLL_MIN) / ( self.R_SHOULDER_ROLL_MAX - self.R_SHOULDER_ROLL_MIN)
            data_frame.iloc[row, 4] = (data_frame.iloc[row, 4] - self.L_ELBOW_YAW_MIN) / ( self.L_ELBOW_YAW_MAX - self.L_ELBOW_YAW_MIN)
            data_frame.iloc[row, 5] = (data_frame.iloc[row, 5] - self.L_ELBOW_ROLL_MIN) / ( self.L_ELBOW_ROLL_MAX - self.L_ELBOW_ROLL_MIN)
            data_frame.iloc[row, 6] = (data_frame.iloc[row, 6] - self.R_ELBOW_YAW_MIN) / ( self.R_ELBOW_YAW_MAX - self.R_ELBOW_YAW_MIN)
            data_frame.iloc[row, 7] = (data_frame.iloc[row, 7] - self.R_ELBOW_ROLL_MIN) / ( self.R_ELBOW_ROLL_MAX - self.R_ELBOW_ROLL_MIN)

        return data_frame

    def denormalization(self, data_frame):

        for row in range(data_frame.shape[0]):
            data_frame.iloc[row, 0] = data_frame.iloc[row, 0] * (self.L_SHOULDER_PITCH_MAX - self.L_SHOULDER_PITCH_MIN) + self.L_SHOULDER_PITCH_MIN
            data_frame.iloc[row, 1] = data_frame.iloc[row, 1] * (self.L_SHOULDER_ROLL_MAX - self.L_SHOULDER_ROLL_MIN) + self.L_SHOULDER_ROLL_MIN
            data_frame.iloc[row, 2] = data_frame.iloc[row, 2] * (self.R_SHOULDER_PITCH_MAX - self.R_SHOULDER_PITCH_MIN) + self.R_SHOULDER_PITCH_MIN
            data_frame.iloc[row, 3] = data_frame.iloc[row, 3] * (self.R_SHOULDER_ROLL_MAX - self.R_SHOULDER_ROLL_MIN) + self.R_SHOULDER_ROLL_MIN
            data_frame.iloc[row, 4] = data_frame.iloc[row, 4] * (self.L_ELBOW_YAW_MAX - self.L_ELBOW_YAW_MIN) + self.L_ELBOW_YAW_MIN
            data_frame.iloc[row, 5] = data_frame.iloc[row, 5] * (self.L_ELBOW_ROLL_MAX - self.L_ELBOW_ROLL_MIN) + self.L_ELBOW_ROLL_MIN
            data_frame.iloc[row, 6] = data_frame.iloc[row, 6] * (self.R_ELBOW_YAW_MAX - self.R_ELBOW_YAW_MIN) + self.R_ELBOW_YAW_MIN
            data_frame.iloc[row, 7] = data_frame.iloc[row, 7] * (self.R_ELBOW_ROLL_MAX - self.R_ELBOW_ROLL_MIN) + self.R_ELBOW_ROLL_MIN

        return data_frame



###############DELETE AFTER CODING###################
#load .csv data in angles format
#dp = data_preprocessor()
#dataFrame = dp.loadAnglesAsCSV('video_1_forward.csv')
#print(dataFrame)
#normalizer = Normalize()
#normalized = normalizer.normalization(dataFrame)
#denormalized = normalizer.denormalization(normalized)
#print(denormalized)
###############DELETE AFTER CODING###################