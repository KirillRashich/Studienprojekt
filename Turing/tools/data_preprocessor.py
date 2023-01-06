import os
import pandas as pd
from tools.KeypointsToAngles import KeypointsToAngles
import logging
import traceback
from pathlib import Path



class data_preprocessor:
    """This class provides some possibilities to prepare data given in .csv format, that contains information about
       32 joints for further using of this data as training and test data set for our models.
    """

    def __init__(self):
        self.cwd = os.getcwd()

    """This function becomes data, that are in rare format, it means, that each joint has 4 corresponding
       columns (X, Y, Z, visibility) and gets back (X, Y, Z) for each joint. Main goal of this function is mapping
       of all necessary joints to open-pose (openpose is an analog of media-pipe): Neck, Right Shoulder, Right Elbow,
       Right Wrist, Left Shoulder, Left Elbow, Left Wrist and Mid Heap (other joints are excluded, because they are not
       needed, because Pepper does not have legs and these joints are not observed in our further model training).
       Mid Heap is calculated from joints that represent left and right heaps. 

                           Parameters:
                           data - rare data in media-pipe format (result of data_extraction.py that is
                                  stored in data/rare-data)
                           
                           Returns:
                           openpose_df - data frame in openpose format
    """

    def mediapipe_to_openpose(self, data):
        #openpose_df = pd.DataFrame([])

        #Iterate each joint and map its (X, Y, Z) coordinates to openpose
        #for i in range(32):
        #    openpose_df[str(i) + ':x'] = data[str(i) + ':x']
        #    openpose_df[str(i) + ':y'] = data[str(i) + ':y']
        #    openpose_df[str(i) + ':z'] = data[str(i) + ':z']

        # Mediapipe to OpenPose mapping
        openpose_df = pd.DataFrame()
        openpose_df['Neck:x'] = (data['11:x'] + data['12:x']) / 2
        openpose_df['Neck:y'] = (data['11:y'] + data['12:y']) / 2
        openpose_df['Neck:z'] = (data['11:z'] + data['12:z']) / 2

        openpose_df['RShoulder:x'] = data['12:x']
        openpose_df['RShoulder:y'] = data['12:y']
        openpose_df['RShoulder:z'] = data['12:z']

        openpose_df['RElbow:x'] = data['14:x']
        openpose_df['RElbow:y'] = data['14:y']
        openpose_df['RElbow:z'] = data['14:z']

        openpose_df['RWrist:x'] = data['16:x']
        openpose_df['RWrist:y'] = data['16:y']
        openpose_df['RWrist:z'] = data['16:z']

        openpose_df['LShoulder:x'] = data['11:x']
        openpose_df['LShoulder:y'] = data['11:y']
        openpose_df['LShoulder:z'] = data['11:z']

        openpose_df['LElbow:x'] = data['13:x']
        openpose_df['LElbow:y'] = data['13:y']
        openpose_df['LElbow:z'] = data['13:z']

        openpose_df['LWrist:x'] = data['15:x']
        openpose_df['LWrist:y'] = data['15:y']
        openpose_df['LWrist:z'] = data['15:z']

        openpose_df['MidHip:x'] = (data['23:x'] + data['24:x']) / 2
        openpose_df['MidHip:y'] = (data['23:y'] + data['24:y']) / 2
        openpose_df['MidHip:z'] = (data['23:z'] + data['24:z']) / 2

        return openpose_df


    """This function becomes data from mediapipe_to_openpose() in openpose format (see description for
       mediapipe_to_openpose() for more details). Than it converts absolute real-wold (X,Y,X) of each joint into 
       Pitch, Roll and Yaw angles using KeypointsToAngles.py script. Pitch, Roll and Yaw angles represent what angles
       should have each joint component to reach the absolute real-world (X, Y, Z) positions by Pepper. For more
       information about joints to angles convertation see the description of KeypointsToAngles.py 

                             Parameters:
                             data - DataFrame in openpose format (result of mediapipe_to_openpose())
                             Returns:
                             data_angles - DataFrame, that contains {Left Shoulder Pitch, Left Shoulder Roll,
                             Right Shoulder Pitch, Right Shoulder Roll, Left Elbow Yaw, Right Elbow_Yaw,
                             Right Elbow Roll}. Each of these entities is represented as column.
    """
    def openpose_keypoints_to_angles(self, data):
        keypoints_to_angles = KeypointsToAngles()

        data_angles = pd.DataFrame()
        LShoulderPitchRoll_angles = []
        RShoulderPitchRoll_angles = []
        LElbowYawRoll_angle = []
        RElbowYawRoll_angle = []

        #Calculate joints angles using KeypointsToAngles.py
        for x in range(data.shape[0]):
            LShoulderPitchRoll_angles.append(keypoints_to_angles.obtain_LShoulderPitchRoll_angles(
                [data['Neck:x'][x], data['Neck:y'][x], data['Neck:z'][x]],
                [data['LShoulder:x'][x], data['LShoulder:y'][x],data['LShoulder:z'][x]],
                [data['LElbow:x'][x], data['LElbow:y'][x], data['LElbow:z'][x]],
                [data['MidHip:x'][x], data['MidHip:y'][x], data['MidHip:z'][x]]))

            RShoulderPitchRoll_angles.append(keypoints_to_angles.obtain_RShoulderPitchRoll_angles(
                [data['Neck:x'][x], data['Neck:y'][x], data['Neck:z'][x]],
                [data['RShoulder:x'][x], data['RShoulder:y'][x], data['RShoulder:z'][x]],
                [data['RElbow:x'][x], data['RElbow:y'][x], data['RElbow:z'][x]],
                [data['MidHip:x'][x], data['MidHip:y'][x], data['MidHip:z'][x]]))

            LElbowYawRoll_angle.append(keypoints_to_angles.obtain_LElbowYawRoll_angle(
                [data['Neck:x'][x], data['Neck:y'][x], data['Neck:z'][x]],
                [data['LShoulder:x'][x], data['LShoulder:y'][x], data['LShoulder:z'][x]],
                [data['LElbow:x'][x], data['LElbow:y'][x], data['LElbow:z'][x]],
                [data['LWrist:x'][x], data['LWrist:y'][x], data['LWrist:z'][x]]))

            RElbowYawRoll_angle.append(keypoints_to_angles.obtain_RElbowYawRoll_angle(
                [data['Neck:x'][x], data['Neck:y'][x], data['Neck:z'][x]],
                [data['RShoulder:x'][x], data['RShoulder:y'][x], data['RShoulder:z'][x]],
                [data['RElbow:x'][x], data['RElbow:y'][x], data['RElbow:z'][x]],
                [data['RWrist:x'][x], data['RWrist:y'][x], data['RWrist:z'][x]]))




        #Save Roll, Pitch and Yaw angles of left and right hand joints as DataFrame

        data_angles['LShoulder_Pitch'] = pd.DataFrame(LShoulderPitchRoll_angles).iloc[:, 0].tolist()
        data_angles['LShoulder_Roll'] =  pd.DataFrame(LShoulderPitchRoll_angles).iloc[:, 1].tolist()
        data_angles['RShoulder_Pitch']= pd.DataFrame(RShoulderPitchRoll_angles).iloc[:, 0].tolist()
        data_angles['RShoulder_Roll'] =  pd.DataFrame(RShoulderPitchRoll_angles).iloc[:, 1].tolist()
        data_angles['LElbow_Yaw'] =      pd.DataFrame(LElbowYawRoll_angle).iloc[:, 0].tolist()
        data_angles['LElbow_Roll'] =     pd.DataFrame(LElbowYawRoll_angle).iloc[:, 1].tolist()
        data_angles['RElbow_Yaw'] =      pd.DataFrame(RElbowYawRoll_angle).iloc[:, 0].tolist()
        data_angles['RElbow_Roll'] =     pd.DataFrame(RElbowYawRoll_angle).iloc[:, 1].tolist()
        return data_angles





    """Save DataFrame, that represents Roll, Pitch and Yaw angles of left and right hand joints as .csv file. 
       This DataFrame is a result of openpose_keypoints_to_angles() function. 

                                 Parameters:
                                 dataFrame - DataFrame, that is result of openpose_keypoints_to_angles() function.
                                 save_as_name  - name, that the saved file will have
                                 path - path where the data will be saved. (default path 
                                        is: data/data-in-angles-format folder)
        """
    def saveAnglesAsCSV(self, dataFrame, save_as_name, path = 'data\\data-in-angles-format\\'):
        try:
            path = str(Path(__file__).parent.parent)+'\\'+path+save_as_name
            dataFrame.to_csv(path)
            print('Data Frame saved in', path)
        except Exception:
            logging.error('Error by saving of DataFrame' + traceback.format_exc())





    """Load .csv data, that represents Roll, Pitch and Yaw angles of left and right hand joints. 
       This .csv data is a result of saveAnglesAsCSV() function.

                                 Parameters:
                                 data_name - csv. file, that should be loaded.
                                 path - path, where the loaded data can be find (default path is data/data-in-angles-format).
                                 
                                 Returns:
                                 data_in_angles_format - DataFrame, that represents Roll, Pitch and Yaw angles of left
                                 and right hand joints.
        """
    def loadAnglesAsCSV(self, data_name, path ='data\\data-in-angles-format\\'):
        try:
            path = str(Path(__file__).parent.parent)+'\\'+path
            data_in_angles_format = pd.read_csv(path + data_name, index_col= 0)
            return data_in_angles_format
        except Exception:
            logging.error('Error by loading of Dataframe'+traceback.format_exc())
            return 0





    """This function takes names of the .csv files, that contain Pitch, Yaw and Roll angles of hand joints. These .csv 
       files were saved by saveAnglesAsCSV(). Than this function uses loadAnglesAsCSV() to load corresponding .csv files
       and concatenates these DataFrames into one DataFrame for Training Set (train_X) and one DataFrame for Training
       Set Labels. In our model we will use Pitch, Roll, Yaw angles in time {T} (it means each row) as input and 
       Pitch, Roll, Yaw angles in time {T + shift} as corresponding Label. 

                                 Parameters:
                                 data_names_list - names of .csv files, that will be loaded and concatenated.
                                 shift  - row shift for each label set of corresponding .csv file. (represents time shift
                                          for each loaded frame)
                                 
                                 Returns:
                                 train_X - DataFrame, that will be used as training input for the model.
                                 train_Y - DataFrame, that will be used as label set for input training set.
        """
    def get_concatTrainAndTest(self, data_names_list, shift):
        train_X = pd.DataFrame()
        train_Y = pd.DataFrame()

        for i in range(len(data_names_list)):
            df = pd.DataFrame(self.loadAnglesAsCSV(data_names_list[i]))
            data_Y = df.iloc[shift:]
            data_X = df.iloc[0:(df.shape[0] - shift)]
            train_X = pd.concat([train_X, data_X])
            train_Y = pd.concat([train_Y, data_Y])

        return train_X, train_Y


#Usage example
#d = data_preprocessor()
#dl = data_loader
#dat = dl.load_data(dl, os.getcwd()+'\\..\\data\\rare-data\\video_2.csv')
#dat = d.mediapipe_to_openpose(dat)
#dat = d.openpose_keypoints_to_angles(dat)
#d.saveAnglesAsCSV(dat, 'video_2.csv')
#print(d.loadAnglesAsCSV('video_3.csv'))
#d.get_concatTrainAndTest(['video_1.csv','video_2.csv','video_3.csv'], 10)
