import cv2
import logging
import os
import mediapipe as mp
import numpy as np
import pandas as pd



class data_extractor:
    """This class provides some possibilities to extract pose data from prerecorded videos"""

    def __init__(self):
        self.cwd = os.getcwd()



    def extract_data_from_videos(self, from_path, save_to_path):
        """Extract pose data from each video file in folder and save it

                       Parameters:
                       from_path     - path to the file, where the processed video file are located.
                       save_to_path  - path where the extracted data shall be saved

                """
        print(os.listdir(from_path))
        if(os.path.exists(from_path)):
            for file in os.listdir(from_path):
                if(file.endswith('.mp4')):
                    self.extract_data_from_video(self, file, save_to_path, from_path)
        else:
            logging.error("Path ",from_path," does not exist")






    def extract_data_from_video(self, file_name, save_to_path, path):
        """Extract pose data from one single prerecorded video and save it

               Parameters:
               file_name    - file name without path description
               save_to_path - path where the extracted data shall be saved
               path         - path to the processed video file. Per default the path is the current working directory

        """
        cwd = path
        full_file_name = path+'\\'+file_name
        cap = cv2.VideoCapture(full_file_name)

        # Load drawing untility and pose untility
        mpDraw = mp.solutions.drawing_utils
        mpPose = mp.solutions.pose
        pose = mpPose.Pose()

        #DataFrame for data collection
        col = []
        for i in range(0,32):
            col.extend([str(i)+':x', str(i)+':y', str(i)+':z',str(i)+':visibility'])
        data = pd.DataFrame([], columns=col)

        if(cap.isOpened() == False):
            logging.error('Error opening Video Stream or File for following File: ',file_name)

        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret == True:
                # Convert color system to fit into mediapipe (OpenCv support BGR-format, MediaPipe support RGB-format)
                imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(imgRGB)

                # print(results.pose_landmarks)
                if (results.pose_landmarks):
                    landmarks = results.pose_landmarks.landmark
                if results.pose_landmarks:
                    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                cv2.imshow('frame', frame)

                #Reshape Data
                curr_landmark = []
                for i in range(0,32):
                    curr_landmark.extend([landmarks[i].x,landmarks[i].y, landmarks[i].z, landmarks[i].visibility])

                data.loc[len(data.index)] = curr_landmark

                if cv2.waitKey(25) == ord('q'):
                    break

            else:
                break

        cap.release()
        cv2.destroyAllWindows()
        print(data)
        self.save_data(data, save_to_path, file_name.split('.')[0])



    def save_data(data, save_to_path, name_of_saved_file):
        #TODO: Documentation, but this code is simple to understand
        print(save_to_path+'\\'+name_of_saved_file + '.csv')
        data.to_csv(save_to_path+'\\'+name_of_saved_file+'.csv')


d = data_extractor
d.extract_data_from_videos(d, os.getcwd()+'\\..\\data\\prerecorded-videos\\', os.getcwd()+'\\..\\data\\rare-data')
#d.extract_data_from_videos(d, 'D:\\Projects\\Uni\\Studienproject\\Turing\\data\\prerecorded-videos', '')