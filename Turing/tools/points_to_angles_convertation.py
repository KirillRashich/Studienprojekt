import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU

'''
     body_mapping {openpose to mediapipe} = {'0':  "Nose",  -> 0
                '1':  "Neck",      -? 11+12

                '2':  "RShoulder", -> 12
                '3':  "RElbow",    -> 14
                '4':  "RWrist",    -> 16

                '5':  "LShoulder", -> 11
                '6':  "LElbow",    -> 13
                '7':  "LWrist",    -> 15

                '8':  "MidHip"      -> 23+24}
    '''
df = pd.read_csv('D:\\Projects\\Uni\\Studienproject\\Turing\\data\\rare-data\\20220810_175314_4_1.csv', index_col=0)

clear_df = pd.DataFrame([])

for i in range(32):
    clear_df[str(i)+':x'] = df[str(i)+':x']
    clear_df[str(i)+':y'] = df[str(i)+':y']
    clear_df[str(i)+':z'] = df[str(i)+':z']

print((clear_df['11:x'] - clear_df['12:x'])/2)
#Mediapipe to OpenPose mapping
clear_df_open_pose = pd.DataFrame()
clear_df_open_pose['Neck:x'] = (clear_df['11:x'] - clear_df['12:x'])/2
clear_df_open_pose['Neck:y'] = (clear_df['11:y'] - clear_df['12:y'])/2
clear_df_open_pose['Neck:z'] = (clear_df['11:z'] - clear_df['12:z'])/2

clear_df_open_pose['RShoulder:x'] = clear_df['12:x']
clear_df_open_pose['RShoulder:y'] = clear_df['12:y']
clear_df_open_pose['RShoulder:z'] = clear_df['12:z']

clear_df_open_pose['RElbow:x'] = clear_df['14:x']
clear_df_open_pose['RElbow:y'] = clear_df['14:y']
clear_df_open_pose['RElbow:z'] = clear_df['14:z']

clear_df_open_pose['RWrist:x'] = clear_df['16:x']
clear_df_open_pose['RWrist:y'] = clear_df['16:y']
clear_df_open_pose['RWrist:z'] = clear_df['16:z']

clear_df_open_pose['LShoulder:x'] = clear_df['11:x']
clear_df_open_pose['LShoulder:y'] = clear_df['11:y']
clear_df_open_pose['LShoulder:z'] = clear_df['11:z']

clear_df_open_pose['LElbow:x'] = clear_df['13:x']
clear_df_open_pose['LElbow:y'] = clear_df['13:y']
clear_df_open_pose['LElbow:z'] = clear_df['13:z']

clear_df_open_pose['LWrist:x'] = clear_df['15:x']
clear_df_open_pose['LWrist:y'] = clear_df['15:y']
clear_df_open_pose['LWrist:z'] = clear_df['15:z']

clear_df_open_pose['MidHip:x'] = (clear_df['23:x'] - clear_df['24:x'])/2
clear_df_open_pose['MidHip:y'] = (clear_df['23:y'] - clear_df['24:y'])/2
clear_df_open_pose['MidHip:z'] = (clear_df['23:z'] - clear_df['24:z'])/2

print(clear_df_open_pose)
print(clear_df_open_pose.iloc[20:,:])

model =Sequential()

model.add(GRU(128, input_shape=(1, 24), return_sequences=True))
model.add(GRU(128, return_sequences=False))

model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

model.fit(clear_df_open_pose.iloc[1,:], clear_df_open_pose.iloc[2,:])


#convertation = KeypointsToAngles.KeypointsToAngles(clear_df.iloc[0])



