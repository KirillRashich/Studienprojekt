import os
import logging
import pandas as pd

"""This class provides functions for loading and listing mediapipe data, that contains joint values in rare format.

                """
class data_loader():

    def __init__(self):
        self.cwd = os.getcwd()




    def load_data(self, path):
        """Load joints in rare-format from the  specified path.

                               Parameters:
                               path - path to the folder with saved .csv data

                               Returns:
                               DataFrame containing joint values in rare format (32 joints, for each joint (X, Y, Z, visibility))

                               """

        if (os.path.exists(path)):
            return pd.read_csv(path, index_col=0)
        else:
            logging.error("Path ", path, " does not exist")






    def list_data(self, path):
        """List all .csv data in one specified folder, this function can be used as helping function.

                                   Parameters:
                                   path - path to the folder with saved .csv data.

                            """

        if (os.path.exists(path)):
            print('List of data stored in ', path)
            for file in os.listdir(path):
                if (file.endswith('.csv')):
                    print(file)
        else:
            logging.error("Path ", path, " does not exist")



#Using example
#dl = data_loader()
#dl.list_data(os.getcwd()+'\\..\\data\\rare-data')
#data = dl.load_data(os.getcwd()+'\\..\\data\\rare-data\\20220810_175314_1.csv')
#print(data)