import pybullet as p
from qibullet import SimulationManager as sim
from tools.data_preprocessor import data_preprocessor
from tensorflow import keras
from pathlib import Path
import pandas as pd
import numpy as np
import time
from tools.normalization import Normalize

"""This class provides PyBullet Pepper simulation with specified model.

                """


class Simulation():

    def __init__(self, robot='pepper'):
        sim_mngr = sim()
        qi_client = sim_mngr.launchSimulation(True)
        p.setGravity(0, 0, -10)
        self.pepper = sim_mngr.spawnPepper(qi_client, [0, 0, 0], spawn_ground_plane=True)
        self.pepper.goToPosture("StandInit", 0.6)
        time.sleep(0.5)

    def updateSimulation(self):
        p.stepSimulation()

    def closeSimulation(self):
        p.disconnect()

        """This function is used to simulate Pepper reaching movements based on the given trained model in PyBullet.

                                   Parameters:
                                   path_to_model  - path to the saved model.
                                   simulation_duration  - number of update-steps that should be performed.

                            """

    def simulationOnAbsolutePosition(path_to_model, simulation_duration):
        normalizer = Normalize()
        preprocessor = data_preprocessor()
        print(str(Path(__file__).parent.parent) + str(path_to_model))


        model = keras.models.load_model(str(Path(__file__).parent.parent) + str(path_to_model))
        #data = [[1.5587493848416123, -0.03238759872452013, 1.552117994460908, -0.43527844392714, -1.129318556835431,-0.5667067307406319, 0.9211347537371364, 0.43622624884243644]]

        predicted_simulation = normalizer.normalization(pd.DataFrame(data=[
            [1.55, -0.03, 1.55, -0.43, -1.12, -0.56, 0.92, 0.43]],
                                            columns=["LShoulderPitch", "LShoulderRoll", "RShoulderPitch",
                                                     "RShoulderRoll", "LElbowYaw", "LElbowRoll",
                                                     "RElbowYaw", "RElbowRoll"]))

        for iteration in range(simulation_duration):
            # predicted_simulation = pd.concat([predicted_simulation, pd.DataFrame(model.predict(predicted_simulation[-3:]))])
            predicted_simulation.loc[predicted_simulation.shape[0]] = (model.predict(pd.DataFrame(predicted_simulation.iloc[predicted_simulation.shape[0]-1]).transpose())[-1])
            #print("*******************************************************************************")
            #print(predicted_simulation)
            #print("===============================================================================")
            #print(model.predict(pd.DataFrame(predicted_simulation.iloc[predicted_simulation.shape[0]-1]).transpose())[-1])
            #print("*******************************************************************************")
            #print(predicted_simulation.RShoulderPitch, predicted_simulation.RElbowRoll)

        simulation = Simulation()
        predicted_simulation = normalizer.denormalization(predicted_simulation)
        predicted_simulation.to_csv('dat.csv')
        for x in range(3, simulation_duration):
            print(predicted_simulation.loc[x].to_string())
            simulation.pepper.setAngles("LShoulderPitch",predicted_simulation.iloc[x,0].item(), 1)
            simulation.pepper.setAngles("LShoulderRoll", predicted_simulation.iloc[x, 1].item(), 1)
            simulation.pepper.setAngles("RShoulderPitch",predicted_simulation.iloc[x,2].item(), 1)
            simulation.pepper.setAngles("RShoulderRoll", predicted_simulation.iloc[x, 3].item(), 1)
            simulation.pepper.setAngles("LElbowYaw",  predicted_simulation.iloc[x, 4].item(), 1)
            simulation.pepper.setAngles("LElbowRoll", predicted_simulation.iloc[x,5].item(), 1)
            simulation.pepper.setAngles("RElbowYaw",  predicted_simulation.iloc[x, 6].item(), 1)
            simulation.pepper.setAngles("RElbowRoll", predicted_simulation.iloc[x, 7].item(), 1)
            time.sleep(0.50)
        simulation.closeSimulation()


simulation = Simulation
#simulation.simulationOnAbsolutePosition('\\models\\DNN\\CNN_b1_300', 200)
simulation.simulationOnAbsolutePosition('\\models\\DNN\\CNN_b8_300', 200)
#simulation.simulationOnAbsolutePosition('\\models\\DNN\\CNN_b1_300', 200)
#simulation.simulationOnAbsolutePosition('\\models\\LSTM\\BLSTM_b1_300', 200)

