import pybullet as p
from qibullet import SimulationManager as sim
from tools.data_preprocessor import data_preprocessor

class Simulation:

    def __init__(self, robot= 'pepper'):
        sim_mngr = sim()
        qi_client = sim_mngr.launchSimulation(True)
        p.setGravity(0,0,-10)
        self.pepper = sim_mngr.spawnPepper(qi_client, [0, 0, 0], spawn_ground_plane=True)

    def updateSimulation(self):
        p.stepSimulation()

    def closeSimulation(self):
        p.disconnect()

    def simulationOnAbsolutePosition(path_to_data):
        simulation = Simulation()
        preprocessor = data_preprocessor()
        angles = preprocessor.loadAnglesAsCSV(path_to_data)
        print(angles)

        for x in range(angles.shape[0]):
            simulation.pepper.setAngles("LShoulderPitch", float(angles['LShoulder_Pitch'][x]), 1)
            simulation.pepper.setAngles("LShoulderRoll",  float(angles['LShoulder_Roll'][x]),  1)
            simulation.pepper.setAngles("RShoulderPitch", float(angles['RShoulder_Pitch'][x]), 1)
            simulation.pepper.setAngles("RShoulderRoll",  float(angles['RShoulder_Roll'][x]),  1)
            simulation.pepper.setAngles("LElbowYaw",  float(angles['LElbow_Yaw'][x]),  1)
            simulation.pepper.setAngles("LElbowRoll", float(angles['LElbow_Roll'][x]), 1)
            simulation.pepper.setAngles("RElbowYaw",  float(angles['RElbow_Yaw'][x]),  1)
            simulation.pepper.setAngles("RElbowRoll", float(angles['RElbow_Roll'][x]), 1)
        simulation.closeSimulation()

simulation = Simulation
simulation.simulationOnAbsolutePosition('video_9_forward.csv')


