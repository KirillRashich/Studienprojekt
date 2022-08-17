import pybullet as p
from qibullet import  simulation_manager
import time
import pybullet_data
from qibullet import SimulationManager as sim, Camera

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



