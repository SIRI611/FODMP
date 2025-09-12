import pybullet as p
import time
import pybullet_data
import os

p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setTimeStep(1. / 240.)

# floor = os.path.join(pybullet_data.getDataPath(), "mjcf/inverted_pendulum.xml")
floor = os.path.join(pybullet_data.getDataPath(), "mjcf/inverted_pendulum.xml")
anyid = p.loadMJCF('objects/binA.xml')
# p.loadMJCF(floor)
# planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
# boxId = p.loadURDF("r2d2.urdf",startPos, startOrientation)
# anyid = p.loadMJCF('objects/binA.xml')
# floor = os.path.join(pybullet_data.getDataPath(), "mjcf/walker2d.xml")
# p.loadMJCF(floor)
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos,startOrientation)
for i in range (100000):
    p.stepSimulation()
    time.sleep(1./40.)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
# p.disconnect()
