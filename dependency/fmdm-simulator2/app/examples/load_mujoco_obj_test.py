import pybullet as p
import time
import pybullet_data
import os
import os
import gym
import time
import random
import pathlib
import gym_env
import os
import math
import numpy as np
import random


p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setTimeStep(1. / 240.)

def load_single_object(bullet_client, obj_file, scale_factor=[2, 2, 0.2], mass=0.1, position=[-0.2,0,0.01], orientation=[0,0,0], rgba_color=[.15, .15, .15, 1], collision_scale=None):
    # scale_factor = [0.2, 0.2, 0.2]
    print(obj_file)
    obj_multi_body_flags = bullet_client.URDF_ENABLE_SLEEPING
    if collision_scale == None:
        obj_collision_shape = bullet_client.createCollisionShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=obj_file,
            meshScale=scale_factor
        )
    else:
        obj_collision_shape = bullet_client.createCollisionShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=obj_file,  #collision_obj_file,
            meshScale=scale_factor #collision_scale
        )
    if rgba_color == None or rgba_color == [0,0,0,0]:
        obj_visual_shape = bullet_client.createVisualShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=obj_file,
            meshScale=scale_factor,
            rgbaColor=rgba_color
        )
    else:
        obj_visual_shape = bullet_client.createVisualShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=obj_file,
            meshScale=scale_factor,
            rgbaColor=rgba_color
        )
    obj_orientation_quat = bullet_client.getQuaternionFromEuler(
       orientation
    )
    obj_id = bullet_client.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=obj_collision_shape,
        baseVisualShapeIndex=obj_visual_shape,
        flags=obj_multi_body_flags,
        basePosition=position,
        baseOrientation=obj_orientation_quat
    )
    assert obj_id > -1, 'Multi body of obj could not be created'
    return obj_id


# load_single_object(p, os.environ['path-to-assets'] + 'pegs' + '/cylinder.obj')
# floor = os.path.join(pybullet_data.getDataPath(), "mjcf/inverted_pendulum.xml")
# floor = os.path.join(os.environ['path-to-assets'], "assets_v2/objects/hammer.xml")
# print(floor)
# p.loadMJCF(floor)
p.loadURDF(os.path.join(os.environ['path-to-assets'],"assets_v2/objects/hammer.urdf"),[1,1,0.2], globalScaling=1)

# p.loadURDF(os.path.join(os.environ['path-to-assets'],"assets_v2/objects/table.urdf"),[1,1,2], globalScaling=1)

p.loadURDF(os.path.join(os.environ['path-to-assets'],"assets_v2/objects/mug.urdf"),[0,0,0.05], globalScaling=1)

p.loadURDF(os.path.join(os.environ['path-to-assets'],"assets_v2/objects/puck_goal.urdf"),[-0.2,0,0.1], globalScaling=1)

# ball = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"soccerball.urdf"),[1,0,0.5], globalScaling=0.05)
ball = p.loadURDF(os.path.join(os.environ['path-to-assets'],"soccerball/soccerball.urdf"),[1,0,0.5], globalScaling=0.05)
p.loadURDF(os.path.join(os.environ['path-to-assets'],"assets_v2/objects/drawer_closed.urdf"),[0.5,0,0], globalScaling=1)

planeId = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"plane.urdf"))
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
#  <material name="puck_black" rgba=".15 .15 .15 1" shininess="1" reflectance="1" specular=".5"/>

# pick up the cup and place it on the drawer from the top
# pick up the hammer and place it into the drawer
# push the ball into the goal

# camera config
# scripted policy.
# collect data.

# Baselines:
# BC: data (expert data)

# pybullet only supports 32 color bit image texture.

# camera quart, topview, corner, corner2
# [ 1.        ,  0.        ,  0.        ,  0.        ],
# [ 0.30520839, -0.2308559 , -0.55733544,  0.73683825],
# [-0.39503823,  0.26266734, -0.43493704,  0.76536295],

# the corresponding up vector is
# [0, 0, 1]
# [-0.6804138144587442, -0.680413820500869, 0.2721655267757353]
# [0.7457052108497266, -0.45824210588374314, 0.4836712839385827]

# a = (0.0, -1.0, 7.141914102248847e-05, 0.0, 1.0, 0.0, -0.0, 0.0, -0.0, 7.141914102248847e-05, 1.0, 0.0, 0.10000000149011612, 0.4498928487300873, -1.5000321865081787, 1.0)
# b = (0.1983172595500946, -0.25679662823677063, -0.945899486541748, 0.0, 0.38121190667152405, 0.9092913866043091, -0.16693325340747833, 0.0, 0.9029660820960999, -0.3274824023246765, 0.27822208404541016, 0.0, -0.1711459457874298, 0.27772971987724304, -1.2741960287094116, 1.0)
# c = (-0.5027289986610413, -0.10085028409957886, 0.8585410714149475, 0.0, 0.08942031115293503, -0.9939103126525879, -0.06439058482646942, 0.0, 0.8598066568374634, 0.04439999908208847, 0.5086856484413147, 0.0, -0.2743556499481201, -0.11651670932769775, -1.6885356903076172, 1.0)
# a
# array([[ 0.00000000e+00, -1.00000000e+00,  7.14191410e-05, 0.00000000e+00],
#        [ 1.00000000e+00,  0.00000000e+00, -0.00000000e+00, 0.00000000e+00],
#        [-0.00000000e+00,  7.14191410e-05,  1.00000000e+00, 0.00000000e+00],
#        [ 1.00000001e-01,  4.49892849e-01, -1.50003219e+00, 1.00000000e+00]])
# b
# array([[ 0.19831726, -0.25679663, -0.94589949,  0.        ],
#        [ 0.38121191,  0.90929139, -0.16693325,  0.        ],
#        [ 0.90296608, -0.3274824 ,  0.27822208,  0.        ],
#        [-0.17114595,  0.27772972, -1.27419603,  1.        ]])
# c
# array([[-0.502729  , -0.10085028,  0.85854107,  0.        ],
#        [ 0.08942031, -0.99391031, -0.06439058,  0.        ],
#        [ 0.85980666,  0.0444    ,  0.50868565,  0.        ],
#        [-0.27435565, -0.11651671, -1.68853569,  1.        ]])