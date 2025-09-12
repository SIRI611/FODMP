import torch
import numpy as np

def numpy_img_normalize(obs):
    obs = obs.transpose((2, 0, 1))/255  # transform to C X H X W for pytorch support
    obs += -(np.min(obs))
    obs_max = np.max(obs)
    if obs_max != 0:
        obs /= np.max(obs) / 2
    obs += -1
    return obs

def numpy_normalize(obs):
    obs += -(np.min(obs))
    obs_max = np.max(obs)
    if obs_max != 0:
        obs /= np.max(obs) / 2
    obs += -1
    return obs

def numpy_img_to_torch(obs):
    return obs.transpose((2, 0, 1))/255

src=np.array([[-0.5181628074336426, -0.49614404689009106],
[-0.6397596530875982, -0.04162859362116144],
[-0.5710854912277217, -0.39703866494872087],
[-0.5683494724133736, -0.4364248445408527],
[-0.5206087662146915, -0.4776898518368136],
[-0.6034805235865642, 0.0022554545670963183],
[-0.5021769506630013, -0.5494967428565282],
[-0.5895243903072322, 0.01003231757471118],
[-0.5716361846366498, -0.4026056749186443],
[-0.5792278637318913, -0.35894348055200637],
[-0.6073646427007457, -0.02939994227490741],
[-0.4952199507355415, -0.4873725800097284]])

dst=np.array([[312,380],
[264,199],
[292,332],
[292,357],
[311,373],
[282,185],
[317,402],
[287,183],
[293,335],
[289,326],
[276,194],
[320,366]])

from skimage import transform
tform = transform.estimate_transform('affine', src, dst)


def world_coors_to_image_uv(x, y):
    u, v = tform(np.array([x, y]))[0]
    return [int(u), int(v)]

def crop_left_right_to_border(x, patch_size_half, width):
    x_border_left = x - patch_size_half
    if x_border_left < 0:
        x_border_left = 0
        x_border_right = patch_size_half * 2
    else:
        x_border_right = x + patch_size_half
        if x_border_right >= width:
            x_border_right = width
            x_border_left = x_border_right - patch_size_half * 2
    return x_border_left, x_border_right
# y_border top, y_border_down, height