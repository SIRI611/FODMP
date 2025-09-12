import os
import sys
from pyobject2urdf import ObjectUrdfBuilder

# Build single URDFs
object_folder = "./"

builder = ObjectUrdfBuilder(object_folder, urdf_prototype='prototype.urdf')
builder.build_urdf(filename="TrashCan/pobelle.obj", force_overwrite=True, decompose_concave=True, force_decompose=False, center = 'mass')
