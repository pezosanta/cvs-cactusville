import numpy as np

from detectorHOGSVM.definitions import ImgObject

camera_matrix = np.array([[6.0604821777343750e+02, 0., 3.1442626953125000e+02],
                          [0., 6.0498577880859375e+02, 2.4605038452148438e+02],
                          [0., 0., 1.]])
inverse_cm = np.linalg.inv(camera_matrix)


def get3DPosition(depth, obj: ImgObject):
    cdepth = depth[obj.v, obj.u] / 1000  # mm to meters

    pos_vect = np.array([obj.u, obj.v, 1]) * cdepth
    pos_vect = np.expand_dims(pos_vect, axis=-1)
    pos_vect = np.matmul(inverse_cm, pos_vect).squeeze()

    obj.x, obj.y, obj.z = pos_vect
    obj.y *= -1  # y axis is top-to-bottom by default, invert it

    return obj
