import numpy as np


class PlaneFrame:
    def __init__(self, origin, u_axis, v_axis):
        self.origin = origin
        self.u_axis = u_axis
        self.v_axis = v_axis
        self.normal = np.cross(u_axis, v_axis)

    def local_to_motor(self, u, v):
        return self.origin + u * self.u_axis + v * self.v_axis
