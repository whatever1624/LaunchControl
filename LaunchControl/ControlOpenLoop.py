"""
Calculate open-loop control input from a predefined trajectory
Uses the vCar state as the interpolation variable
"""
import numpy as np


def calc_control_open_loop(x, trajLUT):
    """Calculate open loop control input and gear selection at the car speed state, from the trajectory LUT"""
    # Unpack current state
    vCar, wEngine, rSlip = x

    # Unpack trajectory
    ts = trajLUT[:, 0]
    xs = trajLUT[:, 1:4]
    us = trajLUT[:, 4:6]
    NGears = np.array(trajLUT[:, -1])
    vCars = xs[:, 0]

    # Linear interpolate the control inputs at the current car speed
    Nu = len(us[0])
    u = np.zeros(Nu)
    for iu in range(Nu):
        u[iu] = np.interp(vCar, vCars, us[:, iu])

    # Calculate gear selection
    NGear = int(np.ceil(np.interp(vCar, vCars, NGears)))

    return u, NGear
