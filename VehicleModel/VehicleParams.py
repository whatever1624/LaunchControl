"""
Centralised nominal vehicle parameters
"""
import copy
import numpy as np

# Unit conversion constants
rad_to_deg = 180 / np.pi
deg_to_rad = np.pi / 180
rads_to_rpm = 30 / np.pi
rpm_to_rads = np.pi / 30
kph_to_ms = 1 / 3.6
ms_to_kph = 3.6
g_to_ms2 = 9.81
ms2_to_g = 1 / 9.81
ratio_to_percent = 100
percent_to_ratio = 0.01

# Vehicle parameters
params = {
    # Model parameters
    'mCar': 530 + 30,               # Total car mass, including fuel, possibly including equivalent rotational inertias of front wheels (m)
    'IEngine': 0.120,               # Engine rotational inertia, including all inertias calculated at engine angular speed (kg m^2)
    'IGearbox': 0.018,              # Gearbox rotational inertia, including all inertias calculated at gearbox input angular speed (kg m^2)
    'rGears': np.array([2.36, 1.94, 1.56, 1.29, 1.10, 0.92]), # Gear ratios (1)
    'rGearFinal': 3.10,             # Final drive ratio (1)
    'IWheel': 0.9 * 2,              # Driven axle rotational inertia, including all inertias calculated at wheel angular speed (kg m^2)
    'lWheel': 0.285,                # Wheel radius, assumed to be both the loaded and rolling radii (m)
    'CTyreB': 9.700752028700489,    # Pacejka magic formula parameter (-)
    'CTyreC': 2.24569266231334,     # Pacejka magic formula parameter (-)
    'CTyreD': 6486.9999761951285,   # Pacejka magic formula parameter (-)
    'CTyreE': 0.9999999999995062,   # Pacejka magic formula parameter (-)

    # Additional processing and nonlinearities
    'wClutchLocked': 10,                # Clutch slip angular speed under which to consider the clutch locked (rad/s)
    'wEngineLimit': 6500 * rpm_to_rads, # Limiter engine angular speed (rad/s)
    'MClutchMax': 400,                  # Max clutch torque (Nm)
    # Look-up table of maximum engine torque: [wEngine, MEngineMax]
    'EngineLUT': np.array([[0.0, 70.0], [52.36, 70.0], [104.72, 70.0], [157.08, 70.0], [209.44, 113.0], [261.799, 118.0], [314.159, 128.0], [366.519, 129.0], [418.879, 114.0], [471.239, 112.0], [523.599, 110.0], [575.959, 103.0], [628.319, 100.0], [680.678, 91.0], [733.038, 79.0], [785.398, 49.0], [837.758, 20.0], [890.118, 0.0]]),
}


def modify_params(rMultMEngine, rMultFTyre, params):
    """Copy and modify vehicle parameters"""
    params = copy.deepcopy(params)
    params['EngineLUT'] *= rMultMEngine
    params['CTyreD'] *= rMultFTyre
    return params
