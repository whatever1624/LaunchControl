"""
Evaluate performance and robustness of a control scheme

For a given set of perturbations (with a chosen number of points in each direction) of:
    - tStepInput (time between input updates - from 0 to 2x the nominal tStepInput)
    - rMultMEngine (multiplier on engine torque)
    - rMultFTyre (multiplier on the FTyre function by multiplying CTyreD? or should it be scaling on both axes)

Calculates the extra time to vCarCutoff compared to optimal control under the same circumstances

Cost function is the extra time to vCarCutoff, weighted sum of:
    - Average
    - 90% high
    - 95% high

Input:
    - Numpy array of time to vCarCutoff (3D array)
    - Numpy array of tStepInputs corresponding to the above (3D array)
    - Numpy array of rMultMEngine corresponding to the above (3D array)
    - Numpy array of rMultFTyre corresponding to the above (3D array)
    - Cost function weightings

Output:
    - Cost
"""