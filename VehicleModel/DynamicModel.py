"""
Dynamic vehicle model and associated regime-specific dynamic models
"""
import numpy as np
from VehicleModel.EngineModel import calc_MEngineMax
from VehicleModel.TyreModel import calc_FTyre


# Settings
vCarMin = 1     # Minimum vCar to avoid numerical instability (m/s)


def calc_dx(x, u, NGear, params):
    """Calculate time-derivative of the state vector [dvCar, dwEngine, drSlip]"""
    vCar, wEngine, rSlip = x
    rThrottleLinear, rClutchLinear = u

    # Avoid numerical instability at low speed
    vCar = max(vCarMin, vCar)

    # Intermediate variables
    rGear = params['rGears'][NGear - 1] * params['rGearFinal']
    IEquiv = (params['IGearbox'] * rGear ** 2 + params['IWheel']) / params['lWheel']
    tyreParams = (params['CTyreB'], params['CTyreC'], params['CTyreD'], params['CTyreE'])
    FTyre = calc_FTyre(rSlip, *tyreParams)

    # Calculate MEngine (with RPM limiter)
    if wEngine >= params['wEngineLimit']:
        MEngine = 0
    else:
        MEngineMax = calc_MEngineMax(wEngine, params['EngineLUT'])
        MEngine = rThrottleLinear * MEngineMax

    # Calculate MClutch with locking clutch slip interpolation
    wClutchSlips = wEngine - (vCar * rSlip + vCar) / params['lWheel'] * rGear
    MClutch = (1 - rClutchLinear) * params['MClutchMax']
    MClutch = np.interp(wClutchSlips, [-params['wClutchLocked'], params['wClutchLocked']], [-MClutch, MClutch])

    # Solve linear system to get dx = dvCar, dwEngine, drSlip
    A = [[params['mCar'],           0,                      0],
         [0,                        params['IEngine'],      0],
         [IEquiv * (1 + rSlip),     0,                      IEquiv * vCar]]
    B = [FTyre,
         MEngine - MClutch,
         rGear * MClutch - params['lWheel'] * FTyre]
    dx = np.linalg.solve(A, B)

    return dx


def integrate_dx(ts, xInitial, u, NGear, params):
    """Explicit Runge-Kutta 4th order integration of the time-derivative of the state vector, for a constant control input over the time interval"""
    tSteps = ts[1:] - ts[:-1]
    xs = np.full((len(ts), len(xInitial)), xInitial)

    x = xInitial.copy()
    for i, tStepSim in enumerate(tSteps, start=1):
        # Explicit Runge-Kutta 4th order integration of the dynamic model
        k1 = calc_dx(x, u, NGear, params)
        k2 = calc_dx(x + 0.5 * tStepSim * k1, u, NGear, params)
        k3 = calc_dx(x + 0.5 * tStepSim * k2, u, NGear, params)
        k4 = calc_dx(x + tStepSim * k3, u, NGear, params)
        x += tStepSim / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs[i] = x.copy()

    return xs


def calc_dx_1(x, u, NGear, params):
    """Calculate time-derivative of the state vector [dvCar, dwEngine, drSlip], for simplified state space regime 1"""
    vCar, wEngine, rSlip = x
    rThrottleLinear, rClutchLinear = u

    # Avoid numerical instability at low speed
    vCar = max(vCarMin, vCar)

    # Intermediate variables
    rGear = params['rGears'][NGear - 1] * params['rGearFinal']
    IEquiv = (params['IGearbox'] * rGear ** 2 + params['IWheel']) / params['lWheel']
    tyreParams = (params['CTyreB'], params['CTyreC'], params['CTyreD'], params['CTyreE'])
    FTyre = calc_FTyre(rSlip, *tyreParams)

    # Calculate MEngine (no RPM limiter) and MClutch (assumed clutch slipping)
    MEngineMax = calc_MEngineMax(wEngine, params['EngineLUT'])
    MEngine = rThrottleLinear * MEngineMax
    MClutch = (1 - rClutchLinear) * params['MClutchMax']

    # Solve linear system to get dx = dvCar, dwEngine, drSlip
    A = [[params['mCar'],           0,                      0],
         [0,                        params['IEngine'],      0],
         [IEquiv * (1 + rSlip),     0,                      IEquiv * vCar]]
    B = [FTyre,
         MEngine - MClutch,
         rGear * MClutch - params['lWheel'] * FTyre]
    dx = np.linalg.solve(A, B)

    return dx


def calc_dx_2(xReduced, uReduced, NGear, params):
    """Calculate time-derivative of the reduced state vector [dvCar, drSlip], for simplified state space regime 2"""
    vCar, rSlip = xReduced
    rClutchLinear = uReduced[0]

    # Avoid numerical instability at low speed
    vCar = max(vCarMin, vCar)

    # Intermediate variables
    rGear = params['rGears'][NGear - 1] * params['rGearFinal']
    IEquiv = (params['IGearbox'] * rGear ** 2 + params['IWheel']) / params['lWheel']
    tyreParams = (params['CTyreB'], params['CTyreC'], params['CTyreD'], params['CTyreE'])
    FTyre = calc_FTyre(rSlip, *tyreParams)

    # Calculate MClutch (assumed clutch slipping)
    MClutch = (1 - rClutchLinear) * params['MClutchMax']

    # Solve linear system to get dxReduced = dvCar, drSlip
    A = [[params['mCar'],           0],
         [IEquiv * (1 + rSlip),     IEquiv * vCar]]
    B = [FTyre,
         rGear * MClutch - params['lWheel'] * FTyre]
    dxReduced = np.linalg.solve(A, B)

    return dxReduced


def calc_dx_3(xReduced, uReduced, NGear, params):
    """Calculate time-derivative of the reduced state vector [dwEngine], for simplified state space regime 3"""
    wEngine = xReduced[0]
    rClutchLinear = uReduced[0]
    rThrottleLinear = 1.

    # Calculate MEngine (no RPM limiter) and MClutch (assumed clutch slipping)
    MEngineMax = calc_MEngineMax(wEngine, params['EngineLUT'])
    MEngine = rThrottleLinear * MEngineMax
    MClutch = (1 - rClutchLinear) * params['MClutchMax']

    # Solve equation of motion to get dxReduced = dwEngine
    dwEngine = (MEngine - MClutch) / params['IEngine']
    dxReduced = np.array([dwEngine])

    return dxReduced


def calc_dx_4(xReduced, uReduced, NGear, params):
    """Calculate time-derivative of the reduced state vector [dvCar, drSlip], for simplified state space regime 4"""
    vCar, rSlip = xReduced
    rThrottleLinear = uReduced[0]

    # Avoid numerical instability at low speed
    vCar = max(vCarMin, vCar)

    # Intermediate variables
    rGear = params['rGears'][NGear - 1] * params['rGearFinal']
    wEngine = rGear * (vCar * rSlip + vCar) / params['lWheel']
    IEquiv = ((params['IEngine'] + params['IGearbox']) * rGear ** 2 + params['IWheel']) / params['lWheel']
    tyreParams = (params['CTyreB'], params['CTyreC'], params['CTyreD'], params['CTyreE'])
    FTyre = calc_FTyre(rSlip, *tyreParams)

    # Calculate MEngine (no RPM limiter) and MClutch (assumed clutch slipping)
    MEngineMax = calc_MEngineMax(wEngine, params['EngineLUT'])
    MEngine = rThrottleLinear * MEngineMax

    # Solve linear system to get dxReduced = dvCar, drSlip
    A = [[params['mCar'],           0],
         [IEquiv * (1 + rSlip),     IEquiv * vCar]]
    B = [FTyre,
         rGear * MEngine - params['lWheel'] * FTyre]
    dxReduced = np.linalg.solve(A, B)

    return dxReduced


def res_control_ref_1(uCandidate, x, NGear, params):
    """Calculate reference control input for 0 dwEngine and 0 drSlip, for simplified state space regime 1"""
    return calc_dx_1(x, uCandidate, NGear, params)[1:]


def res_control_ref_2(uReducedCandidate, xReduced, NGear, params):
    """Calculate reference control input for 0 dwEngine and 0 drSlip, for simplified state space regime 2"""
    return calc_dx_2(xReduced, uReducedCandidate, NGear, params)[1:]


def res_control_ref_3(uReducedCandidate, xReduced, NGear, params):
    """Calculate reference control input for 0 dwEngine and 0 drSlip, for simplified state space regime 3"""
    return calc_dx_3(xReduced, uReducedCandidate, NGear, params)


def res_control_ref_4(uReducedCandidate, xReduced, NGear, params):
    """Calculate reference control input for 0 dwEngine and 0 drSlip, for simplified state space regime 4"""
    return calc_dx_4(xReduced, uReducedCandidate, NGear, params)[1:]
