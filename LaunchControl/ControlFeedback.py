"""
Calculate feedback control input, given the current vehicle state
Uses the gear and control LUTs and regime-switching parameters calculated during control optimisation
"""
import numpy as np
from OptimalLaunch.OptGearSelection import select_gear


def clamp(x, lb, ub):
    """Clamp x to be between the lower and upper bounds (inclusive)"""
    return np.maximum(lb, np.minimum(x, ub))


def calc_control_feedback(x, wEngineMarginPower, wEngineMarginSpeed, wEngineBlendUpper, wEngineBlendLower,
                           controlLUT1, controlLUT2, controlLUT3, controlLUT4, gearLUT, params):
    """Calculate feedback control input and gear selection at the car speed state, from the trajectory LUT"""
    # Unpack current state
    vCar, wEngine, rSlip = x

    # Select gear
    NGear = select_gear(vCar, gearLUT)

    # Calculate gearbox input angular speed
    rGear = params['rGears'][NGear - 1] * params['rGearFinal']
    wGearbox = rGear * vCar * (rSlip + 1) / params['lWheel']

    # Get interpolated control rows from each control LUT
    controlRow1 = np.array([np.interp(vCar, controlLUT1[:, 0], controlLUT1[:, col]) for col in range(len(controlLUT1[0]))])
    controlRow2 = np.array([np.interp(vCar, controlLUT2[:, 0], controlLUT2[:, col]) for col in range(len(controlLUT2[0]))])
    controlRow3 = np.array([np.interp(wEngine, controlLUT3[:, 0], controlLUT3[:, col]) for col in range(len(controlLUT3[0]))])
    controlRow4 = np.array([np.interp(vCar, controlLUT4[:, 0], controlLUT4[:, col]) for col in range(len(controlLUT4[0]))])
    Nu = 2

    # Calculate control inputs for regime 1
    Nu1 = 2
    x1 = x
    Nx1 = len(x1)
    xRef1 = controlRow1[:Nx1]
    u1 = controlRow1[Nx1:Nx1 + Nu]
    KGains1 = controlRow1[Nx1 + Nu:].reshape((Nu1, Nx1))
    u1 += np.matmul(KGains1, xRef1 - x1)

    # Calculate control inputs for regime 2
    NuReduced2 = 1
    xReduced2 = x[[0, 2]]
    NxReduced2 = len(xReduced2)
    xReducedRef2 = controlRow2[:NxReduced2]
    u2 = controlRow2[NxReduced2:NxReduced2 + Nu]
    KGains2 = controlRow2[NxReduced2 + Nu:].reshape((NuReduced2, NxReduced2))
    u2[1] += np.matmul(KGains2, xReducedRef2 - xReduced2)[0]

    # Calculate control inputs for regime 3
    NuReduced3 = 1
    xReduced3 = x[[1]]
    NxReduced3 = len(xReduced3)
    xReducedRef3 = controlRow3[:NxReduced3]
    u3 = controlRow3[NxReduced3:NxReduced3 + Nu]
    KGains3 = controlRow3[NxReduced3 + Nu:].reshape((NuReduced3, NxReduced3))
    u3[1] += np.matmul(KGains3, xReducedRef3 - xReduced3)[0]

    # Calculate control inputs for regime 4
    NuReduced4 = 1
    xReduced4 = x[[0, 2]]
    NxReduced4 = len(xReduced4)
    xReducedRef4 = controlRow4[:NxReduced4]
    u4 = controlRow4[NxReduced4:NxReduced4 + Nu]
    KGains4 = controlRow4[NxReduced4 + Nu:].reshape((NuReduced4, NxReduced4))
    u4[0] += np.matmul(KGains4, xReducedRef4 - xReduced4)[0]

    # Clamp control inputs for each regime
    uClamped1 = clamp(u1, 0, 1)
    uClamped2 = clamp(u2, 0, 1)
    uClamped3 = clamp(u3, 0, 1)
    uClamped4 = clamp(u4, 0, 1)

    # Calculate control ratio
    rControl1_partial = np.interp(wEngine - params['wEngineLimit'] + 2 * wEngineMarginPower, [-wEngineBlendUpper, 0], [0, 1])
    rControl2_partial = 1 if uClamped2[1] > uClamped3[1] else np.interp(wEngine - controlRow3[0] - wEngineMarginSpeed, [0, wEngineBlendLower], [0, 1])
    rControl3_partial = 1 - rControl2_partial
    rControl4 = np.interp(wEngine - wGearbox, [params['wClutchLocked'], 2 * params['wClutchLocked']], [1, 0])
    rControl1 = (1 - rControl4) * rControl1_partial
    rControl2 = (1 - rControl4) * (1 - rControl1) * rControl2_partial
    rControl3 = (1 - rControl4) * (1 - rControl1) * rControl3_partial

    # Blend control inputs using the control ratios
    u = (uClamped1 * rControl1) + (uClamped2 * rControl2) + (uClamped3 * rControl3) + (uClamped4 * rControl4)

    """
    rControls = [rControl1, rControl2, rControl3, rControl4]
    print(' ' * 60, [round(float(r), 3) for r in rControls])
    """

    return u, NGear