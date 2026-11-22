"""
Calculate state space matrices for each of the 4 operating regimes - P-control only
"""
import numpy as np
from VehicleModel.EngineModel import calc_MEngineMax, calc_dMEngineMax_dwEngine
from VehicleModel.TyreModel import calc_dFTyre_drSlip
from VehicleModel.DynamicModel import calc_dx_1, calc_dx_2, calc_dx_3, calc_dx_4


# Settings
vCarMin = 1     # Minimum vCar to avoid numerical instability (m/s)


def calc_state_space_1(x, u, NGear, params, BMEnginePeak=False, BFTyrePeak=False):
    """Calculate state space A and B matrices for regime 1, linearised about the point with state x and input u"""
    dvCar, dwEngine, drSlip = calc_dx_1(x, u, NGear, params)
    vCar, wEngine, rSlip = x
    rThrottleLinear, rClutchLinear = u

    # Avoid numerical instability at low speed
    vCar = max(vCarMin, vCar)

    # Intermediate variables
    rGear = params['rGears'][NGear - 1] * params['rGearFinal']
    IEquiv = (params['IGearbox'] * rGear ** 2 + params['IWheel']) / params['lWheel']
    dMEngineMax_dwEngine = 0 if BMEnginePeak else calc_dMEngineMax_dwEngine(wEngine, params['EngineLUT'])
    MEngineMax = calc_MEngineMax(wEngine, params['EngineLUT'])
    tyreParams = (params['CTyreB'], params['CTyreC'], params['CTyreD'], params['CTyreE'])
    dFTyre_drSlip = 0 if BFTyrePeak else calc_dFTyre_drSlip(rSlip, *tyreParams)

    L = np.array([[params['mCar'],          0,                      0],
                  [0,                       params['IEngine'],      0],
                  [IEquiv * (1 + rSlip),    0,                      IEquiv * vCar]])

    M = np.array([[0,                       0,                      -dFTyre_drSlip],
                  [0,     -rThrottleLinear * dMEngineMax_dwEngine,  0],
                  [IEquiv * drSlip,         0,                      IEquiv * dvCar + params['lWheel'] * dFTyre_drSlip]])

    N = np.array([[0,                       0],
                  [-MEngineMax,             -params['MClutchMax']],
                  [0,                       rGear * params['MClutchMax']]])

    # Calculate A and B matrices
    L_inv = np.linalg.inv(L)
    A = np.matmul(-L_inv, M)
    B = np.matmul(-L_inv, N)

    return A, B


def calc_state_space_2(xReduced, uReduced, NGear, params, BFTyrePeak=False):
    """Calculate state space A and B matrices for regime 1, linearised about the point with state x and input u"""
    dvCar, drSlip = calc_dx_2(xReduced, uReduced, NGear, params)
    vCar, rSlip = xReduced
    rClutchLinear = uReduced[0]

    # Avoid numerical instability at low speed
    vCar = max(vCarMin, vCar)

    # Intermediate variables
    rGear = params['rGears'][NGear - 1] * params['rGearFinal']
    IEquiv = (params['IGearbox'] * rGear ** 2 + params['IWheel']) / params['lWheel']
    tyreParams = (params['CTyreB'], params['CTyreC'], params['CTyreD'], params['CTyreE'])
    dFTyre_drSlip = 0 if BFTyrePeak else calc_dFTyre_drSlip(rSlip, *tyreParams)

    L = np.array([[params['mCar'],          0],
                  [IEquiv * (1 + rSlip),    IEquiv * vCar]])

    M = np.array([[0,                       -dFTyre_drSlip],
                  [IEquiv * drSlip,         IEquiv * dvCar + params['lWheel'] * dFTyre_drSlip]])

    N = np.array([[0],
                  [rGear * params['MClutchMax']]])

    # Calculate A and B matrices
    L_inv = np.linalg.inv(L)
    A = np.matmul(-L_inv, M)
    B = np.matmul(-L_inv, N)

    return A, B


def calc_state_space_3(xReduced, uReduced, NGear, params, BMEnginePeak=False):
    """Calculate state space A and B matrices for regime 1, linearised about the point with state x and input u"""
    wEngine = xReduced[0]
    rClutchLinear = uReduced[0]
    rThrottleLinear = 1.

    # Intermediate variables
    dMEngineMax_dwEngine = 0 if BMEnginePeak else calc_dMEngineMax_dwEngine(wEngine, params['EngineLUT'])

    L = params['IEngine']
    M = -rThrottleLinear * dMEngineMax_dwEngine
    N = -params['MClutchMax']

    # Calculate A and B matrices
    L_inv = 1 / L
    A = np.array([[-L_inv * M]])
    B = np.array([[-L_inv * N]])

    return A, B


def calc_state_space_4(xReduced, uReduced, NGear, params, BFTyrePeak=False):
    """Calculate state space A and B matrices for regime 1, linearised about the point with state x and input u"""
    dvCar, drSlip = calc_dx_4(xReduced, uReduced, NGear, params)
    vCar, rSlip = xReduced
    rThrottleLinear = uReduced[0]

    # Avoid numerical instability at low speed
    vCar = max(vCarMin, vCar)

    # Intermediate variables
    rGear = params['rGears'][NGear - 1] * params['rGearFinal']
    wEngine = rGear * (vCar * rSlip + vCar) / params['lWheel']
    IEquiv = ((params['IEngine'] + params['IGearbox']) * rGear ** 2 + params['IWheel']) / params['lWheel']
    dMEngineMax_dwEngine = calc_dMEngineMax_dwEngine(wEngine, params['EngineLUT'])
    dMEngineMax_dvCar = (rGear / params['lWheel']) * dMEngineMax_dwEngine * (rSlip + 1) * rThrottleLinear
    dMEngineMax_drSlip = (rGear / params['lWheel']) * dMEngineMax_dwEngine * vCar * rThrottleLinear
    MEngineMax = calc_MEngineMax(wEngine, params['EngineLUT'])
    tyreParams = (params['CTyreB'], params['CTyreC'], params['CTyreD'], params['CTyreE'])
    dFTyre_drSlip = 0 if BFTyrePeak else calc_dFTyre_drSlip(rSlip, *tyreParams)

    L = np.array([[params['mCar'],                                      0],
                  [IEquiv * (1 + rSlip),                                IEquiv * vCar]])

    M = np.array([[0,                                                   -dFTyre_drSlip],
                  [IEquiv * drSlip - rGear * dMEngineMax_dvCar,
                                        IEquiv * dvCar + params['lWheel'] * dFTyre_drSlip - rGear * dMEngineMax_drSlip]])

    N = np.array([[0],
                  [-rGear * MEngineMax]])

    # Calculate A and B matrices
    L_inv = np.linalg.inv(L)
    A = np.matmul(-L_inv, M)
    B = np.matmul(-L_inv, N)

    return A, B


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from VehicleParams import params

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

    # Reference states and inputs
    vCarRef = 50 * kph_to_ms            # Reference car speed (m/s)
    wEngineRef = 3750 * rpm_to_rads     # Reference engine angular speed (rad/s)
    rSlipRef = 0.1                      # Reference slip ratio (1)
    rThrottleLinearRef = 0.5            # Reference throttle linear input (1)
    rClutchLinearRef = 0.5              # Reference clutch linear input (1)
    NGear = 1                           # Gear number (-)

    # Perturb amounts
    NPerturb = 101                      # Number of perturb points in each direction
    vCarPerturb = 5                     # Reference car speed (m/s)
    wEnginePerturb = 500 * rpm_to_rads  # Reference engine angular speed (rad/s)
    rSlipPerturb = 0.01                 # Reference slip ratio (1)
    rThrottleLinearPerturb = 0.1        # Reference throttle linear input (1)
    rClutchLinearPerturb = 0.1          # Reference clutch linear input (1)

    # Reference state/inputs for each regime
    xuRefRegimes = [[vCarRef, wEngineRef, rSlipRef, rThrottleLinearRef, rClutchLinearRef],
                    [vCarRef, rSlipRef, rClutchLinearRef],
                    [wEngineRef, rClutchLinearRef],
                    [vCarRef, rSlipRef, rThrottleLinearRef]]

    # Max state/input perturbations for each regime
    rScalePerturbsRegimes = [[vCarPerturb, wEnginePerturb, rSlipPerturb, rThrottleLinearPerturb, rClutchLinearPerturb],
                             [vCarPerturb, rSlipPerturb, rClutchLinearPerturb],
                             [wEnginePerturb, rClutchLinearPerturb],
                             [vCarPerturb, rSlipPerturb, rThrottleLinearPerturb]]

    # Reference nonlinear dx functions for each regime
    dxRefFuncRegimes = [calc_dx_1, calc_dx_2, calc_dx_3, calc_dx_4]

    # Linearised state space matrix functions for each regime
    stateSpaceFuncRegimes = [calc_state_space_1, calc_state_space_2, calc_state_space_3, calc_state_space_4]

    # State derivative names, state names, and input names for each regime
    dxNamesRegimes = [['dvCar', 'dwEngine', 'drSlip'],
                      ['dvCar', 'drSlip'],
                      ['dwEngine'],
                      ['dvCar', 'drSlip']]
    xNamesRegimes = [['vCar', 'wEngine', 'rSlip'],
                     ['vCar', 'rSlip'],
                     ['wEngine'],
                     ['vCar', 'rSlip']]
    uNamesRegimes = [['rThrottleLinear', 'rClutchLinear'],
                     ['rClutchLinear'],
                     ['rClutchLinear'],
                     ['rThrottleLinear']]

    # For each regime
    rPerturbs = np.linspace(-1, 1, NPerturb * 2 + 1)
    for iRegime in range(len(dxRefFuncRegimes)):
        print(f"\nTesting validity of state space model for operating regime {iRegime + 1}")

        # Get variables for the specific regime
        xuRef = np.array(xuRefRegimes[iRegime], dtype=float)
        rScalePerturbs = rScalePerturbsRegimes[iRegime]
        dxRefFunc = dxRefFuncRegimes[iRegime]
        stateSpaceFunc = stateSpaceFuncRegimes[iRegime]
        dxNames = dxNamesRegimes[iRegime]
        xNames = xNamesRegimes[iRegime]
        uNames = uNamesRegimes[iRegime]

        # Split combined reference state/input vector into separate reference state vector and reference input vector
        xRef = xuRef[:len(xNames)]
        uRef = xuRef[len(xNames):]

        # Combine state and input names into 1 vector
        xuNames = xNames + uNames

        # Set up plot (each row is a state derivative, each column is a perturbation by a different state/input
        fig, axs = plt.subplots(len(dxNames), len(xuNames), layout='constrained', figsize=(12, 6))
        axs = axs.reshape((len(dxNames), len(xuNames)))

        # For each element in the combined state and input vector
        for ixu in range(len(xuNames)):
            print(f"\tPerturbing {xuNames[ixu]}")

            # Calculate reference state derivatives
            dxRef = dxRefFunc(xRef, uRef, NGear, params)

            # Get linearised state space matrices around the reference states and inputs
            A, B = stateSpaceFunc(xRef, uRef, NGear, params)

            # Calculate delta from the reference combined state and input vector, for each perturbation
            xuDeltas = np.zeros((len(rPerturbs), len(xuNames)))
            xuDeltas[:, ixu] = rPerturbs * rScalePerturbs[ixu]

            # Calculate state derivatives at the perturbed state/input points
            dxModel = np.zeros((len(rPerturbs), len(dxRef)))
            dxStateSpace = np.zeros((len(rPerturbs), len(dxRef)))
            for i, xuDelta in enumerate(xuDeltas):
                xDelta = xuDelta[:len(xNames)]
                uDelta = xuDelta[len(xNames):]
                dxModel[i] = dxRefFunc(xRef + xDelta, uRef + uDelta, NGear, params)
                dxStateSpace[i] = dxRef + np.matmul(A, xDelta) + np.matmul(B, uDelta)

            # Plot the perturbed state derivatives from the model and from the linearised state space matrices
            for i, ax in enumerate(axs[:, ixu]):
                ax.plot(xuRef[ixu] + xuDeltas[:, ixu], dxModel[:, i], label='Model')
                ax.plot(xuRef[ixu] + xuDeltas[:, ixu], dxStateSpace[:, i], label='State Space')
                if ixu == 0:
                    ax.set_ylabel(dxNames[i])
            axs[-1, ixu].set_xlabel(xuNames[ixu])
        axs[0, -1].legend()
        fig.suptitle(f"Operating Regime {iRegime + 1}")
        plt.show()
