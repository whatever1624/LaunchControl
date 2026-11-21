"""
Optimise nominal state and input trajectory for the launch
Currently only using max torque RPM and optimum slip ratio for simplicity
TODO: Optimise wEngineMin, rSlipMax and wEngineStart as a 2-stage meta optimisation with the input interval optimisation
TODO: But actually this problem should probably be solved as an optimal control problem - would be a nice introduction
"""
import scipy
import numpy as np
from EngineModel import calc_MEngineMax
from TyreModel import calc_FTyre, calc_dFTyre_drSlip
from DynamicModel import integrate_dx


# Print settings
wCol = 10   # Column width for info (-)


def opt_launch(xInitial, vCarCutoff, tStepInputInitial, tStepInput, tStepSim, BReturnTraj=False, BVerbose=False):
    """Optimise inputs to maximise vCar at the end of each input interval, constrained where wEngine >= wEngine for peak
    torque, and rSlip <= rSlip for peak FTyre"""
    def print_info(t, x, u):
        """Print info of the system state at the timestep"""
        printStr = ""
        cols = [f"{round(t, 6)}"] + [f"{xi:.3f}" for xi in x] + [f"{ui:.3f}" for ui in u]
        for col in cols:
            printStr += col + ' ' * (wCol - len(col))
        print(printStr.rstrip())

    def obj_u_interval(u, wEngineMin, rSlipMax, uNominal, tsInterval, x, NGear, params):
        """Objective function for input interval (maximise final vCar, constrained by min wEngine and max rSlip limits)"""
        rScaleNormalisation = tsInterval[-1] - tsInterval[0]
        rScalewEngineViolation = 1e6 * rScaleNormalisation / wEngineMEnginePeak
        rScalerSlipViolation = 1e6 * rScaleNormalisation / rSlipFTyrePeak
        rScaleuPrimePenalty = 1e-3 * rScaleNormalisation / rSlipFTyrePeak

        # Simulate interval
        xsInterval = integrate_dx(tsInterval, x, u, NGear, params)
        vCarFinal = xsInterval[-1, 0]

        # Calculate multiplier for constraint violations and deviation from nominal control input
        wEngineViolation = max(0, wEngineMin - min(xsInterval[1:, 1]))
        rSlipViolation = max(0, max(xsInterval[1:, 2]) - rSlipMax)
        uDeviation = np.sum(np.abs(u - uNominal))
        penalty = 0
        penalty += rScalewEngineViolation * wEngineViolation
        penalty += rScalerSlipViolation * rSlipViolation
        penalty += rScaleuPrimePenalty * uDeviation

        # Objective function result - negative since vCarFinal should be maximised
        obj = -vCarFinal + penalty
        return obj

    def obj_u_interval_clutch(rClutchLinear, wEngineMin, rSlipMax, rClutchLinearNominal, tsInterval, x, NGear, params):
        """Wrapper for obj_u_interval, assuming full rThrottleLinear"""
        rThrottleLinear = 1.
        rClutchLinear = rClutchLinear[0]
        u = np.array([rThrottleLinear, rClutchLinear])
        uNominal = np.array([rThrottleLinear, rClutchLinearNominal])
        return obj_u_interval(u, wEngineMin, rSlipMax, uNominal, tsInterval, x, NGear, params)

    # Sanitise input update timesteps
    tStepInputInitial = max(tStepInputInitial, tStepSim)
    tStepInput = max(tStepInput, tStepSim)

    # Find wEngine for max MEngine and rSlip for max FTyre
    wEngineMEnginePeak = params['EngineLUT'][np.argmax(params['EngineLUT'][:, 1]), 0]
    tyreParams = (params['CTyreB'], params['CTyreC'], params['CTyreD'], params['CTyreE'])
    rSlipFTyrePeak = scipy.optimize.root(calc_dFTyre_drSlip, 1e-3, args=tyreParams).x[0]
    FTyrePeak = calc_FTyre(rSlipFTyrePeak, *tyreParams)

    if BVerbose:
        print(f"wEngine for max MEngine: {wEngineMEnginePeak:.3f} rad/s")
        print(f"rSlip for max FTyre:     {rSlipFTyrePeak:.3f} rad/s\n")

    # Simulation setup
    wEngineMin = wEngineMEnginePeak
    rSlipMax = rSlipFTyrePeak
    t = 0
    tInput = tStepInputInitial
    x = np.array(xInitial, dtype=float)
    NGear = 1

    # Launch trajectory variables
    if BReturnTraj:
        ts = [[0]]
        xs = [[x]]
        us = [[0, 0]]
        NGears = [[NGear]]

    # Input cycle loop
    while x[0] < vCarCutoff:
        # Timekeeping
        tIntervalEnd = tInput if tInput - t > tStepSim else tInput + (tStepSim * 1e-9)
        tsInterval = np.arange(t, tIntervalEnd, tStepSim)
        tInput += tStepInput

        # Upshift logic - upshift if the engine speed is predicted to exceed the RPM limit before the next input interval
        # TODO: improve this - use gearbox speed instead of engine speed, incorporate torque curve
        if t > 0:
            wEngineDelta = xsInterval[-1, 1] - xsInterval[0, 1]
            if xsInterval[-1, 1] + wEngineDelta >= params['wEngineLimit'] and NGear < len(params['rGears']):
                NGear += 1

        # Check if overpowered or underpowered at the current MEngineMax with peak FTyre
        MEngineMax = calc_MEngineMax(x[1], params['EngineLUT'])
        BOverpowered = MEngineMax > FTyrePeak * params['lWheel'] / params['rGears'][NGear - 1] / params['rGearFinal']

        # Optimise input interval
        if BOverpowered:
            # Optimise both throttle and clutch inputs
            uNominal = np.array([0.5, 0.5])
            res = scipy.optimize.minimize(obj_u_interval, uNominal,
                                          bounds=[(0, 1), (0, 1)], method='Nelder-Mead',
                                          args=(wEngineMin, rSlipMax, uNominal, tsInterval, x, NGear, params))
            u = res.x
        else:
            # Optimise clutch position, assuming full throttle
            rClutchPedalNominal = 0.
            res = scipy.optimize.minimize(obj_u_interval_clutch, rClutchPedalNominal,
                                          bounds=[(0, 1)], method='Nelder-Mead',
                                          args=(wEngineMin, rSlipMax, rClutchPedalNominal, tsInterval, x, NGear, params))
            u = [1., res.x[0]]

        # Solve trajectory for the interval with the optimised input
        xsInterval = integrate_dx(tsInterval, x, u, NGear, params)
        t = tsInterval[-1]
        x = xsInterval[-1]

        # Update launch trajectory
        if BReturnTraj:
            ts.append(tsInterval[1:])
            xs.append(xsInterval[1:])
            us.append(np.full((len(xsInterval) - 1, len(u)), u))
            NGears.append(np.full(len(xsInterval) - 1, NGear))

        if BVerbose:
            print_info(t, x, u)

    # Calculate launch time
    tLaunch = np.interp(vCarCutoff, xsInterval[:, 0], tsInterval)

    # Collate times, state vectors, input vectors and gear selections for launch trajectory
    if BReturnTraj:
        ts = np.hstack(ts)
        xs = np.vstack(xs)
        us = np.vstack(us)
        us[0] = us[1]
        NGears = np.hstack(NGears)
        traj = (ts, xs, us, NGears)
        return tLaunch, traj
    else:
        return tLaunch


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

    # Simulation initial state
    vCarStart = 0                       # Initial car speed (m/s)
    wEngineStart = 6500 * rpm_to_rads   # Initial engine angular speed (rad/s)
    rSlipStart = 0                      # Initial slip ratio (1)
    NGearStart = 1                      # Initial gear number (-)

    # Launch settings
    vCarCutoff = 90 * kph_to_ms         # Car speed cutoff (m/s)

    # Simulation settings
    tStepInputInitial = 0.025           # First input interval (s)
    tStepInput = tStepInputInitial      # Input update timestep (s)
    tStepSim = 1e-3                     # Simulation integration timestep (s)
    saveFileName = "OptTraj.csv"        # Filename for the trajectory CSV

    # Simulate launch and unpack returned trajectory
    xInitial = [vCarStart, wEngineStart, rSlipStart]
    tLaunch, traj = opt_launch(xInitial, vCarCutoff, tStepInputInitial, tStepInput, tStepSim, True, True)
    ts, xs, us, NGears = traj

    # Print the launch time
    print(f"\nTime to vCarCutoff: {tLaunch:.3f} s")

    # Save launch trajectory as a CSV file, with columns: t, *x, *u, NGear
    traj = np.hstack((ts.reshape((len(ts), 1)), xs, us, NGears.reshape((len(NGears), 1))))
    header = "Columns: t, vCar, wEngine, rSlip, rThrottleLinear, rClutchLinear, NGear"
    np.savetxt(saveFileName, traj, header=header)
    print(f"\nOptimised launch trajectory saved to {saveFileName}")

    # Extract launch trajectory data
    vCars, wEngines, rSlips = xs.T
    rThrottleLinears, rClutchLinears = us.T
    aCars = np.hstack((0, np.diff(vCars) / tStepSim))
    rGears = params['rGears'][NGears - 1].flatten() * params['rGearFinal']
    wClutchSlips = wEngines - (vCars * rSlips + vCars) / params['lWheel'] * rGears

    # Plot simulation history
    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(12, 6))
    ind = 0
    axs[ind].plot(ts, vCars * ms_to_kph, label='vCar')
    axs[ind].set_ylabel('km/h')
    ind += 1
    axs[ind].plot(ts, aCars * ms2_to_g, label='aCar')
    axs[ind].set_ylabel('G')
    ind += 1
    axs[ind].plot(ts, wEngines * rads_to_rpm, label='wEngine')
    axs[ind].plot(ts, wClutchSlips * rads_to_rpm, label='wClutchSlips')
    axs[ind].set_ylabel('RPM')
    ind += 1
    axs[ind].plot(ts, rSlips, label='rSlip')
    axs[ind].set_ylabel('Ratio')
    ind += 1
    axs[ind].plot(ts, rThrottleLinears * ratio_to_percent, label='rThrottleLinear')
    axs[ind].plot(ts, rClutchLinears * ratio_to_percent, label='rClutchLinear')
    axs[ind].set_ylabel('%')
    axs[ind].set_ylim([-10, 110])
    ind += 1
    axs[ind].plot(ts, NGears, label='NGear')
    axs[ind].set_ylim([-0.1, max(NGears) + 0.1])
    axs[ind].set_ylabel('1')
    for ax in axs:
        ax.set_xlim(ts[0], ts[-1])
        ax.legend(loc='right')
    axs[-1].set_xlabel('Time (s)')
    plt.show()
