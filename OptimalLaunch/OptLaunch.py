"""
Optimise nominal state and input trajectory for the launch
Currently only using max torque RPM and optimum slip ratio for simplicity
TODO: Optimise wEngineMin, rSlipMax and wEngineStart as a 2-stage meta optimisation with the input interval optimisation
TODO: But actually this problem should probably be solved as an optimal control problem - would be a nice introduction
"""
import scipy
import numpy as np
from VehicleModel.EngineModel import calc_MEngineMax
from VehicleModel.TyreModel import calc_FTyre
from VehicleModel.DynamicModel import integrate_dx
from OptimalLaunch.CalcStateTargets import calc_wEngineTarget_1, calc_wEngineTarget_3, calc_rSlipTarget_124
from OptimalLaunch.OptGears import opt_gears, select_gear


# Print settings
wCol = 10   # Column width for info (-)


def opt_launch(xInitial, vCarCutoff, tStepInputInitial, tStepInput, tStepSim, wEngineMarginPower, params,
               BReturnTrajLUT=False, BVerbose=False):
    """Optimise inputs to maximise vCar at the end of each input interval, with wEngine and rSlip constraints"""
    def print_info(t, x, u):
        """Print info of the system state at the timestep"""
        printStr = ""
        cols = [f"{round(t, 6)}"] + [f"{xi:.3f}" for xi in x] + [f"{ui:.3f}" for ui in u]
        for col in cols:
            printStr += col + ' ' * (wCol - len(col))
        print(printStr.rstrip())

    def obj_u_interval(uCandidate, uNominal, tsInterval, x, NGear, params):
        """Objective function for input interval (maximise final vCar, constrained by min wEngine and max rSlip limits)"""
        rScaleNormalisation = tsInterval[-1] - tsInterval[0]
        rScalewEngineViolation = 1e6 * rScaleNormalisation / wEngineTarget3
        rScalerSlipViolation = 1e6 * rScaleNormalisation / rSlipTarget124
        rScaleuNominalPenalty = 1e-3 * rScaleNormalisation

        # Simulate interval
        xsInterval = integrate_dx(tsInterval, x, uCandidate, NGear, params)
        vCarFinal = xsInterval[-1, 0]

        # Calculate sum of squared constraint violations and deviations from nominal control inputs
        wEngineUpperViolation = np.sum(np.square(np.maximum(0, xsInterval[1:, 1] - wEngineTarget1)))
        wEngineLowerViolation = np.sum(np.square(np.maximum(0, wEngineTarget3 - xsInterval[1:, 1])))
        rSlipViolation = np.sum(np.square(np.maximum(0, xsInterval[1:, 2] - rSlipTarget124)))
        uDeviation = np.sum(np.square(uCandidate - uNominal))

        # Calculate penalty
        penalty = 0
        penalty += rScalewEngineViolation * wEngineUpperViolation
        penalty += rScalewEngineViolation * wEngineLowerViolation
        penalty += rScalerSlipViolation * rSlipViolation
        penalty += rScaleuNominalPenalty * uDeviation

        # Objective function result - negative since vCarFinal should be maximised
        obj = -vCarFinal + penalty
        return obj

    def obj_u_interval_clutch(rClutchLinearCandidate, rClutchLinearNominal, tsInterval, x, NGear, params):
        """Wrapper for obj_u_interval, assuming full rThrottleLinear"""
        rThrottleLinear = 1.
        rClutchLinearCandidate = rClutchLinearCandidate[0]
        uCandidate = np.array([rThrottleLinear, rClutchLinearCandidate])
        uNominal = np.array([rThrottleLinear, rClutchLinearNominal])
        return obj_u_interval(uCandidate, uNominal, tsInterval, x, NGear, params)

    # Sanitise input update timesteps
    tStepInputInitial = max(tStepInputInitial, tStepSim)
    tStepInput = max(tStepInput, tStepSim)

    # Calculate wEngine and rSlip targets
    wEngineTarget1 = calc_wEngineTarget_1(wEngineMarginPower, params)
    wEngineTarget3 = min(wEngineTarget1, calc_wEngineTarget_3(params))
    rSlipTarget124 = calc_rSlipTarget_124(params)

    # Find peak FTyre
    tyreParams = (params['CTyreB'], params['CTyreC'], params['CTyreD'], params['CTyreE'])
    FTyrePeak = calc_FTyre(rSlipTarget124, *tyreParams)

    # Optimise gear selection
    gearLUT = opt_gears(wEngineMarginPower, params)

    if BVerbose:
        print(f"wEngine for max MEngine:        {calc_wEngineTarget_3(params):.3f} rad/s")
        print(f"rSlip for max FTyre:            {rSlipTarget124:.3f}")
        print()
        print(f"wEngine target (Regime 1):      {wEngineTarget1:.3f} rad/s")
        print(f"wEngine target (Regime 3):      {wEngineTarget3:.3f} rad/s")
        print(f"rSlip target (Regimes 1, 2, 4): {rSlipTarget124:.3f}\n")

    # Simulation setup
    t = 0
    tInput = tStepInputInitial
    x = np.array(xInitial, dtype=float)
    NGear = 1

    # Launch trajectory variables
    if BReturnTrajLUT:
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

        # Apply upshift logic to predicted car speed at the next input interval
        if t > 0:
            dvCarDelta = xsInterval[-1, 0] - xsInterval[0, 0]
            vCarPred = xsInterval[-1, 0] + dvCarDelta
            NGear = select_gear(vCarPred, gearLUT)

        # Check if overpowered or underpowered at the current MEngineMax with peak FTyre
        MEngineMax = calc_MEngineMax(x[1], params['EngineLUT'])
        BOverpowered = MEngineMax > FTyrePeak * params['lWheel'] / params['rGears'][NGear - 1] / params['rGearFinal']

        # Optimise input interval
        if BOverpowered:
            # Optimise both throttle and clutch inputs
            uNominal = np.array([0.5, 0.5])
            res = scipy.optimize.minimize(obj_u_interval, uNominal,
                                          bounds=[(0, 1), (0, 1)], method='Nelder-Mead',
                                          args=(uNominal, tsInterval, x, NGear, params))
            u = res.x
        else:
            # Optimise clutch position, assuming full throttle
            rClutchPedalNominal = 0.
            res = scipy.optimize.minimize(obj_u_interval_clutch, rClutchPedalNominal,
                                          bounds=[(0, 1)], method='Nelder-Mead',
                                          args=(rClutchPedalNominal, tsInterval, x, NGear, params))
            u = [1., res.x[0]]

        # Solve trajectory for the interval with the optimised input
        xsInterval = integrate_dx(tsInterval, x, u, NGear, params)
        t = tsInterval[-1]
        x = xsInterval[-1]

        # Update launch trajectory
        if BReturnTrajLUT:
            ts.append(tsInterval[1:])
            xs.append(xsInterval[1:])
            us.append(np.full((len(xsInterval) - 1, len(u)), u))
            NGears.append(np.full(len(xsInterval) - 1, NGear))

        if BVerbose:
            print_info(t, x, u)

    # Calculate launch time
    tLaunch = np.interp(vCarCutoff, xsInterval[:, 0], tsInterval)

    # Collate times, state vectors, input vectors and gear selections for launch trajectory
    if BReturnTrajLUT:
        ts = np.hstack(ts)
        xs = np.vstack(xs)
        us = np.vstack(us)
        us[0] = us[1]
        NGears = np.hstack(NGears)
        trajLUT = np.hstack((ts.reshape((len(ts), 1)), xs, us, NGears.reshape((len(NGears), 1))))
        return tLaunch, trajLUT
    else:
        return tLaunch


def save_launch_traj(trajLUT, trajFileName):
    """Save launch trajectory to a CSV file"""
    header = "Columns: t, vCar, wEngine, rSlip, rThrottleLinear, rClutchLinear, NGear"
    np.savetxt(trajFileName, trajLUT, header=header)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from VehicleModel.VehicleParams import params

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

    # Launch settings
    vCarCutoff = 90 * kph_to_ms             # Car speed cutoff (m/s)
    wEngineMarginPower = 100 * rpm_to_rads  # Margin to engine limiter (rad/s)

    # Simulation initial state
    vCarStart = 0                           # Initial car speed (m/s)
    wEngineStart = params['wEngineLimit'] - wEngineMarginPower   # Initial engine angular speed (rad/s)
    rSlipStart = 0                          # Initial slip ratio (1)
    NGearStart = 1                          # Initial gear number (-)

    # Simulation settings
    tStepInputInitial = 0.025               # First input interval (s)
    tStepInput = tStepInputInitial          # Input update timestep (s)
    tStepSim = 1e-3                         # Simulation integration timestep (s)
    trajFileName = r"OptTraj.csv"           # Filename to save the trajectory CSV

    # Simulate launch and
    xInitial = [vCarStart, wEngineStart, rSlipStart]
    tLaunch, trajLUT = opt_launch(xInitial, vCarCutoff, tStepInputInitial, tStepInput, tStepSim, wEngineMarginPower,
                                  params, BReturnTrajLUT=True, BVerbose=True)

    # Unpack returned trajectory
    ts = trajLUT[:, 0]
    xs = trajLUT[:, 1:4]
    us = trajLUT[:, 4:6]
    NGears = np.array(trajLUT[:, -1], dtype=int)

    # Print the launch time
    print(f"\nTime to vCarCutoff: {tLaunch:.3f} s")

    # Save launch trajectory as a CSV file, with columns: t, *x, *u, NGear
    save_launch_traj(trajLUT, trajFileName)
    print(f"\nOptimised launch trajectory saved to {trajFileName}")

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
