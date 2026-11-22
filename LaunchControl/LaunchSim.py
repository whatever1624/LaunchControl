"""
Simulate launch, given the vehicle parameters and control input
"""
import numpy as np
from VehicleModel.DynamicModel import integrate_dx

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

# Simulation settings
wCol = 10   # Column width for info (-)


# TODO: implement control logic in here - probably add an argument to take in the function
#       function takes in the state vector x (and maybe vehicle parameter dict?) and returns the new control input u
def simulate_launch(xInitial, vCarCutoff, tStepInput, tStepSim, controlFunc, controlFuncArgs,
                    BReturnTraj=False, BVerbose=False):
    """Simulate launch from initial state xInitial until the vehicle speed exceeds vCarCutoff"""
    def print_info(t, x, u):
        """Print info of the system state at the timestep"""
        printStr = ""
        cols = [f"{round(t, 6)}"] + [f"{xi:.3f}" for xi in x] + [f"{ui:.3f}" for ui in u]
        for col in cols:
            printStr += col + ' ' * (wCol - len(col))
        print(printStr.rstrip())

    # Sanitise input update timestep
    tStepInput = max(tStepInput, tStepSim)

    # Simulation setup
    t = 0
    tInput = tStepInput
    x = np.array(xInitial, dtype=float)
    uInitial = np.array([0., 0.], dtype=float)
    NGear = 1   # TODO: temporary value, probably take this from another control input function thing

    if BVerbose:
        print_info(t, x, uInitial)

    # Launch trajectory variables
    if BReturnTraj:
        ts = [[0]]
        xs = [[x]]
        us = [[uInitial]]
        NGears = [[NGear]]

    # Input cycle loop
    while x[0] < vCarCutoff:
        # Timekeeping
        tIntervalEnd = tInput if tInput - t > tStepSim else tInput + (tStepSim * 1e-9)
        tsInterval = np.arange(t, tIntervalEnd, tStepSim)
        tInput += tStepInput

        # Control input
        u, NGear = controlFunc(x, *controlFuncArgs)

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

        for i in range(1, len(tsInterval)):
            print_info(tsInterval[i], xsInterval[i], u)

    # Calculate launch time
    tLaunch = np.interp(vCarCutoff, xsInterval[:, 0], tsInterval)

    # Collate times, state vectors, input vectors and gear selections for launch trajectory
    if BReturnTraj:
        ts = np.hstack(ts)
        xs = np.vstack(xs)
        us = np.vstack(us)
        us[0] = us[1]
        NGears = np.hstack(NGears)
        trajLUT = np.hstack((ts.reshape((len(ts), 1)), xs, us, NGears.reshape((len(NGears), 1))))
        return tLaunch, trajLUT
    else:
        return tLaunch


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from VehicleModel.VehicleParams import params
    from LaunchControl.ControlOpenLoop import calc_control_open_loop


    # Launch settings
    vCarCutoff = 90 * kph_to_ms             # Car speed cutoff (m/s)
    wEngineMarginPower = 100 * rpm_to_rads  # Margin to engine limiter (rad/s)

    # Simulation initial state and settings
    vCarStart = 0                       # Initial car speed (m/s)
    wEngineStart = params['wEngineLimit'] - wEngineMarginPower   # Initial engine angular speed (rad/s)
    rSlipStart = 0                      # Initial slip ratio (1)

    # Simulation settings
    tStepInput = 1e-3                   # Input update timestep (s)
    tStepSim = 1e-3                     # Simulation integration timestep (s)

    # Open-loop controller settings - Filename of the open-loop trajectory CSV
    trajFilename = r"/Users/willow/Documents/Repos/LaunchControl/OptimalLaunch/OptTraj.csv"

    # Read the trajectory LUT for the open-loop controller
    trajLUT = np.loadtxt(trajFilename)

    # Simulate launch
    xInitial = [vCarStart, wEngineStart, rSlipStart]
    tLaunch, trajLUT = simulate_launch(xInitial, vCarCutoff, tStepInput, tStepSim,
                                    calc_control_open_loop, [trajLUT], True, True)

    # Unpack returned trajectory
    ts = trajLUT[:, 0]
    xs = trajLUT[:, 1:4]
    us = trajLUT[:, 4:6]
    NGears = np.array(trajLUT[:, -1], dtype=int)

    # Print the launch time
    print(f"\nTime to vCarCutoff: {tLaunch:.3f} s")

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
