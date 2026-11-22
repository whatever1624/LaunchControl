"""
Optimise gear selection
Assumes (almost) locked clutch and rSlip for peak FTyre, optimises gear selection for peak wheel torque
Constrained by wGearbox <= wEngineLimit - wEngineMarginPower - wClutchLocked
"""
import scipy
import numpy as np
from VehicleModel.EngineModel import calc_MEngineMax
from OptimalLaunch.CalcStateTargets import calc_wEngineTarget_1, calc_rSlipTarget_124


def opt_gears(wEngineMarginPower, params):
    """Optimise gear selection to produce a LUT with rows [vCarMin, NGear]"""
    NGears = len(params['rGears'])
    gearLUT = np.zeros((NGears, 2))

    # Zero vCar is always 1st gear
    gearLUT[0] = [-np.inf, 1]

    # Calculate engine speed limit and max gearbox limit (considering wEngineMarginPower)
    wEngineLimit = calc_wEngineTarget_1(wEngineMarginPower, params)
    wGearboxLimit = wEngineLimit - params['wClutchLocked']

    # Calculate rSlip of the tyre peak
    rSlipPeak = calc_rSlipTarget_124(params)

    def delta_MWheel_upshift(wGearbox, NGearCurrent):
        """Change in wheel torque from upshifting, given the gearbox speed and current gear"""
        # Calculate current and next gear ratios
        rGearCurrent = params['rGears'][NGearCurrent - 1] * params['rGearFinal']
        rGearNext = params['rGears'][NGearCurrent] * params['rGearFinal']

        # Calculate the engine speed at this car speed, for the next gear
        wGearboxNext = wGearbox * (rGearNext / rGearCurrent)
        wEngineNext = wGearboxNext + params['wClutchLocked']

        # Calculate the wheel torques from both gear choices
        MWheelCurrent = rGearCurrent * calc_MEngineMax(wGearbox, params['EngineLUT'])
        MWheelNext = rGearNext * calc_MEngineMax(wEngineNext, params['EngineLUT'])

        return MWheelNext - MWheelCurrent

    # For each gear from 1st up to the 2nd last gear
    for NGear in range(1, NGears):
        # Calculate current gear ratio
        rGearCurrent = params['rGears'][NGear - 1] * params['rGearFinal']

        if delta_MWheel_upshift(wGearboxLimit, NGear) <= 0:
            # Upshifting at gearbox speed limit decreases wheel torque - upshifting here is optimal
            wGearboxUpshift = wGearboxLimit
        else:
            # Upshifting at gearbox speed limit increases wheel torque - optimise earlier upshift point
            wGearboxUpshift = scipy.optimize.root(delta_MWheel_upshift, wGearboxLimit, args=(NGear)).x[0]

        # Calculate the car speed corresponding to the gearbox speed limit hence engine speed limit
        vCar = (wGearboxLimit * params['lWheel']) / (rGearCurrent * (rSlipPeak + 1))

        # Update gear LUT
        gearLUT[NGear] = [vCar, NGear + 1]

    return gearLUT


def select_gear(vCar, gearLUT):
    """Gear selection logic based on gear LUT"""
    # Find index of row of the highest car speed that vCar is greater than/equal to
    index = np.where(vCar >= gearLUT[:, 0])[0][-1]

    # Return gear given by the index
    return int(gearLUT[index, 1])


if __name__ == '__main__':
    from VehicleModel.VehicleParams import params

    # Regime switching parameters
    wEngineMarginPower = 10  # Margin to engine limiter (rad/s)

    # Optimise gears
    print("Optimising gears")
    gearLUT = opt_gears(wEngineMarginPower, params)
    print(f"Gear LUT:\n{gearLUT}")


