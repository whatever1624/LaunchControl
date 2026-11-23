"""
Optimise full-state feedback gains and other regime switching parameters, by pole placement
The state space models for each regime are linearised around their target states (wEngine and rSlip)
TODO: the optimisation bit - then turn this into a function
"""
import scipy
import numpy as np
from VehicleModel.DynamicModel import res_control_ref_1, res_control_ref_2, res_control_ref_3, res_control_ref_4
from VehicleModel.StateSpaceP import calc_state_space_1, calc_state_space_2, calc_state_space_3, calc_state_space_4
from OptimalLaunch.OptGearSelection import opt_gear_selection, select_gear
from OptimalLaunch.OptLaunch import calc_wEngineTarget_1, calc_wEngineTarget_3, calc_rSlipTarget_124

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

# Gain scheduling settings
vCarCutoff = 90 * kph_to_ms # Car speed cutoff (m/s)
vCarInterval = 1            # Car speed intervals between each gain set (m/s)

# Arbitrary pole choices - assigned to the regimes with those states (not guaranteed the pole corresponds to the state)
pole_vCar = -20              # Pole location for the vCar state
pole_wEngine = -30          # Pole location for the wEngine state
pole_rSlip = -60            # Pole location for the rSlip state
# TODO: zeta damping ratio? from 0 to 2
#       0-1 zeta use average pole magnitude, 1-2 zeta linear interp from average pole magnitude to actual magnitude
#       use average pole location as magnitude to make complex conjugate poles (keep vCar pole real if regime 1)

# State target offsets - rSlipPeakOffset required for controllability reasons
wEnginePeakOffset = 0       # Offset of engine speed target from peak MEngine (rad/s)
rSlipPeakOffset = -1e-6     # Offset of slip ratio target from peak FTyre (1)

# Regime switching parameters
wEngineMarginPower = 10     # Margin to engine limiter (rad/s)
wEngineMarginSpeed = 10     # Margin above peak torque engine speed to start blending from regime 3 to regime 2 (rad/s)
wEngineBlendUpper = 20      # Blend width between regime 1 and regime 2/3 (rad/s)
wEngineBlendLower = 20      # Blend width between regime 2 and regime 3 (rad/s)


from VehicleModel.VehicleParams import params


# Calculate MEngine and FTyre peaks
wEnginePeak = calc_wEngineTarget_3(params)
rSlipPeak = calc_rSlipTarget_124(params)

# Calculate wEngine and rSlip targets
wEngineTarget1 = calc_wEngineTarget_1(wEngineMarginPower, params)
wEngineTarget3 = min(wEngineTarget1, wEnginePeak + wEnginePeakOffset)
rSlipTarget124 = rSlipPeak + rSlipPeakOffset

# Optimise gear selection
gearLUT = opt_gear_selection(wEngineMarginPower, params)

print(f"wEngineLimit:                   {params['wEngineLimit']:.3f} rad/s")
print(f"wEngine for max MEngine:        {wEnginePeak:.3f} rad/s")
print(f"rSlip for max FTyre:            {rSlipPeak:.3f}")
print()
print(f"wEngine target (Regime 1):      {wEngineTarget1:.3f} rad/s")
print(f"wEngine target (Regime 3):      {wEngineTarget3:.3f} rad/s")
print(f"rSlip target (Regimes 1, 2, 4): {rSlipTarget124:.3f}\n")

# Create vCar intervals
NvCarIntervals = int(np.ceil(vCarCutoff / vCarInterval)) + 1
vCarIntervals = np.arange(0, vCarInterval * NvCarIntervals, vCarInterval)

# Create array of NGear corresponding to vCarIntervals
NGearIntervals = np.zeros(NvCarIntervals, dtype=int)
for i, vCar in enumerate(vCarIntervals):
    NGearIntervals[i] = select_gear(vCar, gearLUT)

# Create control LUTs
controlLUT1 = np.zeros((NvCarIntervals, 3 + 2 + 6))     # Clutch slip, overpowered
controlLUT2 = np.zeros((NvCarIntervals, 2 + 2 + 2))     # Clutch slip, underpowered transient
controlLUT3 = np.zeros((1, 1 + 2 + 1))                  # Clutch slip, underpowered steady-state
controlLUT4 = np.zeros((NvCarIntervals, 2 + 2 + 2))     # Locked clutch

# Regimes 1, 2, 4 - Control LUTs have rows: *xRef, *uRef, *KGains
print("Placing poles for operating regimes 1, 2, 4")
for i, vCar in enumerate(vCarIntervals):
    print(f"\tvCar: {vCar:.1f} m/s")
    # Calculate poles for each regime
    poles1 = [pole_vCar, pole_wEngine, pole_rSlip]
    poles2 = [pole_vCar, pole_rSlip]
    poles4 = [pole_vCar, pole_rSlip]
    # Calculate reference state vectors and reference gear for each regime
    xRef1 = np.array([vCar, wEngineTarget1, rSlipTarget124], dtype=float)
    xReducedRef2 = np.array([vCar, rSlipTarget124], dtype=float)
    xReducedRef4 = np.array([vCar, rSlipTarget124], dtype=float)
    NGearRef = NGearIntervals[i]
    # Calculate reference control inputs for each regime
    # Solve for 0 dx for non-vCar states, then repackage as uRef = [rThrottleLinearRef, rClutchLinearRef]
    uRef1 = scipy.optimize.root(res_control_ref_1, [1, 0], args=(xRef1, NGearRef, params)).x
    uReducedRef2 = scipy.optimize.root(res_control_ref_2, [0], args=(xReducedRef2, NGearRef, params)).x
    uReducedRef4 = scipy.optimize.root(res_control_ref_4, [1], args=(xReducedRef4, NGearRef, params)).x
    uRef2 = np.array([1., uReducedRef2[0]])
    uRef4 = np.array([uReducedRef4[0], 0.])
    # Calculate linearised state space matrices for each regime
    A1, B1 = calc_state_space_1(xRef1, uRef1, NGearRef, params,
                                BMEnginePeak=(wEngineTarget1 == wEnginePeak),
                                BFTyrePeak=(rSlipTarget124 == rSlipPeak))
    A2, B2 = calc_state_space_2(xReducedRef2, uReducedRef2, NGearRef, params,
                                BFTyrePeak=(rSlipTarget124 == rSlipPeak))
    A4, B4 = calc_state_space_4(xReducedRef4, uReducedRef4, NGearRef, params,
                                BFTyrePeak=(rSlipTarget124 == rSlipPeak))
    # Place poles for each regime
    KGains1 = scipy.signal.place_poles(A1, B1, poles1).gain_matrix
    KGains2 = scipy.signal.place_poles(A2, B2, poles2).gain_matrix
    KGains4 = scipy.signal.place_poles(A4, B4, poles4).gain_matrix
    # Update control LUTs for each regime
    controlLUT1[i] = np.hstack([xRef1, uRef1, KGains1.flatten()])
    controlLUT2[i] = np.hstack([xReducedRef2, uRef2, KGains2.flatten()])
    controlLUT4[i] = np.hstack([xReducedRef4, uRef4, KGains4.flatten()])

# Regime 3 - Control LUTs have rows: *xRef, *uRef, *KGains
print("Placing poles for operating regime 3")
# Calculate pole
poles3 = [pole_wEngine]
# Calculate reference state vector
wEngine = wEngineTarget3
xReducedRef3 = np.array([wEngineTarget3], dtype=float)
NGearRef = 0
# Calculate reference control input
# Solve for 0 dx for non-vCar states, then repackage as uRef = [rThrottleLinearRef, rClutchLinearRef]
uReducedRef3 = scipy.optimize.root(res_control_ref_3, [0], args=(xReducedRef3, NGearRef, params)).x
uRef3 = np.array([1., uReducedRef3[0]])
# Calculate linearised state space matrices
A3, B3 = calc_state_space_3(xReducedRef3, uReducedRef3, NGearRef, params, BMEnginePeak=(wEngineTarget3 == wEnginePeak))
# Place pole
KGains3 = scipy.signal.place_poles(A3, B3, poles3).gain_matrix
# Update control LUT
controlLUT3[0] = np.hstack([xReducedRef3, uRef3, KGains3.flatten()])


# Save control LUTs
controlLUT1Filename = "ControlLUT1.csv"
controlLUT2Filename = "ControlLUT2.csv"
controlLUT3Filename = "ControlLUT3.csv"
controlLUT4Filename = "ControlLUT4.csv"
header1 = "Columns: vCarRef, wEngineTarget, rSlipTarget, rThrottleLinearRef, rClutchLinearRef, KGain1, KGain2, KGain3, KGain4, KGain5, KGain6"
np.savetxt(controlLUT1Filename, controlLUT1, header=header1)
header2 = "Columns: vCarRef, rSlipTarget, rThrottleLinearRef, rClutchLinearRef, KGain1, KGain2, KGain3"
np.savetxt(controlLUT2Filename, controlLUT2, header=header2)
header3 = "Columns: wEngineTarget, rSlipTarget, rThrottleLinearRef, rClutchLinearRef, KGain1"
np.savetxt(controlLUT3Filename, controlLUT3, header=header3)
header4 = "Columns: vCarRef, rSlipTarget, rThrottleLinearRef, rClutchLinearRef, KGain1, KGain2, KGain3"
np.savetxt(controlLUT4Filename, controlLUT4, header=header4)
