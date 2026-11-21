"""
Fit tyre model to telemetry
Tyre model used is the 4-parameter Pacejka magic formula
"""
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from VehicleParams import params
from TyreModel import calc_FTyre, calc_dFTyre_drSlip

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

# Script settings
motecCSVFolder = r"Tatuus FA01 Data/Tatuus FA01 MoTeC CSV"
vCarCutoff = 100 * kph_to_ms    # Car speed cutoff (m/s)
aCarCutoff = 1                  # Longitudinal acceleration cutoff (m/s^2)
ACutoff = 20 * deg_to_rad       # Angle cutoff for acceleration vector, relative to directly forwards (rad)
rAnomaly = 0.1                  # Deviation ratio to mark the data point as an anomaly (1)

# Get all CSV file paths in the folder
motecCSVFiles = [entry.path for entry in os.scandir(motecCSVFolder) if entry.name.lower().endswith('.csv')]

# Parse and filter data points
print("Parsing telemetry and filtering data points")
rowChannelNames = 13
rowDataStart = 17
dataPoints = {'rSlip': np.array([]),
              'FTyre': np.array([])}
telemChannelNames = ['CG Accel Lateral',
                     'CG Accel Longitudinal',
                     'Ground Speed',
                     'Tire Slip Ratio FL',
                     'Tire Slip Ratio FR',
                     'Tire Slip Ratio RL',
                     'Tire Slip Ratio RR']
telem = {channel: np.array([]) for channel in telemChannelNames}
removeQuotes = lambda x: x.replace('"', '').replace("'", "")
for motecCSVFile in motecCSVFiles:
    print(f"\tProcessing {motecCSVFile}")

    # Read channel names
    channelNames = np.loadtxt(motecCSVFile, dtype=str, delimiter=',', converters=removeQuotes,
                              skiprows=rowChannelNames, max_rows=1)

    # Read telem data corresponding to the desired channel names
    cols = [np.argwhere(channelNames == telemChannelName)[0][0] for telemChannelName in telemChannelNames]
    telemData = np.loadtxt(motecCSVFile, dtype=float, delimiter=',', converters=removeQuotes, usecols=cols, skiprows=rowDataStart)
    for i, col in enumerate(cols):
        telem[telemChannelNames[i]] = telemData[:, i]

    # Convert telem units to SI
    telem['CG Accel Lateral'] *= g_to_ms2
    telem['CG Accel Longitudinal'] *= g_to_ms2
    telem['Ground Speed'] *= kph_to_ms
    telem['Tire Slip Ratio FL'] *= percent_to_ratio
    telem['Tire Slip Ratio FR'] *= percent_to_ratio
    telem['Tire Slip Ratio RL'] *= percent_to_ratio
    telem['Tire Slip Ratio RR'] *= percent_to_ratio

    # Filter data points
    indsSpeed = np.where(telem['Ground Speed'] <= vCarCutoff)[0]
    indsAccel = np.where(telem['CG Accel Longitudinal'] >= aCarCutoff)[0]
    AAccel = np.arctan2(telem['CG Accel Lateral'], telem['CG Accel Longitudinal'])
    indsAngle = np.where(np.abs(AAccel) <= ACutoff)[0]
    indsFilter = np.intersect1d(indsSpeed, indsAccel)
    indsFilter = np.intersect1d(indsFilter, indsAngle)
    telem = {key: val[indsFilter] for key, val in telem.items()}

    # Calculate rSlip using estimate of whether the car is FWD, RWD or AWD during that data point
    rSlipF = (telem['Tire Slip Ratio FL'] + telem['Tire Slip Ratio FR']) / 2
    rSlipR = (telem['Tire Slip Ratio RL'] + telem['Tire Slip Ratio RR']) / 2
    indsFWD = np.where(rSlipF > 0.5 * rSlipR)[0]
    indsRWD = np.where(rSlipR > 0.5 * rSlipF)[0]
    indsAWD = np.intersect1d(indsFWD, indsRWD)
    rSlip = np.zeros_like(rSlipF)
    rSlip[indsFWD] = rSlipF[indsFWD]
    rSlip[indsRWD] = rSlipR[indsRWD]
    rSlip[indsAWD] = (rSlipF[indsAWD] + rSlipR[indsAWD]) / 2

    # Estimate FTyre by acceleration and vehicle mass (ignores track slope, drag etc.)
    FTyre = params['mCar'] * telem['CG Accel Longitudinal']

    # Collate (rSlip, FTyre) data points
    dataPoints['rSlip'] = np.hstack((dataPoints['rSlip'], rSlip))
    dataPoints['FTyre'] = np.hstack((dataPoints['FTyre'], FTyre))

# Estimate ballpark tyre model parameters for the initial guess
indsSort = np.argsort(dataPoints['FTyre'])
indUpper = int(0.99 * (len(indsSort) - 1))
rSlipUpper = dataPoints['rSlip'][indsSort][indUpper]
FTyreUpper = dataPoints['FTyre'][indsSort][indUpper]
CTyreB = 1.55741 / rSlipUpper
CTyreC = 2
CTyreD = FTyreUpper
CTyreE = 1
tyreParamsIters = np.array([[CTyreB, CTyreC, CTyreD, CTyreE]])

# Set bounds for tyre model parameters
boundLower = np.array([0, 0, 0, 0])
boundUpper = np.array([np.inf, np.inf, np.inf, 1])

# Tyre fitting function
def fit_tyre_model(tyreParams, rSlip_points, FTyre_points, optParamInds=None):
    """
    Optimise indexes optParamInds of the tyre parameters tyreParams to minimise squared error to the data points
    (rSlip_points, FTyre_points)
    """
    tyreParamsOpt = tyreParams.copy()
    if optParamInds is None:
        optParamInds = np.arange(0, len(tyreParamsOpt), 1, dtype=int)
    x0 = tyreParamsOpt[optParamInds]
    bounds = (boundLower[optParamInds],
              boundUpper[optParamInds])

    def calc_FTyre_error(x, rSlip_points, FTyre_points):
        """Calculate vector of residuals between tyre model FTyre prediction and the data points"""
        tyreParamsOpt[optParamInds] = x
        return FTyre_points - calc_FTyre(rSlip_points, *tyreParamsOpt)

    result = scipy.optimize.least_squares(calc_FTyre_error, x0, bounds=bounds, args=(rSlip_points, FTyre_points))
    tyreParamsOpt[optParamInds] = result.x
    return tyreParamsOpt

# Least squares fit for CTyreB and CTyreD first for a better initial guess
print(f"Fitting tyre to {len(dataPoints['rSlip'])} data points")
rSlip_points = dataPoints['rSlip']
FTyre_points = dataPoints['FTyre']
tyreParamsOpt = fit_tyre_model(tyreParamsIters[-1], rSlip_points, FTyre_points, optParamInds=[0, 2])
tyreParamsIters = np.vstack((tyreParamsIters, tyreParamsOpt))

# Remove anomalous data points and optimise fit - repeat until no more anomalous data points
while True:
    # Remove anomalies (allow double the anomaly ratio if first removal iteration)
    FTyreModel = calc_FTyre(rSlip_points, *tyreParamsIters[-1])
    FTyreModel[np.where(FTyreModel == 0)[0]] += 1e-3
    rAnomalyActual = rAnomaly * 2 if len(tyreParamsIters) <= 2 else rAnomaly
    indsNominal = np.where(np.abs((FTyreModel - FTyre_points) / FTyreModel) < rAnomalyActual)[0]

    # If no anomalies, stop the fitting process - otherwise re-fit to nominal data points only
    if len(indsNominal) == len(rSlip_points):
        break
    else:
        rSlip_points = rSlip_points[indsNominal]
        FTyre_points = FTyre_points[indsNominal]
        tyreParamsOpt = fit_tyre_model(tyreParamsIters[-1], rSlip_points, FTyre_points)
        tyreParamsIters = np.vstack((tyreParamsIters, tyreParamsOpt))

# Print fitted tyre parameters (as variable names)
print("\nFitted tyre parameters:")
paramNames = ['CTyreB', 'CTyreC', 'CTyreD', 'CTyreE']
for i, name in enumerate(paramNames):
    print(f"\t{name} = {tyreParamsOpt[i]}")

# Print fitted tyre parameters (for VehicleParams dictionary)
print("\nFitted tyre parameters:")
buffer = 22
paramNames = ['CTyreB', 'CTyreC', 'CTyreD', 'CTyreE']
for i, name in enumerate(paramNames):
    paramStr = str(tyreParamsOpt[i]) + ','
    print(f"\t'{name}': {paramStr}{' ' * (buffer - len(paramStr))}# Pacejka magic formula parameter (-)")

# Plot iterations of fitted tyre model against the processed telemetry data points
NPointsModel = 100
rSlipMax = max(0.2, max(dataPoints['rSlip']))
rSlipModel = np.linspace(0, rSlipMax, NPointsModel)
plt.plot(dataPoints['rSlip'], dataPoints['FTyre'], marker='.', alpha=0.2, ls='', c='C0', label='Telemetry (Anomaly)')
plt.plot(rSlip_points, FTyre_points, marker='.', ls='', c='C0', label='Telemetry (Nominal)')
for i, tyreParams in enumerate(tyreParamsIters):
    FTyreModel = calc_FTyre(rSlipModel, *tyreParams)
    plt.plot(rSlipModel, FTyreModel, alpha=(i + 1) / len(tyreParamsIters), c='C1', label=f'Model (Iter {i})')
plt.xlim(0, max(dataPoints['rSlip']))
plt.ylim(0, max(dataPoints['FTyre']))
plt.xlabel("rSlip (1)")
plt.ylabel("FTyre (N)")
plt.legend()
plt.show()
