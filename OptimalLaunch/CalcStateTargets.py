"""
Calculate state targets for each operating regime
"""
import scipy
import numpy as np
from VehicleModel.TyreModel import calc_dFTyre_drSlip


def calc_wEngineTarget_1(wEngineMarginPower, params):
    """Calculate the engine angular speed target for operating regime 1 (max engine speed - wEngineMarginPower)"""
    return params['wEngineLimit'] - wEngineMarginPower


def calc_wEngineTarget_3(params):
    """Calculate the engine angular speed target for operating regime 3 (for max engine torque)"""
    return params['EngineLUT'][np.argmax(params['EngineLUT'][:, 1]), 0]


def calc_rSlipTarget_124(params):
    """Calculate the slip ratio target for operating regimes 1, 2, 4 (for max FTyre)"""
    tyreParams = (params['CTyreB'], params['CTyreC'], params['CTyreD'], params['CTyreE'])
    return scipy.optimize.root(calc_dFTyre_drSlip, 1e-3, args=tyreParams).x[0]