"""
Tyre model - 4-parameter Pacejka magic formula
Provides tyre force at slip ratio and the derivative with respect to slip ratio
"""
import numpy as np


def calc_FTyre(rSlip, CTyreB, CTyreC, CTyreD, CTyreE):
    """Pacejka magic formula"""
    return CTyreD * np.sin(CTyreC * np.arctan(CTyreB * rSlip - CTyreE * (CTyreB * rSlip - np.arctan(CTyreB * rSlip))))


def calc_dFTyre_drSlip(rSlip, CTyreB, CTyreC, CTyreD, CTyreE):
    """Pacejka magic formula, derivative referenced from my derivation: https://www.desmos.com/calculator/cdebm8v3wh"""
    n1 = CTyreB * CTyreC * CTyreD * (1 - CTyreE + CTyreE / (CTyreB ** 2 * rSlip ** 2 + 1))
    n2 = np.cos(CTyreC * np.arctan(CTyreB * rSlip - CTyreB * CTyreE * rSlip + CTyreE * np.arctan(CTyreB * rSlip)))
    d = (CTyreB * rSlip - CTyreB * CTyreE * rSlip + CTyreE * np.arctan(CTyreB * rSlip)) ** 2 + 1
    return (n1 * n2) / d


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from VehicleParams import params

    rSlipMax = 0.5
    NPoints = 1000
    eps = 1e-9

    rs = np.linspace(0, rSlipMax, NPoints)
    tyreParams = (params['CTyreB'], params['CTyreC'], params['CTyreD'], params['CTyreE'])

    FTyres = calc_FTyre(rs, *tyreParams)
    dFtyre_drSlip_calcs = calc_dFTyre_drSlip(rs, *tyreParams)
    dFTyre_drSlip_ests = [(calc_FTyre(r + eps, *tyreParams) - calc_FTyre(r - eps, *tyreParams)) / (2 * eps) for r in rs]

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(rs, FTyres)
    axs[0].set_ylabel('FTyre (N)')

    axs[1].plot(rs, dFtyre_drSlip_calcs, label='Calculated')
    axs[1].plot(rs, dFTyre_drSlip_ests, label='Estimated')
    axs[1].legend()
    axs[1].set_ylabel('dFtyre_drSlip (N/ratio)')
    axs[1].set_xlabel('rSlip (ratio)')

    plt.show()

