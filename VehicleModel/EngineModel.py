"""
Engine model - linear interpolation of lookup table between wEngine and MEngineMax
Provides max engine torque at a given engine angular speed and the forwards derivative with respect to engine angular speed
"""
import numpy as np

def calc_MEngineMax(wEngine, engineLUT):
    """Linear interpolation of lookup table"""
    return np.interp(wEngine, engineLUT[:, 0], engineLUT[:, 1])

def calc_dMEngineMax_dwEngine(wEngine, engineLUT):
    """Numerical central gradient of lookup table"""
    eps = 1e-3
    return (calc_MEngineMax(wEngine + eps, engineLUT) - calc_MEngineMax(wEngine - eps, engineLUT)) / (2 * eps)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from VehicleParams import params

    NPoints = 1000
    eps = 1e-9

    ws = np.linspace(params['EngineLUT'][0, 0], params['EngineLUT'][-1, 0], NPoints)

    MEngineMaxs = calc_MEngineMax(ws, params['EngineLUT'])
    dMEngineMax_dwEngine_calcs = [calc_dMEngineMax_dwEngine(w, params['EngineLUT']) for w in ws]
    dMEngineMax_dwEngine_ests = [(calc_MEngineMax(w + eps, params['EngineLUT']) - calc_MEngineMax(w - eps, params['EngineLUT'])) / (2 * eps) for w in ws]

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(ws, MEngineMaxs, label='MEngineMax')
    axs[0].plot([params['wEngineLimit'], params['wEngineLimit']], [min(MEngineMaxs), max(MEngineMaxs)], label='Limiter')
    axs[0].set_ylabel('MEngineMax (Nm)')
    axs[0].legend()

    axs[1].plot(ws, dMEngineMax_dwEngine_calcs, label='Calculated')
    axs[1].plot(ws, dMEngineMax_dwEngine_ests, label='Estimated')
    axs[1].set_ylabel('dMEngineMax_dwEngine (Nm/(rad/s))')
    axs[1].set_xlabel('wEngine (rad/s)')
    axs[1].legend()

    plt.show()
