"""
Optimise full-state feedback gains and other regime-switching parameters for best performance and robustness

See EvalControl.py for the cost function to minimise

Instead of directly tuning gains, maybe optimise the eigenvalues and other regime-switching parameters with pole placement

But actually a more likely approach for the control gain optimisation is robust control, probably more like using the
control system synthesis methods in https://python-control.readthedocs.io/en/0.10.2/functions.html#control-system-synthesis
    - Linear-quadratic regulator
    - Discrete-time linear-quadratic regulator
    - H2 control synthesis
    - H-infinity control synthesis
    - Mixed-sensitivity H-infinity synthesis - probably the best option? - https://www.youtube.com/watch?v=kRt7H0k8A4k
For each of these control system synthesis methods, effectively the problem shifts from choosing the individual gains/
eigenvalues, to designing the cost function used for the control gain optimisation - normalising vCar error to 1, do a
meta-optimisation of the cost function parameters (and regime-switching parameters)
    - Costs for each of the remaining states (wEngine, rSlip)
    - Costs for each of the inputs (rThrottleLinear, rClutchLinear)
    - Regime-switching parameters
    - Ratio of designed uncertainty compared to the evaluated uncertainty (for mixed-sensitivity H-infinity synthesis)

Make an OptControl script for each of these:
    - OptControlGains
    - OptControlPoles
    - OptControlLQR
    - OptControlDLQR
    - etc.
"""
# TODO: this