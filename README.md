Launch control, based around a 3-state system with a full-state feedback controller, linearised around an "optimal" launch trajectory and using gain scheduling with car speed/engine RPM

This project really just is a fun way for me to apply the content from the Control and Instrumentation lectures at Uni, and extend into learning and applying more advanced control theory

Note for LUA app implementation, the linear throttle/clutch inputs must be reverse-mapped with LUTs to their pedal inputs (though iirc only throttle has LUTs, the clutch i think is always linear?)

Already before I've even really implemented anything, I can see a few areas for improvement:
- Optimal launch trajectory probably requires optimal control to solve (would be a good introduction for doing optimal control for my lap sim project)
- Vehicle model probably should be implemented in Modelica, then exported as an FMU, so that the linearised system matrices can be computed easily and the model fidelity can be increased more easily
- Controller probably should be a model-predictive controller but that would be out-of-scope i think

---

Ideation dump:
- Currently P control only
  - Implement this first to validate the concept
- I control can be added relatively easily
  - Add the integral states (theta_engine and integral of rSlip) to the state vector, making the associated changes to the system matrices
  - Change in integral error for a given regime is: error * tStepInput * rRegime * (1 - BSaturated), where rRegime is the ratio of control given to the regime, and BSaturated is whether the throttle/clutch control input is saturated
- D control more challenging to add
  - Differentiate the theta_engine and theta_wheel Lagrange equations with respect to time
  - Note the control input will now be the rate of change of rThrottleLinear and the rate of change of rClutchLinear
  - Then re-derive the system matrices