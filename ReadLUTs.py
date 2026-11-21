"""
Read .ini files for engine torque and throttle LUTs and print the equivalent 2D lists
"""
import os
import numpy as np

# Settings
LUTFolder = r"Tatuus FA01 Data/tatuusfa1/data"  # Folder containing the LUT files
LUTEngineFile = "power.lut"                     # Filename of the RPM | torque LUT
LUTThrottleFile = "throttle.lut"                # Filename of the throttle input | actual throttle LUT
pBoost = 1.1                                    # Boost (bar)
NRound = 3                                      # Rounding decimal places

# Unit conversion constants
radsToRPM = 30 / np.pi
RPMToRads = np.pi / 30

# Engine torque LUT
EngineLUT = np.loadtxt(os.path.join(LUTFolder, LUTEngineFile), comments=';', delimiter='|')
EngineLUT[:, 0] *= RPMToRads
EngineLUT = np.round(EngineLUT, NRound)
EngineLUTList = EngineLUT.tolist()
print("\nEngineLUT")
print(f"np.array({EngineLUTList})")

# Throttle LUT
ThrottleLUT = np.loadtxt(os.path.join(LUTFolder, LUTThrottleFile), comments=';', delimiter='|')
ThrottleLUT /= 100
ThrottleLUT = np.round(ThrottleLUT, NRound)
ThrottleLUTList = ThrottleLUT.tolist()
print("\nThrottleLUT")
print(f"np.array({ThrottleLUTList})")
