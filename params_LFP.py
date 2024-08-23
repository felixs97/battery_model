#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:00:33 2023

@author: felix
"""
from params_sys import Tamb

parameter = {
    "Anode": {
        "length":                       74    *10**(-6),        # m
        "thermal conductivity":          1.11,                  # W/m/K
        "heat capacity L":               24.6,                  # J/mol/K
        "electric conductivity":      2204,                     # S/m
        "peltier coefficient":        -104 * 10**3,             # J/mol  
        "diffusion coefficient":        10    *10**(-11),       # m2/s
        "initial concentration":     0.8*30555,                 # mol/m3
        "thermodynamic factor":          0.1                    # -
        },
    "Cathode": {
        "length":                       67    *10**(-6),        # m
        "thermal conductivity":          0.32,                  # W/m/K
        "heat capacity L":               24.6,                  # J/mol/K
        "electric conductivity":        6.75,                   # S/m
        "peltier coefficient":         122 * 10**3,             # J/mol  
        "diffusion coefficient":        10    *10**(-11),       # m2/s
        "initial concentration":      0.2*22800,                # mol/m3
        "thermodynamic factor":         0.5                     # -
        },
    "Electrolyte": {
        "length":                       12    *10**(-6),        # m
        "thermal conductivity":          0.2,                   # W/m/K
        "electric conductivity":         0.23,                  # S/m
        "peltier coefficient":          -24.7  * 10**3,         # J/mol
        "transference coefficient L":  - 0.97,
        "transference coefficient D":    0.9,
        "onsager coefficient LL":        3.7  * 10**(-11)*Tamb, # mol^2 K/(J m s)
        "onsager coefficient DD":       53.7  * 10**(-11)*Tamb, # mol^2 K/(J m s)
        "onsager coefficient LD":       11.3  * 10**(-11)*Tamb, # mol^2 K/(J m s)
        "heat of transfer L":            1.6  * 10**3,          # J/mol
        "heat of transfer D":            0.3  * 10**3,          # J/mol
        "thermodynamic factor LL":       1.45,                  # -
        "thermodynamic factor LD":     - 0.29,                  # -
        "thermodynamic factor DL":     - 0.98,                  # -
        "thermodynamic factor DD":       1.23,                  # -
        "initial concentration L":     1000 + 50,               # mol/m3
        "initial concentration D":     4127 - 6.5,              # mol/m3
        "initial concentration E":     7500                     # mol/m3
        },
    "Anode Surface": {
        "length":                      50     * 10**(-9),      # m
        "thermal conductivity":         0.65,                  # W/m/K
        "OCP":                         -0.15,                  # V
        "exchange current density":     0.8,                   # A/m^2
        "peltier heat":              -104     * 10**3,         # J/mol
        "correction factor":           14                      # -
        }, 
    "Cathode Surface": {
        "length":                      10     * 10**(-9),      # m
        "thermal conductivity":         1.11,                  # W/m/K
        "OCP":                          3.45,                  # V
        "exchange current density":     1.7,                   # A/m^2
        "peltier heat":               122     * 10**3,         # J/mol
        "correction factor":          110                      # -
        }
    }

