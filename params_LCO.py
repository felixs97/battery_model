#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:00:33 2023

@author: felix
"""
from params_sys import Tamb

anode = {
    "length":                       74    *10**(-6),        # m
    "thermal conductivity":          1.11,                  # W/m/K
    "electric conductivity":      2204.8,                   # S/m
    "peltier coefficient":           5.74 *Tamb,            # J/mol
    "thermodynamic factor":          1                      # -
    }

cathode = {
    "length":                       67    *10**(-6),        # m
    "thermal conductivity":          2.1,                   # W/m/K
    "electric conductivity":        10,                     # S/m
    "peltier coefficient":          14.50 *Tamb,            # J/mol
    "thermodynamic factor":         1                       # -
    } 

electrolyte = {
    "length":                       23    *10**(-6),        # m
    "thermal conductivity":          0.2,                   # W/m/K
    "electric conductivity":         0.23,                  # S/m
    "peltier coefficient":          24.7  * 10**3,          # J/mol
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
    }

anode_sf = {
    "length":                      50     * 10**(-9),      # m
    "thermal conductivity":         0.65,                  # W/m/K
    "OCP":                          0.1,                   # V
    "exchange current density":     0.8,                   # A/m^2
    "peltier heat":             - 104     * 10**3,         # J/mol
    "correction factor":           14                      # -
    }

cathode_sf = {
    "length":                      10     * 10**(-9),      # m
    "thermal conductivity":         1.11,                  # W/m/K
    "OCP":                          3.9,                   # V
    "exchange current density":     26,                    # A/m^2
    "peltier heat":                49     * 10**3,         # J/mol
    "correction factor":          110                      # -
    }