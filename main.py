#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:14:36 2023

@author: felix
"""
from params_sys import Tamb
import params_LCO 
import classes as c

LCOmodel = c.LiionModel(params_LCO)
LCOmodel.init_mesh({"Anode":       100,
                 "Electrolyte": 20,
                 "Cathode":     100})
LCOmodel.boundary_conditions(Tamb, Tamb+1)
LCOmodel.solve()