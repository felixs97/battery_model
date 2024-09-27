#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:14:36 2023

@author: felix
"""
from params_sys import Tamb, Tamb_left, Tamb_right
import params_LFP
import classes as c

model = c.LiionModel("Baseline Scenario", params_LFP)  
model.init_mesh({"Anode":       100, 
                 "Electrolyte":  20,
                 "Cathode":     100}) 
model.boundary_conditions(Tamb_left, Tamb_right)
model.solve()
model.plot()

# optional output about model consistency
model.consistency_check(show_subsystems=True)

