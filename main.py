#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:14:36 2023

@author: felix
"""
from params_sys import Tamb
import params_LFP 
import classes_new as c

model = c.LiionModel("LFP", params_LFP, mass_trans = True)  
model.init_mesh({"Anode":       100, 
                 "Electrolyte":  20,
                 "Cathode":     100}) 
model.boundary_conditions(Tamb, Tamb)
model.solve()
#model.plot()
model.plot_single("sigma accumulated")
model.consistency_check()