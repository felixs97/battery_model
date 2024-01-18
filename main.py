#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:14:36 2023

@author: felix
"""
from params_sys import Tamb
import params_LFP 
import classes_new as c

model = c.LiionModel("LFP", params_LFP)  
model.init_mesh({"Anode":       100, 
                 "Electrolyte":  20,
                 "Cathode":     100}) 
model.boundary_conditions(Tamb, Tamb)
model.solve()
#model.plot()
model.plot_single("c")
model.consistency_check(show_subsystems=False)

dU = (model.cathode.vars["phi"][-1])
dQ = (model.cathode.vars["Jq"][-1] - model.anode.vars["Jq"][0])
#print(dQ)

P = 30*dU
Pideal = 30*3.3
Plost = Pideal-P

print((P+dQ))
