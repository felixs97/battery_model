#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:14:36 2023

@author: felix
"""
from params_sys import Tamb, j, F
import params_LFP
import params_LCO
import classes_new as c
import numpy as np
from scipy.integrate import cumtrapz

model = c.LiionModel("LFP", params_LFP)  
model.init_mesh({"Anode":       100, 
                 "Electrolyte":  20,
                 "Cathode":     100}) 
model.boundary_conditions(Tamb, Tamb)
model.solve()
model.plot()
#model.plot_single("Jq")
#model.plot_single("T")
#model.consistency_check(show_subsystems=False)



T_as = model.anode_sf.vars["T"]
T_cs = model.cathode_sf.vars["T"]
Pi_as = model.anode_sf.params["peltier heat"]
Pi_cs = model.cathode_sf.params["peltier heat"]

print(Pi_cs/T_cs + Pi_as/T_as)
print(Pi_cs + Pi_as)

"""
dU = (model.cathode.vars["phi"][-1])
dQ = (model.cathode.vars["Jq"][-1] - model.anode.vars["Jq"][0])
#print(dQ)

P = 30*dU
Pideal = 30*3.3
Plost = Pideal-P


dphi_a = model.anode.vars["phi"][-1]
dphi_c = model.cathode.vars["phi"][-1]-model.cathode.vars["phi"][0]
dphi_e = -(model.electrolyte.vars["phi"][-1]-model.electrolyte.vars["phi"][0])

pi = model.electrolyte.params["peltier coefficient"]
t_L = model.electrolyte.params["transference coefficient L"]
t_D = model.electrolyte.params["transference coefficient D"]
q_L = model.electrolyte.params["heat of transfer L"]
q_D = model.electrolyte.params["heat of transfer D"]
l_LL = model.electrolyte.params["onsager coefficient LL"]
l_DD = model.electrolyte.params["onsager coefficient DD"]
l_LD = model.electrolyte.params["onsager coefficient LD"]
kappa = model.electrolyte.params["electric conductivity"]
lambda_ = model.electrolyte.params["thermal conductivity"]

dmuLdx = model.electrolyte.vars["dmuLdx"]
dmuDdx = model.electrolyte.vars["dmuDdx"]
T = model.electrolyte.vars["T"]
dTdx = model.electrolyte.vars["dTdx"]
Jq = (model.electrolyte.vars["Jq"])[-1]

thermCon = (-lambda_*dTdx)[-1]
muLCon = (-q_L*l_LL*dmuLdx/T)[-1]
muDCon =( -q_D*l_DD*dmuDdx/T)[-1]
elecCon = pi/F*j

print("Electrolyte")
print(f"Jq: {Jq}")
print(f"Sum: {thermCon + muLCon + muDCon + elecCon}")
print(f"thermCon: {thermCon}")
print(f"muLCon: {muLCon}")
print(f"muDCon: {muDCon}")
print(f"elecCon: {elecCon}")

lambda_ = model.anode.params["thermal conductivity"]
pi = model.anode.params["peltier coefficient"]

T = model.anode.vars["T"]
dTdx = model.anode.vars["dTdx"]
Jq = np.mean(model.anode.vars["Jq"])

thermCon = np.mean(-lambda_*dTdx)
elecCon = pi/F*j

print("Anode")
print(f"Jq: {Jq}")
print(f"Sum: {thermCon + elecCon}")
print(f"thermCon: {thermCon}")
print(f"elecCon: {elecCon}")

lambda_ = model.cathode.params["thermal conductivity"]
pi = model.cathode.params["peltier coefficient"]

T = model.cathode.vars["T"]
dTdx = model.cathode.vars["dTdx"]
Jq = np.mean(model.cathode.vars["Jq"])

thermCon = np.mean(-lambda_*dTdx)
elecCon = pi/F*j

print("Cathode")
print(f"Jq: {Jq}")
print(f"Sum: {thermCon + elecCon}")
print(f"thermCon: {thermCon}")
print(f"elecCon: {elecCon}")


"""









