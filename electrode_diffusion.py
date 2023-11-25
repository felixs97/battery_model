#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 18:47:16 2023

@author: felix
"""

from battery_model import prp, res, j, F
import numpy as np
import pde
import matplotlib.pyplot as plt

def c_electrode(layer):
    L  = prp[layer]['L']
    nx = len(res[layer]['x'])
    D = prp[layer]['D']
    c0 = prp[layer]['c0']
    
    # Define grid & inital conditons
    grid = pde.CartesianGrid([[0, L]], nx)
    state = pde.ScalarField(grid, c0)
    
    # set boundary conditions
    if layer == 'A':
        lbc = {"derivative": 0}
        rbc = {"derivative": -j/F/D}
    else:
        lbc = {"derivative": j/F/D}
        rbc = {"derivative": 0}
   
    # Define equation
    eq = pde.DiffusionPDE(D, bc=[lbc, rbc])
    storage = pde.MemoryStorage()
    eq.solve(state, t_range=3600, dt=0.01, tracker=["progress", storage.tracker(1)])

    sol = []
    for idx, field in storage.items():
        sol.append(field.data)
    sol = np.array(sol)
    
    return sol[300]

c_a = c_electrode('A')
c_c = c_electrode('C')
x_a = res['A']['x']
x_c = res['C']['x']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(x_a*10**(6), c_a)
ax2.plot(x_c*10**(6), c_c)
    
ax1.set_xlabel("x / $\mu m$")
ax1.set_ylabel("c$_{Li}$ / $mol m^{-3}$")
ax1.set_xlim(min(x_a)*10**6, max(x_a)*10**6)
ax1.legend()
ax1.set_title("Concentration along Anode")

ax2.set_xlabel("x / $\mu m$")
ax2.set_ylabel("c$_{Li}$ / $mol m^{-3}$")
ax2.set_xlim(min(x_c)*10**6, max(x_c)*10**6)
ax2.set_title("Concentration along Cathode")