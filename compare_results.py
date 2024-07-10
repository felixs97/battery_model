#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 12:46:41 2024

@author: felix
"""

from params_sys import Tamb
import params_LFP, params_LFP2, params_LFP3, params_LFP4  
import params_LCO
import classes_new as c
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.lines import Line2D


model1 = c.LiionModel("local RHE", params_LFP)  
model1.init_mesh({"Anode":       100, 
                 "Electrolyte":  20,
                 "Cathode":     100}) 
model1.boundary_conditions(Tamb, Tamb)
model1.solve()

model2 = c.LiionModel("average RHE", params_LFP2)  
model2.init_mesh({"Anode":       100, 
                 "Electrolyte":  20,
                 "Cathode":     100}) 
model2.boundary_conditions(Tamb, Tamb)
model2.solve()

model3 = c.LiionModel("without RHE", params_LFP3)  
model3.init_mesh({"Anode":       100, 
                 "Electrolyte":  20,
                 "Cathode":     100}) 
model3.boundary_conditions(Tamb, Tamb)
model3.solve()

model4 = c.LiionModel("LCO", params_LCO)  
model4.init_mesh({"Anode":       100, 
                 "Electrolyte":  20,
                 "Cathode":     100}) 
model4.boundary_conditions(Tamb, Tamb)
model4.solve()

model5 = c.LiionModel("LFP", params_LFP)  
model5.init_mesh({"Anode":       100, 
                 "Electrolyte":  20,
                 "Cathode":     100}) 
model5.boundary_conditions(Tamb, Tamb)
model5.solve()

def plot(*models):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)

    # Vertical spans 
    ax.axvspan(0, 74, facecolor='b', alpha=0.1)
    ax.axvspan(86.06, 153.06, facecolor='r', alpha=0.1)
    
    ax.set_xlabel(r' $x$ / $\mu$m', fontsize=12)
    ax.set_xlim(0, 153.06)
    
    ax.set_ylabel("$T$ / K")
    
    data = {}
    colors = ["r", "green", "orange", "blue"]
    linestyle = ["-", "-", "-", "--"]
    linewidth = [2, 2, 2, 1]
    for model in models:
        data[model.name] = {"T": np.array([]), "x": np.array([])}
        
        for submodel in [model.anode, model.anode_sf, model.electrolyte, model.cathode_sf, model.cathode]:
            if submodel in [model.anode_sf, model.cathode_sf]:
                data[model.name]["x"] = np.append(data[model.name]["x"], submodel.vars["x"] * 10**6) 
                data[model.name]["T"] = np.append(data[model.name]["T"], np.ones(2) * submodel.vars["T"])
            else:
                data[model.name]["x"] = np.append(data[model.name]["x"], submodel.vars["x"] * 10**6) 
                data[model.name]["T"] = np.append(data[model.name]["T"], submodel.vars["T"])
        
    for i, (name, values) in enumerate(data.items()):
        ax.plot(values["x"], values["T"], color=colors[i], linestyle =linestyle[i], linewidth=linewidth[i], label=name)
        ax.legend()

    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{tick:.4f}" for tick in y_ticks])
    
    ax.plot([0,153.06], [290,290], color="blue", linestyle="--")

def plot_Jq(*models):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)

    # Vertical spans 
    ax.axvspan(0, 74, facecolor='b', alpha=0.1)
    ax.axvspan(86.06, 153.06, facecolor='r', alpha=0.1)
    
    ax.set_xlabel(r' $x$ / $\mu$m', fontsize=12)
    ax.set_xlim(0, 153.06)
    
    ax.set_ylabel("$J'_q$ / W m$^{-2}$")
    
    colors = ["r", "green", "orange", "blue"]
    labels = []
    lines = ()
    for i, model in enumerate(models):
        labels.append(model.name)
        lines = lines + (Line2D([0], [0], color = colors[i], linestyle="-"),)
        for submodel in [model.anode, model.electrolyte, model.cathode]:
            ax.plot(submodel.vars["x"]*10**6, submodel.vars["Jq"], color=colors[i], linewidth=2)
            
    ax.legend(lines, labels)





def plot_surface(*models, surface):
    fig, ax = plt.subplots(figsize=(3, 4), dpi=200)
    
    # Vertical spans 
    ax.axvspan(0, 74, facecolor='b', alpha=0.1)
    ax.axvspan(86.06, 153.06, facecolor='r', alpha=0.1)
    ax.axvspan(74, 74.05, facecolor='k', alpha=0.3)
    ax.axvspan(86.05, 86.06, facecolor='k', alpha=0.3)

    ax.set_xlabel(r' $x$ / $\mu$m', fontsize=12)
    
    if surface == "Anode":
        # Anode Surface:
        ax.set_xlim(73.95, 74.1)
        #ax.set_ylim(290.00065, 290.00075)
        ax.set_ylim(290.0005, 290.001)
    else:
        # Cathode Surface:
        ax.set_xlim(86, 86.15)
        #ax.set_ylim(289.99885, 289.99895)
        ax.set_ylim(289.9987, 289.9992)
    
    ax.set_ylabel("$T$ / K")
    
    data = {}
    colors = ["r", "green", "orange", "blue"]
    linestyle = ["-", "-", "-", "--"]
    linewidth = [2, 2, 2, 1]
    for model in models:
        data[model.name] = {"T": np.array([]), "x": np.array([])}
        
        for submodel in [model.anode, model.anode_sf, model.electrolyte, model.cathode_sf, model.cathode]:
            if submodel in [model.anode_sf, model.cathode_sf]:
                data[model.name]["x"] = np.append(data[model.name]["x"], submodel.vars["x"] * 10**6) 
                data[model.name]["T"] = np.append(data[model.name]["T"], np.ones(2) * submodel.vars["T"])
            else:
                data[model.name]["x"] = np.append(data[model.name]["x"], submodel.vars["x"] * 10**6) 
                data[model.name]["T"] = np.append(data[model.name]["T"], submodel.vars["T"])
        
    for i, (name, values) in enumerate(data.items()):
        ax.plot(values["x"], values["T"], color=colors[i], linestyle = linestyle[i], linewidth=linewidth[i], label=name)
    
    ax.legend(loc="upper right")

    #y_ticks = ax.get_yticks()
    #ax.set_yticks(y_ticks)
    #ax.set_yticklabels([f"{tick:.5f}" for tick in y_ticks])
    
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.4f}"))
    


#plot_Jq(model5, model4)
#plot(model5, model4)
#plot(model1, model2, model3)
plot_Jq(model1, model2, model3)

print(f"{model3.name}, out - in: {model3.cathode.vars['Jq'][-1] - model3.anode.vars['Jq'][0]}")
print(f"{model2.name}, out - in: {model2.cathode.vars['Jq'][-2] - model2.anode.vars['Jq'][0]}")
print(f"{model1.name}, out - in: {model1.cathode.vars['Jq'][-1] - model1.anode.vars['Jq'][0]}")

print(f"{model4.name}, out - in: {model4.cathode.vars['Jq'][-1] - model4.anode.vars['Jq'][0]}")
print(f"{model5.name}, out - in: {model5.cathode.vars['Jq'][-1] - model5.anode.vars['Jq'][0]}")
#plot(model1)
#plot_surface(model1, surface="Anode")
#plot_surface(model1, surface="Cathode")

#plot_surface(model1, model2, model3, model4, surface="Anode")
#plot_surface(model1, model2, model3, model4, surface="Cathode")
