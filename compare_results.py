#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 11:32:37 2023

@author: felix
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#%% Functions     
def create_df(res):
    x = np.concatenate((res['A']['x'], res['AS']['x'], res['E']['x'], res['CS']['x'], res['C']['x']))
    T = np.concatenate((res['A']['T'], res['AS']['T'], res['E']['T'], res['CS']['T'], res['C']['T']))
    phi = np.concatenate((res['A']['phi'], res['AS']['phi'], res['E']['phi'], res['CS']['phi'], res['C']['phi']))
    Jq = np.concatenate((res['A']['Jq'], res['AS']['Jq'], res['E']['Jq'], res['CS']['Jq'], res['C']['Jq']))
    sigma = np.concatenate((res['A']['sigma'], res['AS']['sigma'], res['E']['sigma'], res['CS']['sigma'], res['C']['sigma']))
    
    return pd.DataFrame({'x'     : x, 
                          'T'    : T,
                          'phi'  : phi,
                          'Jq'   : Jq,
                          'sigma': sigma})
def load_dict(dirName):
    filePath = os.path.join(os.getcwd(), 'results', dirName, 'results_dict.pkl')
    with open(filePath, 'rb') as fp:
        res = pickle.load(fp)
    return res

def plot_mu(res):
    plt.figure()
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    
    # Vertical spans 
    plt.axvspan(0, res['A']['x'][-1]*10**(6), facecolor='b', alpha=0.1)
    plt.axvspan(res['C']['x'][0]*10**(6), res['C']['x'][-1]*10**(6), facecolor='r', alpha=0.1)
    plt.axvspan(res['AS']['x'][0]*10**(6), res['AS']['x'][-1]*10**(6), facecolor='k', alpha=0.4)
    plt.axvspan(res['CS']['x'][0]*10**(6), res['CS']['x'][-1]*10**(6), facecolor='k', alpha=0.4)
    
    plt.plot(res['E']['x']*10**(6), res['E']['mu1'], label='LiPF$_6$', linewidth=2)
    plt.plot(res['E']['x']*10**(6), res['E']['mu2'], label='DEC', linewidth=2)
    
    plt.title('Chemical Potential', fontsize=14)
    plt.xlim(res['A']['x'][-2]*10**(6), res['C']['x'][1]*10**(6))
    plt.xlabel(' x / ${\mu m}$', fontsize=12)
    plt.ylabel('$\mu$ / $J/mol$', fontsize=12)
    plt.legend()
    
    # Add frame
    ax = plt.gca()
    ax.tick_params(labelsize=12)
    
def plot_c(res):
    plt.figure()
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    
    # Vertical spans 
    plt.axvspan(0, res['A']['x'][-1]*10**(6), facecolor='b', alpha=0.1)
    plt.axvspan(res['C']['x'][0]*10**(6), res['C']['x'][-1]*10**(6), facecolor='r', alpha=0.1)
    plt.axvspan(res['AS']['x'][0]*10**(6), res['AS']['x'][-1]*10**(6), facecolor='k', alpha=0.4)
    plt.axvspan(res['CS']['x'][0]*10**(6), res['CS']['x'][-1]*10**(6), facecolor='k', alpha=0.4)
    
    plt.plot(res['E']['x']*10**(6), res['E']['c1'], label='LiPF$_6$', linewidth=2)
    plt.plot(res['E']['x']*10**(6), res['E']['c2'], label='DEC', linewidth=2)
    
    plt.title('Concentration Profile', fontsize=14)
    plt.xlim(res['A']['x'][-2]*10**(6), res['C']['x'][1]*10**(6))
    plt.xlabel(' x / ${\mu m}$', fontsize=12)
    plt.ylabel('c / $mol/m^3$', fontsize=12)
    plt.legend()
    
    # Add frame
    ax = plt.gca()
    ax.tick_params(labelsize=12)

def create_plot(quantity, ylabel, title, labels, dicts, plot_sf=None):
    fig, ax = plt.subplots(dpi=200)
    
    # Vertical spans 
    ax.axvspan(0, dicts[0]['A']['x'][-1]*10**(6), facecolor='b', alpha=0.1)
    ax.axvspan(dicts[0]['C']['x'][0]*10**(6), dicts[0]['C']['x'][-1]*10**(6), facecolor='r', alpha=0.1)
    ax.axvspan(dicts[0]['AS']['x'][0]*10**(6), dicts[0]['AS']['x'][-1]*10**(6), facecolor='k', alpha=0.3)
    ax.axvspan(dicts[0]['CS']['x'][0]*10**(6), dicts[0]['CS']['x'][-1]*10**(6), facecolor='k', alpha=0.3)
    
    colors = ["b", "r", "g", "o", "c", "m", "y"]
    lines = []
    
    for i, res in enumerate(dicts):
        lines.append(Line2D([0], [0], color = colors[i], linestyle="-"))
        for layer in ['A', 'AS', 'E', 'CS', 'C']:
            if layer in ['AS', 'CS']:
                if plot_sf == "*" :
                    ax.plot(res[layer]['x']*10**(6), res[layer][quantity], label=labels[i], color=colors[i], marker="*")
                elif plot_sf != "no":
                    ax.plot(res[layer]['x']*10**(6), res[layer][quantity], label=labels[i], color=colors[i], linewidth=2)
            else:
                ax.plot(res[layer]['x']*10**(6), res[layer][quantity], label=labels[i], color=colors[i], linewidth=2)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlim(res['A']['x'][0]*10**(6), res['C']['x'][-1]*10**(6))
    
    # Set y-axis tick labels
    if quantity == "T":
        y_ticks = ax.get_yticks()
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:.4f}" for tick in y_ticks])
    
    ax.set_xlabel(' x / ${\mu m}$', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(lines, labels)

#%% Main  
dirNameLCO = 'C6_LCO'
resLCO = load_dict(dirNameLCO)
dfLCO  = create_df(resLCO)

dirNameLFP = 'C6_LFP'
resLFP = load_dict(dirNameLFP)
dfLFP  = create_df(resLFP)

dirNameLFPinit = 'C6_LFP_initialPi'
resLFPinit = load_dict(dirNameLFPinit)
dfLFPinit  = create_df(resLFPinit)

dirNameLFPexMass = 'C6_LFP_initialPi_exclMass'
resLFPexMass = load_dict(dirNameLFPexMass)
dfLFPexMass  = create_df(resLFPexMass)

dirNameLCOexMass = 'C6_LCO_exclMass'
resLCOexMass = load_dict(dirNameLCOexMass)
dfLCOexMass  = create_df(resLCOexMass)

dirNameLCOk1 = 'C6_LCO_k1'
resLCOk1 = load_dict(dirNameLCOk1)
dfLCOk1  = create_df(resLCOk1)

dirNameLCOkSw = 'C6_LCO_kSwitched'
resLCOkSw = load_dict(dirNameLCOkSw)
dfLCOkSw  = create_df(resLCOkSw)


#%%% compare LFP and LCO
labels = ["LFP", "LCO"]
dicts = [resLFP, resLCO]
create_plot("T", "T / $K$", "Temperature profile", labels, dicts)
create_plot("phi", "$\phi$ / $V$", "Potential profile", labels, dicts)
create_plot("Jq", "J'$_q$ / $W m^{-2}$", "Measurable heatflux", labels, dicts, plot_sf="no")
create_plot("sigma", "$\sigma$ / $W m^{-2} K^{-1}$", "Entropy production", labels, dicts, plot_sf="*")


#plot_mu(resLCO)
#plot_c(resLCO)
