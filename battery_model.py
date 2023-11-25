#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:54:12 2023

@author: felix
"""

import numpy as np
import pandas as pd
import os
import pickle
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

#%% Define functions
#%%% Setting properties
def set_electrode_prp(L, pi, lambda_, kappa):
    """
    create dictionary for electrode parameters

    Parameters
    ----------
    L : float
        length / m
    pi : float
        peltier coefficient / J/mol
    lambda_ : float
        thermal conductivity / W/m/K
    kappa : float
        electric conductivity / S/m

    Returns
    -------
    Electrode : dictionary
        containing the material value accessable due to the name.
    """
    names  = ['L', 'pi', 'lambda_', 'kappa']
    values = [L, pi, lambda_, kappa]
    Electrode = {}
    for i, name in enumerate(names):
        Electrode[name] = values[i]
    return Electrode

def set_electrolyte_prp(L, pi, lambda_, kappa, t1, t2, l11, l22, l12, q1, q2, 
                        TDF_LL, TDF_LD, TDF_DL, TDF_DD, inclMass = True):
    """
    create dictionary for electrolyte parameters 
    defining constant coefficients for better readablity

    Parameters
    ----------
    index 1: LiPF6
    index 2: DEC
    L : float
        length / m
    pi : float
        peltier coefficient / J/mol
    lambda_ : float
        thermal conductivity / W/m/K
    kappa : float
        electric conductivity / S/m
    t1/t2 : float
        transference coefficient / - (dimenonsless)
    l11/l22/l12 : float
        onsager mass transfer coefficients
    q1/q2 : float
        heats of transfer 
    TDF_LL/TDF_LD/TDF_DL/TDF_DD : float
        thermodynamic factor
    Returns
    -------
    Electrolyte : dict
        DESCRIPTION.
    """
    if inclMass:
        # defining constant coefficients
        a1   = l22*(l11*q1 - l12*q2)/(l11*l22 - l12**2)
        a2   = q2 - (l12/l22 * a1)
        b1   = (t1*l22 - t2*l12)/(l11*l22 - l12**2)
        b2   = t2/l22 - (l12/l22 * b1)
        aq   = (q1*l11*a1) + (q2*l22*a2)
        bq   = pi - q1*l11*b1 - q2*l22*b2
        aphi = pi - (t1*a1) - (t2*a2)
        bphi = (t1*b1) + (t2*b2)
    else:
        a1, a2, b1, b2, aq, bphi = 0, 0, 0, 0, 0, 0
        bq, aphi = pi, pi
    
    names = ['L', 'pi', 'lambda_', 'kappa', 'a1', 'a2', 'b1', 'b2', 'aq', 'bq', 'aphi', 'bphi', 
             'TDF_LL', 'TDF_LD', 'TDF_DL', 'TDF_DD']
    values = [L, pi, lambda_, kappa, a1, a2, b1, b2, aq, bq, aphi, bphi, TDF_LL, TDF_LD, TDF_DL, TDF_DD]
    Electrolyte = {}
    for i, name in enumerate(names):
        Electrolyte[name] = values[i]
    return Electrolyte

def set_surface_prp(L, lambda_, dPhi0, j0, k, pi):
    dG       =  - dPhi0 * F
    eta      = 2*R*Tamb/F * np.log10(j/j0)
    lambda_s = lambda_ / (k*L)
    
    names  = ['L', 'lambda_', 'pi', 'eta', 'dG']
    values = [L, lambda_s, pi, eta, dG]
    Surface = {}
    for i, name in enumerate(names):
        Surface[name] = values[i]
    return Surface

#%%% System of ODEs
def dSdx(x, S, layer):
    pi      = prp[layer]['pi']
    lambda_ = prp[layer]['lambda_']
    kappa   = prp[layer]['kappa']
    
    T, dTdx = S
    
    if layer == 'E':
        aq      = prp[layer]['aq']
        aphi    = prp[layer]['aphi']
        bphi    = prp[layer]['bphi']
        
        denom   = lambda_ - aq/(T**2)
        rhs     = -j*aphi/(T*F*denom) * dTdx - bphi*j**2/(F**2*denom) * T - j**2/(kappa*denom)
    else:
        rhs     = - pi*j / (lambda_*F*T) * dTdx - j**2 / (lambda_*kappa)
    return[dTdx, rhs]

def dSdx_sf(i, o, sf):
    lambda_o, lambda_i, lambda_s = prp[o]['lambda_'], prp[i]['lambda_'], prp[sf]['lambda_']
    pi_o,     pi_i,     pi_s     = prp[o]['pi'], prp[i]['pi'], prp[sf]['pi']
    eta      = prp[sf]['eta']
    T        = res[i]['T']
    dTdx     = res[i]['dTdx']
    
    dTis = lambda_i/lambda_s * dTdx[-1]
    Ts = dTis + T[-1]
    dTso = -((pi_i*j/(F*Ts) - lambda_s)*dTis - pi_s*j/F + eta*j)/(pi_o*j/(F*Ts) + lambda_s)
    To = dTso + Ts
    
    dTdxo = lambda_s/lambda_o * dTso
    
    return[To, dTdxo, Ts]

def solve_odes(T0, dTdx0):
    # Anode
    S0 = (T0, dTdx0)
    x = res['A']['x']
    sol = solve_ivp(dSdx, (x[0], x[-1]), S0, t_eval=x, args='A')
    res['A']['T'], res['A']['dTdx'] = sol.y
    
    # Anode Surface as boundary condition
    Te, dTdxe, Ts = dSdx_sf('A', 'E', 'AS')
    res['AS']['T'] = [Ts, Ts]
    
    # Electrolyte
    S0 = (Te, dTdxe)
    x = res['E']['x']
    sol = solve_ivp(dSdx, (x[0], x[-1]), S0, t_eval=x, args='E')
    res['E']['T'], res['E']['dTdx'] = sol.y
    
    # Cathode Surface as boundary condtion
    Tc, dTdxc, Ts = dSdx_sf('E', 'C', 'CS')
    res['CS']['T'] = [Ts, Ts]
    
    # Cathode
    S0 = (Tc, dTdxc)
    x = res['C']['x']
    sol = solve_ivp(dSdx, (x[0], x[-1]), S0, t_eval=x, args='C')
    res['C']['T'], res['C']['dTdx'] = sol.y

def opt_inital_guess(dTdxGuess):
    solve_odes(T0l, dTdxGuess[0])
    return[T0r - res['C']['T'][-1]]
#%%% Calculate electrical Potential
 
def phi_bulk(layer, phi0):
    pi      = prp[layer]['pi']
    kappa   = prp[layer]['kappa']
    T       = res[layer]['T']
    dTdx    = res[layer]['dTdx']
    dx      = np.gradient(res[layer]['x'])
    
    if layer == 'E':
        aphi    = prp[layer]['aphi']
        bphi    = prp[layer]['bphi']
        
        dphi = (-aphi/(T*F)*dTdx - bphi*T*j/(F**2) - j/kappa) * dx
    else:
        dphi = (- pi/F * dTdx/T - j/kappa) * dx 
    
    phi = np.zeros(dphi.size)
    phi[0] = phi0
    for i in range(0, len(phi)-1):
        phi[i+1] = phi[i] + dphi[i]
        
    return phi
    
def phi_sf(i, o, s):
    pi_i, pi_o = prp[i]['pi'], prp[o]['pi']
    Ti, To, Ts = res[i]['T'][-1], res[o]['T'][0], res[s]['T'][0]
    eta        = prp[s]['eta']
    dG         = prp[s]['dG']
    
    dphi = -pi_i*(Ts-Ti)/(Ti*F) - pi_o*(To-Ts)/(To*F) - eta - dG/F
    
    return [res[i]['phi'][-1], res[i]['phi'][-1] + dphi]

#%%% Calculate chemical Potential
def dmudx(layer):
    a1   = prp['E']['a1']
    a2   = prp['E']['a2']
    b1   = prp['E']['b1']
    b2   = prp['E']['b2']
    T    = res['E']['T']
    dTdx = res['E']['dTdx']
    
    dmu1dx = -a1 * dTdx/T + b1*j*T/F
    dmu2dx = -a2 * dTdx/T + b2*j*T/F
    
    dmudx = [dmu1dx, dmu2dx]

    return dmudx

def mu(layer, mu0):
    dx      = np.gradient(res[layer]['x'])

    dmu1dx = res[layer]['dmu1dx']
    dmu2dx = res[layer]['dmu2dx']
    mu01, mu02 = mu0
    
    mu1 = np.zeros(dmu1dx.size)
    mu2 = np.zeros(dmu2dx.size)
    
    for i in range(0, len(dmu1dx)-1):
        mu1[i+1] = mu1[i] + dmu1dx[i]*dx[i]
        mu2[i+1] = mu2[i] + dmu2dx[i]*dx[i]
    
    mu = [mu1, mu2]

    return mu
#%%% Calculate Li-concentration
def dcdx(x, c, TDF_LL, TDF_LD, TDF_DL, TDF_DD, dmu1dx, dmu2dx):
    return (dmu1dx - TDF_LD/TDF_DD * dmu2dx)/(TDF_LL - TDF_LD/TDF_DD*TDF_DL) * c/(R*Tamb)

def c(c0):
    x = res['E']['x']
    TDF_LL, TDF_LD, TDF_DL, TDF_DD = prp['E']['TDF_LL'], prp['E']['TDF_LD'], prp['E']['TDF_DL'], prp['E']['TDF_DD']
    dmu1dx, dmu2dx = np.mean(res['E']['dmu1dx']), np.mean(res['E']['dmu1dx'])
    
    sol = solve_ivp(dcdx, (x[0], x[-1]), [c0], t_eval=x, args=(TDF_LL, TDF_LD, TDF_DL, TDF_DD, dmu1dx, dmu2dx))
    c = sol.y[0]
    
    return c
#%%% Calcualte Heatflux
def Jq(layer):
    lambda_ = prp[layer]['lambda_']
    dTdx    = res[layer]['dTdx']
    
    if layer == 'E':
        aq = prp[layer]['aq']
        bq = prp[layer]['bq']
        T  = res[layer]['T']
        
        Jq = -(lambda_ - aq/(T**2))*dTdx + j*bq/F
    else:  
        pi = prp[layer]['pi']
        
        Jq = -lambda_*dTdx + pi*j/F
        
    return Jq

#%%% Calculate Entropy production
def sigma_bulk(layer):
    Jq     = res[layer]['Jq']
    T      = res[layer]['T']
    dTdx   = res[layer]['dTdx']
    dphidx = np.gradient(res[layer]['phi'])
    
    sigma = -Jq * dTdx/(T**2) - j*dphidx/T
    
    return sigma

def sigma_sf(i, o, s):
    Jqi, Jqo = res[s]['Jq'][0], res[s]['Jq'][-1]
    Ti, To, Ts = res[i]['T'][-1], res[o]['T'][0], res[s]['T'][0]
    dG   = prp[s]['dG']
    dphi = res[s]['phi'][-1] - res[s]['phi'][0]
    dTis, dTso = Ts-Ti, To-Ts
    
    sigma = -Jqi * dTis/(Ti*To) -Jqo * dTso/(Ti*To) - j/Ti*(dG/F + dphi)
    
    return [sigma, sigma]

#%%% Consistent Check
def integrate_sigma():
    sigma0 = 0
    for layer in layers: 
        if layer in ["AS", "CS"]:
            res[layer]["sigma_accum"] = (res[layer]["sigma"][0] + sigma0) * np.ones(2)
        else:
            res[layer]["sigma_accum"] = np.cumsum(res[layer]["sigma"]) + sigma0
        #sigma0 = res[layer]["sigma_accum"][-1]

def calc_entropy_diff(layer):
    Ti, To   = res[layer]["T"][0],  res[layer]["T"][-1]
    Jqi, Jqo = res[layer]["Jq"][0], res[layer]["Jq"][-1]

    Jsi = 0#1/Ti*Jqi 
    Jso = 1/To*Jqo + j/F*100
    dJs = -(Jsi - Jso)
    print(dJs)
#%%% Create Plot
def plot_all():
    fig, axes = plt.subplot(nrows=2, ncols=2, figsize=(10,8), dpi=200)
    
    pass

def plot_single(quantity, ylabel, title, plot_sf=False):
    fig, ax = plt.subplots(dpi=200)
    
    # Vertical spans 
    ax.axvspan(0, res['A']['x'][-1]*10**(6), facecolor='b', alpha=0.1)
    ax.axvspan(res['C']['x'][0]*10**(6), res['C']['x'][-1]*10**(6), facecolor='r', alpha=0.1)
    ax.axvspan(res['AS']['x'][0]*10**(6), res['AS']['x'][-1]*10**(6), facecolor='k', alpha=0.3)
    ax.axvspan(res['CS']['x'][0]*10**(6), res['CS']['x'][-1]*10**(6), facecolor='k', alpha=0.3)
    
    for layer in layers:
        if layer in ['AS', 'CS']:
            if plot_sf:
                ax.plot(res[layer]['x']*10**(6), res[layer][quantity], marker="*", color="r")
        else:
            ax.plot(res[layer]['x']*10**(6), res[layer][quantity], color="r", linewidth=2)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlim(res['A']['x'][0]*10**(6), res['C']['x'][-1]*10**(6))
    
    # Set y-axis tick labels
    if quantity == "T":
        y_ticks = ax.get_yticks()
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:.4f}" for tick in y_ticks])
    
    ax.set_xlabel(' x / ${\mu m}$', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

def plot_profile(quantity, ylabel, title):
    
    x = np.concatenate((res['A']['x'], res['AS']['x'], res['E']['x'], res['CS']['x'], res['C']['x']))
    y = np.concatenate((res['A'][quantity], res['AS'][quantity], res['E'][quantity], res['CS'][quantity], res['C'][quantity]))

    fig, ax = plt.subplots(dpi=200)
    
    # Vertical spans 
    ax.axvspan(0, res['A']['x'][-1]*10**(6), facecolor='b', alpha=0.1)
    ax.axvspan(res['C']['x'][0]*10**(6), res['C']['x'][-1]*10**(6), facecolor='r', alpha=0.1)
    ax.axvspan(res['AS']['x'][0]*10**(6), res['AS']['x'][-1]*10**(6), facecolor='k', alpha=0.3)
    ax.axvspan(res['CS']['x'][0]*10**(6), res['CS']['x'][-1]*10**(6), facecolor='k', alpha=0.3)
    
    ax.plot(x*10**6, y, color="r", linewidth=2)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlim(res['A']['x'][0]*10**(6), res['C']['x'][-1]*10**(6))
    
    # Set y-axis tick labels
    if quantity == "T":
        y_ticks = ax.get_yticks()
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:.4f}" for tick in y_ticks])
    
    ax.set_xlabel(' x / ${\mu m}$', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

def plot_sf_temp(x, y, sf):
    plt.figure()
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    
    # Vertical spans 
    plt.axvspan(0, res['A']['x'][-1]*10**(6), facecolor='b', alpha=0.1)
    plt.axvspan(res['C']['x'][0]*10**(6), res['C']['x'][-1]*10**(6), facecolor='r', alpha=0.1)
    plt.axvspan(res['AS']['x'][0]*10**(6), res['AS']['x'][-1]*10**(6), facecolor='k', alpha=0.4)
    plt.axvspan(res['CS']['x'][0]*10**(6), res['CS']['x'][-1]*10**(6), facecolor='k', alpha=0.4)
    
    plt.plot(x*10**(6), y)
    if sf == 'AS':
        surface = 'Anode Surface'
    else: 
        surface = 'Cathode Surface'
    
    title = 'Zoom on ' + surface
    plt.title(title, fontsize=14)
    plt.xlim(res[sf]['x'][0]*10**(6)-0.1, res[sf]['x'][-1]*10**(6)+0.1)
    plt.xlabel(' x / ${\mu m}$', fontsize=12)
    plt.ylabel('T / K', fontsize=12)
    
    plt.ylim(res[sf]['T'][0]-5/100000, res[sf]['T'][0]+5/100000)

#%%% Save data
def create_dir(dirName):
    dirPath = os.path.join(os.getcwd(), 'results', dirName)
    if not os.path.isdir(dirPath):
        os.mkdir(dirPath)
    return dirPath
    
def save_to_csv(dirName, overwrite=False):
    
    x = np.concatenate((res['A']['x'], res['AS']['x'], res['E']['x'], res['CS']['x'], res['C']['x']))
    T = np.concatenate((res['A']['T'], res['AS']['T'], res['E']['T'], res['CS']['T'], res['C']['T']))
    phi = np.concatenate((res['A']['phi'], res['AS']['phi'], res['E']['phi'], res['CS']['phi'], res['C']['phi']))
    Jq = np.concatenate((res['A']['Jq'], res['AS']['Jq'], res['E']['Jq'], res['CS']['Jq'], res['C']['Jq']))
    sigma = np.concatenate((res['A']['sigma'], res['AS']['sigma'], res['E']['sigma'], res['CS']['sigma'], res['C']['sigma']))

    
    result = pd.DataFrame({'x / m'              : x, 
                           'T / K'              : T, 
                           'phi / V'            : phi, 
                           'Jq / J/(m2 s)'      : Jq, 
                           'sigma / J/(m2 s K)' : sigma})
    
    dirPath = create_dir(dirName)
    filePath = os.path.join(dirPath,'data.csv')
    if not os.path.exists(filePath) or overwrite:
        result.to_csv(filePath, index=False)

def save_result_dict(dirName):
    dirPath = create_dir(dirName)
    filePath = os.path.join(dirPath, 'results_dict.pkl')
    with open(filePath, 'wb') as fp:
        pickle.dump(res, fp)
#%% Define properties
# system properties 
j    = 30          # A / m2
F    = 96485       # C / mol
Tamb = 290         # K
R    = 8.314       # J / mol / K

# electrochemical cell
layers = ['A', 'AS', 'E', 'CS', 'C']
prp = {layer: {} for layer in layers} # initialise empty dict for properties
res = {layer: {} for layer in layers} # initialise empty dict for results

prp['A'] = set_electrode_prp(  L       = 74 * 10**(-6),          # m
                               pi      =  5.74*Tamb,             # J / mol
                               lambda_ =  1.11,                  # W / K / m
                               kappa   = 2203.8 )                # S / m

prp['C'] = set_electrode_prp(  L       = 67 * 10**(-6),          # m
                               pi      = 14.5*Tamb,              # J / mol
                               lambda_ = 2.1,#0.32,                   # W / K / m
                               kappa   = 10)#6.75 )                  # S / m

prp['E'] = set_electrolyte_prp(L       =  12 * 10**(-6),         # m
                               pi      =  24.7 * 10**3,          # J / mol
                               lambda_ =   0.2,                  # W / K / m
                               kappa   =   0.23,                 # S / m 
                               t1      = - 0.97,
                               t2      =   0.9,
                               l11     =   3.7 * 10**(-11)*Tamb, # mol^2 K / (J m s)
                               l22     =  53.7 * 10**(-11)*Tamb, # mol^2 K / (J m s)
                               l12     =  11.3 * 10**(-11)*Tamb, # mol^2 K / (J m s)
                               q1      =   1.6 * 10**3,          # J / mol
                               q2      =   0.3 * 10**3,          # J / mol
                               TDF_LL  =   1.45,
                               TDF_LD  = - 0.29,
                               TDF_DL  = - 0.98,
                               TDF_DD  =   1.23, inclMass=True)

prp['AS'] = set_surface_prp(   L       =  50 * 10**(-9),         # m
                               lambda_ =   0.65,                 # W / K / m
                               dPhi0   =   0.1,                  # V
                               j0      =   0.8,                  # A / m2  
                               k       =  14,
                               pi      = -104000)                 # J / mol

prp['CS'] = set_surface_prp(   L       =  10 * 10**(-9),         # m
                               lambda_ =   1.11,                 # W / K / m
                               dPhi0   =   3.9,#3.45,                 # V
                               j0      =   26,#1.7,                  # A / m2  
                               k       =   110,
                               pi      =   49000)               # J / mol

#%% Define mesh 
nx = [100, 20, 100]

res['A']['x']  = np.linspace(0, prp['A']['L'], nx[0])
res['E']['x']  = np.linspace(res['A']['x'][-1] + prp['AS']['L'], res['A']['x'][-1] + prp['AS']['L'] + prp['E']['L'], nx[1])
res['C']['x']  = np.linspace(res['E']['x'][-1] + prp['CS']['L'], res['E']['x'][-1] + prp['CS']['L'] + prp['C']['L'], nx[2])
res['AS']['x'] = np.array([res['A']['x'][-1], res['E']['x'][0]])
res['CS']['x'] = np.array([res['E']['x'][-1], res['C']['x'][0]])

#%% Define boundary conditions
T0l = Tamb
T0r = Tamb

#%% Solve for inital condition
dTdxGuess = 1
dTdx0, = fsolve(opt_inital_guess, [dTdxGuess])

#%% Solve System
solve_odes(T0l, dTdx0)

#%%% Electrical Potential
res['A']['phi']  = phi_bulk('A', 0)
res['AS']['phi'] = phi_sf('A', 'E', 'AS')
res['E']['phi']  = phi_bulk('E', res['AS']['phi'][-1])
res['CS']['phi'] = phi_sf('E', 'C', 'CS')
res['C']['phi']  = phi_bulk('C', res['CS']['phi'][-1])

#%%% Chemical Potential
res['E']['dmu1dx'], res['E']['dmu2dx'] = dmudx('E')
res['E']['mu1'], res['E']['mu2'] = mu('E', [0,0])

#%%% Concentration Profile
c0 = 1000 # Li-concentration on anode surface
res['E']['c'] = c(c0)
# Concentration in porous electrodes is set to a fix value -> concentration in solid not in electrolyte between solid
res["A"]["c"] = np.ones(res["A"]["x"].size) * 14000#c0 
res["C"]["c"] = np.ones(res["C"]["x"].size) * 10000#res['E']['c'][-1]

#%%% Heatflux
res['A']['Jq']  = Jq('A')
res['E']['Jq']  = Jq('E')
res['C']['Jq']  = Jq('C')
res['AS']['Jq'] = np.array([res['A']['Jq'][-1], res['E']['Jq'][0]])
res['CS']['Jq'] = np.array([res['E']['Jq'][-1], res['C']['Jq'][0]])

#%%% Entropy Produciton
res['A']['sigma']  = sigma_bulk('A')
res['AS']['sigma'] = sigma_sf('A', 'E', 'AS')
res['E']['sigma']  = sigma_bulk('E')
res['CS']['sigma'] = sigma_sf('E', 'C', 'CS')
res['C']['sigma']  = sigma_bulk('C')

#%%% Consistent Check
integrate_sigma()
calc_entropy_diff("A")

#%% Create Plots
#plot_profile("T", "T / $K$", "Temperature profile")
#plot_profile("phi", "$\phi$ / $V$", "Electrical Potential Profile")
plot_profile("sigma_accum", "$\sigma / Wm^{-2}K^{-1}$", "Entropy Production accumulated")

#plot_single("Jq", "J'$_q$ / $Wm^{-2}$", "Measurable Heatflux")
#plot_single("sigma", "$\sigma / Wm^{-2}K^{-1}$", "Entropy Production", plot_sf=True)
#plot_single("c", "c$_{Li}$ / $m^3 mol^{-1}$", "Concentraton Profile of Lithium")
#%%% Plot surface temperature
#plot_sf_temp(x, T, 'AS')
#plot_sf_temp(x, T, 'CS')

#%% Save data
dirName = 'C6_LCO'
save_to_csv(dirName, overwrite=False)
save_result_dict(dirName)

#plt.plot(res['A']['x']*10**6, res['A']['c']*10**(-3))