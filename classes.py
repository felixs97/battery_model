#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 08:41:49 2023

@author: felix
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from params_sys import j, F, Tamb, R

class Subsystem:
    def __init__(self, parameter):
        self.params = parameter
        self.variables = {}
    
    def init_mesh(self, nx, init_x):
        self.x = np.linspace(init_x, init_x + self.params["length"], nx)

class Surface(Subsystem):
    def __init__(self, parameter):
        super().__init__(parameter)
        self.__def_params()
    
    def __overpotential(self, j0):
        return 2*R*Tamb / F * np.log10(j/j0)
        
    def __def_params(self):
        L = self.params["length"]
        OCP = self.params["OCP"]
        j0 = self.params["exchange current density"]
        lambda_ = self.params["thermal conductivity"]
        k = self.params["correction factor"]

        self.params["gibbs energy"] = - OCP * F
        self.params["overpotential"] = self.__overpotential(j0)
        self.params["thermal conductivity"] = lambda_ / (k*L)
        
    def dphi(self, T_s, T_i, T_o, pi_i, pi_o):
        eta = self.params["overpotential"]
        dG = self.params["gibbs energy"]
        
        return - pi_i/(T_o*F)*(T_s - T_i) - pi_o/(T_o*F)*(T_o - T_s) - eta - dG/F

class Electrolyte(Subsystem):
    def __init__(self, parameter, mass_trans = True):
        super().__init__(parameter)
        self.mass_trans = mass_trans
        self.__def_params()
    
    def __def_params(self):
        l_LL = self.params["onsager coefficient LL"]
        l_LD = self.params["onsager coefficient LD"]
        l_DD = self.params["onsager coefficient DD"]
        q_L  = self.params["heat of transfer L"]
        q_D  = self.params["heat of transfer D"]
        t_L  = self.params["transference coefficient L"]
        t_D  = self.params["transference coefficient D"]
        pi   = self.params["peltier coefficient"]
        
        if not self.mass_trans:
            a_L, a_D, b_L, b_D, a_q, b_phi = 0, 0, 0, 0, 0, 0
            b_q, a_phi = pi, pi
        else:
            a_L = l_DD*(l_LL*q_L - l_LD*q_D)/(l_LL*l_DD - l_LD**2)
            a_D = q_D - (l_LD/l_DD * a_L)
            b_L = (t_L*l_DD - t_D*l_LD)/(l_LL*l_DD - l_LD**2)
            b_D = t_D/l_DD - (l_LD/l_DD * b_L)
            a_q = q_L*l_LL*a_L + q_D*l_DD*a_D
            b_q = pi - q_L*l_LL*b_L - q_D*l_DD*b_D
            a_phi = pi - t_L*a_L - t_D*a_D
            b_phi = t_L*b_L + t_D*b_D

        self.params["a_L"] = a_L
        self.params["a_D"] = a_D
        self.params["b_L"] = b_L
        self.params["b_D"] = b_D
        self.params["a_q"] = a_q
        self.params["b_q"] = b_q
        self.params["a_phi"] = a_phi
        self.params["b_phi"] = b_phi
    
    def dSdx(self, x, S):
        a_q = self.params["a_q"]
        a_phi = self.params["a_phi"]
        b_phi = self.params["b_phi"]
        lambda_ = self.params["thermal conductivity"]
        kappa = self.params["electric conductivity"]
        
        T, dTdx = S
        denom = (lambda_ - a_q/T**2)
        rhs = -j*a_phi/(T * F) / denom * dTdx - b_phi*T*j**2/F**2 / denom - j**2/kappa / denom
        
        return [dTdx, rhs]
    
    def Jq(self, T, dTdx):
        lambda_ = self.params["thermal conductivity"]
        a_q = self.params["a_q"]
        b_q = self.params["b_q"]
        
        self.variables["Jq"] = -(lambda_ - a_q/T**2)*dTdx + j/F*b_q
    
    def dphidx(self, T, dTdx):
        kappa = self.params["electric conductivity"]
        a_phi = self.params["a_phi"]
        b_phi = self.params["b_phi"]
        
        self.variables["dphidx"] = - a_phi/(T*F)*dTdx - (b_phi*T/F**2 + 1/kappa)*j
    
    def dmuLdx(self, T, dTdx):
        a_L = self.params["a_L"]
        b_L = self.params["b_L"]
        
        self.variables["dmuLdx"] = a_L/T*dTdx - b_L*T/F*j
    
    def dmuDdx(self, T, dTdx):
        a_D = self.params["a_D"]
        b_D = self.params["b_D"]
        
        self.variables["dmuDdx"] = a_D/T*dTdx - b_D*T/F*j
    
    def __dcLdx(self, x, c, T, dmuDdx, dmuLdx):
        TDF_LL = self.params["thermodynamic factor LL"]
        TDF_DD = self.params["thermodynamic factor DD"]
        TDF_LD = self.params["thermodynamic factor LD"]
        TDF_DL = self.params["thermodynamic factor DL"]
        
        return (dmuLdx - TDF_LD/TDF_DD*dmuDdx)/(TDF_LL - TDF_LD*TDF_DL/TDF_DD) * c/(R*T)
    
    def c(self, c0, T, dmuDdx, dmuLdx):
        x = self.x
        
        sol = solve_ivp(self.__dcLdx, (x[0], x[-1]), [c0], t_eval=x, args=(T, dmuDdx, dmuLdx))
        c, dcdx = sol.y[0], sol.y[1]
        
        self.variables["c"], self.variables["dcdx"] = c, dcdx
 
class Electrode(Subsystem):
    def __init__(self, parameter, mass_trans=True):
        super().__init__(parameter)
        self.mass_trans = mass_trans
        self.__def_params()
    
    def __def_params(self):
        if not self.mass_trans:
            self.params["diffusion coefficient"] = 0
        
    def dSdx(self, x, S):
        pi = self.params["peltier coefficient"]
        lambda_ = self.params["thermal conductivity"]
        kappa = self.params["electric conductivity"]

        T, dTdx = S
        rhs     = - pi*j / (lambda_*F*T) * dTdx - j**2 / (lambda_*kappa)
        
        return[dTdx, rhs]
    
    def Jq(self, dTdx):
        lambda_ = self.params["thermal conductivity"]
        pi = self.params["peltier coefficient"]
        
        self.variables["Jq"] = - lambda_*dTdx + pi/F*j
    
    def dphidx(self, T, dTdx):
        pi = self.params["peltier coefficient"]
        kappa = self.params["electric conductivity"]
        
        self.variables["dphidx"] = pi/F * dTdx/T + j/kappa
    
    def dcdx(self):
        D = self.params["diffusion coefficient"]
        
        self.variables["dcdx"] = -j / (D*F)
    
    def dmudx(self, c, dcdx):
        TDF = self.params["thermodynamic factor"]
        
        self.variables["dmudx"] = TDF*R*Tamb/c * dcdx
  
class LiionModel:
    def __init__(self, params):
        self.params = params
        self.add_submodels()
    
    def add_submodels(self):
        self.submodels = {}
        self.submodels["Anode"] = Electrode(self.params.anode)
        self.submodels["Anode Surface"] = Surface(self.params.anode_sf)
        self.submodels["Electrolyte"] = Electrolyte(self.params.electrolyte)
        self.submodels["Cathode Surface"] = Surface(self.params.cathode_sf)
        self.submodels["Cathode"] = Electrode(self.params.cathode)
    
    def init_mesh(self, bulk_mesh):
        self.submodels["Anode"].init_mesh(bulk_mesh["Anode"], 0)
        self.submodels["Anode Surface"].init_mesh(2, self.submodels["Anode"].x[-1])
        self.submodels["Electrolyte"].init_mesh(bulk_mesh["Electrolyte"], self.submodels["Anode Surface"].x[-1])
        self.submodels["Cathode Surface"].init_mesh(2, self.submodels["Electrolyte"].x[-1])
        self.submodels["Cathode"].init_mesh(bulk_mesh["Cathode"], self.submodels["Cathode Surface"].x[-1])
        
    def boundary_conditions(self, lbc, rbc):
        self.bc = {
            "lbc" : lbc,
            "rbc" : rbc}
    
    def __solve_surface(self, i, o, sf):
        lambda_i, lambda_o, lambda_s = self.submodels[i].params["thermal conductivity"], self.submodels[o].params["thermal conductivity"], self.submodels[sf].params["thermal conductivity"]
        pi_i, pi_o, pi_s = self.submodels[i].params["peltier coefficient"], self.submodels[o].params["peltier coefficient"], self.submodels[sf].params["peltier heat"]
        eta = self.submodels[sf].params["overpotential"]
        a_q, b_q = self.submodels["Electrolyte"].params["a_q"], self.submodels["Electrolyte"].params["b_q"]
        
        T_i = self.submodels[i].variables["T"][-1]
        dTdx_i = self.submodels[i].variables["dTdx"][-1]
        
        if sf == "Anode Surface":
            dT_is = lambda_i/lambda_s * dTdx_i
            dT_so = dT_is + pi_s*j/(lambda_s*F) - eta*j/lambda_s
            dTdx_oi = (lambda_s * dT_so + j/F*(b_q - pi_o))/(lambda_o - a_q/T_i**2)
        else:
            dT_is = ((lambda_i - a_q/T_i**2)*dTdx_i + j/F*(pi_i - b_q))/lambda_s
            dT_so = dT_is + pi_s*j/(lambda_s*F) - eta*j/lambda_s
            dTdx_oi =  lambda_s/lambda_o * dT_so
        
        T_s = T_i + dT_is
        T_o = T_s + dT_so
        
        self.submodels[sf].variables["T"] = T_s
        self.submodels[o].variables["T"] = T_o
        self.submodels[o].variables["dTdx"] = dTdx_oi
    
    def __solve_bulk(self, T0, dTdx0, domain):
        S0 = (T0, dTdx0)
        x = self.submodels[domain].x
        sol = solve_ivp(self.submodels[domain].dSdx, (x[0], x[-1]), S0, t_eval=x)
        self.submodels[domain].variables["T"], self.submodels[domain].variables["dTdx"] = sol.y
    
    def __solve_temp(self, T0, dTdx0):
        self.__solve_bulk(T0, dTdx0, "Anode")
        self.__solve_surface("Anode", "Electrolyte", "Anode Surface")
        self.__solve_bulk(self.submodels["Electrolyte"].variables["T"], self.submodels["Electrolyte"].variables["dTdx"], "Electrolyte")
        self.__solve_surface("Electrolyte", "Cathode", "Cathode Surface")
        self.__solve_bulk(self.submodels["Cathode"].variables["T"], self.submodels["Cathode"].variables["dTdx"], "Cathode")
    
    def __opt_inital_guess(self, dTdxGuess):
        self.__solve_temp(self.bc["lbc"], dTdxGuess[0])
        return [self.bc["rbc"] - self.submodels["Cathode"].variables["T"][-1]]
    
    def solve(self):
        dTdx0, = fsolve(self.__opt_inital_guess, [1])
        self.__solve_temp(self.bc["lbc"], dTdx0)
        
        self.submodels["Anode"].Jq(self.submodels["Anode"].variables["dTdx"])
        self.submodels["Electrolyte"].Jq(self.submodels["Electrolyte"].variables["T"], self.submodels["Electrolyte"].variables["dTdx"])
        self.submodels["Cathode"].Jq(self.submodels["Cathode"].variables["dTdx"])


        




