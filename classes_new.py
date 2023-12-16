#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:19:31 2023

@author: felix
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from params_sys import j, F, Tamb, R
import params_LCO

#%% cell model
class LiionModel:
    def __init__(self, param_set):
        self.parameter = param_set.parameter
        self.__add_submodels()
        
    def __add_submodels(self):
        """
        creates an instance of each submodel in the current LiionModel instance
        """
        self.anode       = Electrode(self, "Anode")
        self.anode_sf    = Surface(self, "Anode Surface")
        self.electrolyte = Electrolyte(self, "Electrolyte")
        self.cathode_sf  = Surface(self, "Cathode Surface")
        self.cathode       = Electrode(self, "Cathode")

    def init_mesh(self, nodes):
        """
        create mesh for whole cell
        
        Parameters
        ----------
        nodes : dictionary
            keys: bulk phases (Anode, Electrolyte, Cathode)
            values: number of nodes
        """
        self.anode.init_mesh(0, nodes["Anode"])
        self.anode_sf.init_mesh(self.anode.vars["x"][-1], 2)
        self.electrolyte.init_mesh(self.anode_sf.vars["x"][-1], nodes["Electrolyte"])
        self.cathode_sf.init_mesh(self.electrolyte.vars["x"][-1], 2)
        self.cathode.init_mesh(self.cathode_sf.vars["x"][-1], nodes["Cathode"])
    
    def boundary_conditions(self, lbc, rbc):
        """
        set temperature boundary conditions
        
        Parameters
        ----------
        lbc : float
            left boundary condition (ambient temperature)
        rbc : float
            right boundary condition (ambient temperature)
        """
        self.bc = {
            "lbc": lbc,
            "rbc": rbc}
    
    def solve(self):
        """
        final function to call in order to solve system
        """
        dTdx0, = fsolve(self.opt_inital_guess, [1])
        self.solve_temp_system(dTdx0)
    
    def plot(self):
        pass
    
    def solve_temp_surface(self, i, o, sf):
        """
        solves equations for surface
        T_s:     Temperature of the surface T_s
        T_o:     Temperature of the bulk close to the surface on right-hand-side
        dTdx_oi: Temperature gradient of the bulk close to the surface on right-hand-side
        
        Parameters
        ----------
        i : Submodel object (Electrode or Electrolyte)
            bulk phase on the left side of the surface
        o : Submodel object (Electrode or Electrolyte)
            bulk phase on the right side of the surface
        sf : Surface object
            surface to calculate temperature for 
        """
        lambda_i, lambda_o, lambda_s = i.params["thermal conductivity"], o.params["thermal conductivity"], sf.params["thermal conductivity"]
        pi_i, pi_o, pi_s = i.params["peltier coefficient"], o.params["peltier coefficient"], sf.params["peltier heat"]
        eta = sf.params["overpotential"]
        a_q, b_q = self.electrolyte.params["a_q"], self.electrolyte.params["b_q"]
        
        T_io = i.vars["T"][-1]
        dTdx_io = i.vars["dTdx"][-1]
        
        if sf == "Anode Surface":
            dT_is = lambda_i/lambda_s * dTdx_io
            dT_so = dT_is + pi_s*j/F/lambda_s - eta*j/lambda_s 
            dTdx_oi = (lambda_s * dT_so + j/F*(b_q - pi_o))/(lambda_o - a_q/T_io**2)
        else:
            dT_is = ((lambda_i - a_q/T_io**2)*dTdx_io + j/F*(pi_i - b_q))/lambda_s
            dT_so = dT_is + pi_s*j/(lambda_s*F) - eta*j/lambda_s
            dTdx_oi =  lambda_s/lambda_o * dT_so
        
        T_s = T_io + dT_is
        T_oi = T_s + dT_so
        
        sf.vars["T"]   = T_s
        o.vars["T"]    = T_oi
        o.vars["dTdx"] = dTdx_oi

    def solve_temp_bulk(self, T0, dTdx0, bulk):
        """
        solves inital value problem

        Parameters
        ----------
        T0 : float
            inital Temperature, in this case on the left-hand-side
        dTdx0 : float
            inital Temperature gradient, in this case on the left-hand-side
        bulk : object
            submodel of corresponding bulk phase
        """
        S0 = (T0, dTdx0)
        x = bulk.vars["x"]
        sol = solve_ivp(bulk.temp_function, (x[0], x[-1]), S0, t_eval=x)
        bulk.vars["T"], bulk.vars["dTdx"] = sol.y
        
    def solve_temp_system(self, dTdx_guess):
        """
        solves system of equations for temperature and the gradient in each phase
        by using the defined boundary conditions on the left-hand-side and an inital guess for the temperature gradient

        Parameters
        ----------
        dTdx_guess : float
            guess of the temperature gradient at the left boundary
        """
        self.solve_temp_bulk(self.bc["lbc"], dTdx_guess, self.anode)
        self.solve_temp_surface(self.anode, self.electrolyte, self.anode_sf)
        self.solve_temp_bulk(self.electrolyte.vars["T"], self.electrolyte.vars["dTdx"], self.electrolyte)
        self.solve_temp_surface(self.electrolyte, self.cathode, self.cathode_sf)
        self.solve_temp_bulk(self.cathode.vars["T"], self.cathode.vars["dTdx"], self.cathode)
        
    def opt_inital_guess(self, dTdx_guess):
        """
        function to iteratively optimize the guess for the temperature gradient on the left-hand side

        Parameters
        ----------
        dTdx_guess : float
            guess of temperature gradient 

        Returns
        -------
        list
            residue of the temperature on the right-hand-side for the inital guess of the gradient and the specified boundary condtion 
        """
        self.solve_temp_system(dTdx_guess[0])
        return [self.bc["rbc"] - self.cathode.vars["T"][-1]]
        
#%% submodel
class Submodel:
    def  __init__(self, model, name):
        self.params = model.parameter[name]
        self.name = name
        self.vars = {}
    
    def init_mesh(self, init_x, nx):
        """
        initalise mesh for submodel by specifing the inital x value and the number of nodes

        Parameters
        ----------
        init_x : float
            inital x-value of x-vector
        nx : int
            number of nodes
        """
        self.vars["x"] = np.linspace(init_x, init_x + self.params["length"], nx)

#%%% surface    
class Surface(Submodel):
    def __init__(self, params, name):
        super().__init__(params, name)
        self.__def_params()
    
    def __overpotential(self, j0):
        """
        function to determine overpotential of electrode reaction

        Parameters
        ----------
        j0 : float
            exchange current density of electrode

        Returns
        -------
        float
            returns the overpotential
        """
        return 2*R*Tamb / F * np.log10(j/j0)
        
    def __def_params(self):
        """
        define more parameters that can be calculated from inital parameters
        """
        L = self.params["length"]
        OCP = self.params["OCP"]
        j0 = self.params["exchange current density"]
        lambda_ = self.params["thermal conductivity"]
        k = self.params["correction factor"]

        self.params["gibbs energy"] = - OCP * F
        self.params["overpotential"] = self.__overpotential(j0)
        self.params["thermal conductivity"] = lambda_ / (k*L) 

#%%% electrode
class Electrode(Submodel):
    def __init__(self, params, name, mass_trans = True):
        super().__init__(params, name)
        self.mass_trans = mass_trans
        self.__def_params()
    
    def __def_params(self):
        """
        includes the option to neglect mass transport by setting the diffusion coefficent to zero
        """
        if not self.mass_trans:
            self.params["diffusion coefficient"] = 0
    
    def temp_function(self, x, S):
        """
        input function for solve_ivp()
        S is the vector containing the temperature T and its gradient dTdx
        returned are the derivative of T (dTdx) and of dTdx (d2Tdx2)
        
        splits the second order ode into a system of two first order odes
        
        Parameters
        ----------
        x : float
            spatial vector
        S : tuple
            vector of the first-order ode system
            
        Returns
        -------
        list
            derivatives of each equation
            T    -> dTdx
            dTdx -> d2Tdx2
        """
        pi = self.params["peltier coefficient"]
        lambda_ = self.params["thermal conductivity"]
        kappa = self.params["electric conductivity"]

        T, dTdx = S
        rhs     = - pi*j / (lambda_*F*T) * dTdx - j**2 / (lambda_*kappa)
        
        return[dTdx, rhs]

#%%% electrolyte
class Electrolyte(Submodel):
    def __init__(self, params, name, mass_trans = True):
        super().__init__(params, name)
        self.mass_trans = mass_trans
        self.__def_params()
    
    def __def_params(self):
        """
        define new parameters for better readiblity of the temperature equation
        """
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
    
    def temp_function(self, x, S):
        """
        input function for solve_ivp()
        S is the vector containing the temperature T and its gradient dTdx
        returned are the derivative of T (dTdx) and of dTdx (d2Tdx2)
        
        splits the second order ode into a system of two first order odes
        
        Parameters
        ----------
        x : float
            spatial vector
        S : tuple
            vector of the first-order ode system
            
        Returns
        -------
        list
            derivatives of each equation
            T    -> dTdx
            dTdx -> d2Tdx2
        """
        a_q = self.params["a_q"]
        a_phi = self.params["a_phi"]
        b_phi = self.params["b_phi"]
        lambda_ = self.params["thermal conductivity"]
        kappa = self.params["electric conductivity"]
        
        T, dTdx = S
        denom = (lambda_ - a_q/T**2)
        rhs = -j*a_phi/(T * F) / denom * dTdx - b_phi*T*j**2/F**2 / denom - j**2/kappa / denom
        
        return [dTdx, rhs]     
#%% main
model = LiionModel(params_LCO)  
model.init_mesh({"Anode":       100, 
                 "Electrolyte":  20,
                 "Cathode":     100}) 
model.boundary_conditions(Tamb, Tamb)
model.solve()
        
        