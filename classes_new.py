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
from matplotlib.lines import Line2D

from params_sys import j, F, Tamb, R
import params_LCO

#%% cell model
class LiionModel:
#%%% private methods
    def __init__(self, param_set, mass_trans = True):
        self.parameter = param_set.parameter
        self.mass_trans = mass_trans
        self.__add_submodels()

    def __add_submodels(self):
        """
        creates an instance of each submodel in the current LiionModel instance
        """
        self.anode       = Electrode(self, "Anode", mass_trans=self.mass_trans)
        self.anode_sf    = Surface(self, "Anode Surface")
        self.electrolyte = Electrolyte(self, "Electrolyte", mass_trans=self.mass_trans)
        self.cathode_sf  = Surface(self, "Cathode Surface")
        self.cathode       = Electrode(self, "Cathode", mass_trans=self.mass_trans)

    def __solve_temp_surface(self, i, o, sf):
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
        
        sf.vars["T"], sf.vars["dT_is"], sf.vars["dT_so"]  = T_s, dT_is, dT_so
        o.vars["T"], o.vars["dTdx"] = T_oi, dTdx_oi

    def __solve_temp_bulk(self, T0, dTdx0, bulk):
        """
        solves initial value problem

        Parameters
        ----------
        T0 : float
            initial Temperature, in this case on the left-hand-side
        dTdx0 : float
            initial Temperature gradient, in this case on the left-hand-side
        bulk : object
            submodel of corresponding bulk phase
        """
        S0 = (T0, dTdx0)
        x = bulk.vars["x"]
        sol = solve_ivp(bulk.temp_function, (x[0], x[-1]), S0, t_eval=x)
        bulk.vars["T"], bulk.vars["dTdx"] = sol.y
        
    def __solve_temp_system(self, dTdx_guess):
        """
        solves system of equations for temperature and the gradient in each phase
        by using the defined boundary conditions on the left-hand-side and an initial guess for the temperature gradient

        Parameters
        ----------
        dTdx_guess : float
            guess of the temperature gradient at the left boundary
        """
        self.__solve_temp_bulk(self.bc["lbc"], dTdx_guess, self.anode)
        self.__solve_temp_surface(self.anode, self.electrolyte, self.anode_sf)
        self.__solve_temp_bulk(self.electrolyte.vars["T"], self.electrolyte.vars["dTdx"], self.electrolyte)
        self.__solve_temp_surface(self.electrolyte, self.cathode, self.cathode_sf)
        self.__solve_temp_bulk(self.cathode.vars["T"], self.cathode.vars["dTdx"], self.cathode)
        
    def __opt_initial_guess(self, dTdx_guess):
        """
        function to iteratively optimize the guess for the temperature gradient on the left-hand side

        Parameters
        ----------
        dTdx_guess : float
            guess of temperature gradient 

        Returns
        -------
        list
            residue of the temperature on the right-hand-side for the initial guess of the gradient and the specified boundary condtion 
        """
        self.__solve_temp_system(dTdx_guess[0])
        return [self.bc["rbc"] - self.cathode.vars["T"][-1]]
    
    def __calc_Jq(self):
        """
        call functions to calculate measurable heatflux for each domain
        """
        self.anode.vars["Jq"] = self.anode.Jq()
        self.electrolyte.vars["Jq"] = self.electrolyte.Jq()
        self.cathode.vars["Jq"] = self.cathode.Jq()
        
    def __calc_phi(self):
        """
        call functions to calculate dphidx and phi for each domain
        """
        self.anode.vars["dphidx"]       = self.anode.dphidx()
        self.electrolyte.vars["dphidx"] = self.electrolyte.dphidx()
        self.cathode.vars["dphidx"]     = self.cathode.dphidx()
        self.anode_sf.vars["dphi"]      = self.anode_sf.dphi(self.anode, self.electrolyte)
        self.cathode_sf.vars["dphi"]    = self.cathode_sf.dphi(self.electrolyte, self.cathode)
        
        self.anode.vars["phi"] = self.anode.integrate("dphidx")
        self.electrolyte.vars["phi"] = self.electrolyte.integrate("dphidx", y0 = (self.anode.vars["phi"][-1] + self.anode_sf.vars["dphi"]))
        self.cathode.vars["phi"] = self.cathode.integrate("dphidx", y0 = (self.electrolyte.vars["phi"][-1] + self.cathode_sf.vars["dphi"]))
        
    def __calc_mu_c(self):
        """
        call functions to calculate chemical potenital (mu) and concentration (c)
        done in one function because in electrode c is calculated first and then mu
        and in electrolyte the other way around
        """
        self.anode.vars["dcdx"]   = self.anode.dcdx()
        self.cathode.vars["dcdx"] = self.cathode.dcdx()
        
        self.anode.vars["c"]   = self.anode.integrate("dcdx", y0 = self.anode.params["initial concentration"])
        self.cathode.vars["c"] = self.cathode.integrate("dcdx", y0 = self.cathode.params["initial concentration"])
        
        self.anode.vars["dmudx"]   = self.anode.dmudx()
        self.cathode.vars["dmudx"] = self.cathode.dmudx()
        
        self.electrolyte.vars["dmuLdx"] = self.electrolyte.dmudx("L")
        self.electrolyte.vars["dmuDdx"] = self.electrolyte.dmudx("D")
        self.electrolyte.vars["c"] = self.electrolyte.c()
        self.electrolyte.vars["dcdx"] = self.electrolyte.dcdx()
        
    def __calc_J_i(self):
        """
        call functions to calculate Lithium flux in both electrodes
        """
        self.anode.vars["J_L"] = self.anode.J_i()
        self.cathode.vars["J_L"] = self.cathode.J_i()
        
    def __calc_sigma(self):
        """
        call functions to calculate local entropy production in each domain
        and sum to get accumulated entropy production along cell
        """
        self.anode.vars["sigma"] = self.anode.sigma()
        self.electrolyte.vars["sigma"] = self.electrolyte.sigma()
        self.cathode.vars["sigma"] = self.cathode.sigma()
        
        self.anode_sf.vars["sigma"] = self.anode_sf.sigma(self.anode, self.electrolyte)
        self.cathode_sf.vars["sigma"] = self.cathode_sf.sigma(self.electrolyte, self.cathode)
        
        self.anode.vars["sigma accumulated"] = np.cumsum(self.anode.vars["sigma"])
        self.electrolyte.vars["sigma accumulated"] = np.cumsum(self.electrolyte.vars["sigma"]) + self.anode.vars["sigma accumulated"][-1] + self.anode_sf.vars["sigma"]
        self.cathode.vars["sigma accumulated"] = np.cumsum(self.cathode.vars["sigma"])+  self.electrolyte.vars["sigma accumulated"][-1] + self.cathode_sf.vars["sigma"]

#%%% public methods
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
        dTdx0, = fsolve(self.__opt_initial_guess, [1])
        self.__solve_temp_system(dTdx0)
        
        self.__calc_Jq()
        self.__calc_phi()
        self.__calc_mu_c()
        self.__calc_J_i()
        self.__calc_sigma()
        
    
    def plot(self):
        """
        plots Temperature, Potential, Concentration, Heat Flux and Entropy production

        Returns
        -------
        None.

        """

        fig, ((T, phi, c), (Jq, sigma, sigma_ac)) = plt.subplots(2, 3, figsize=(14, 7), layout="constrained", dpi=400)
        for ax in (T, phi, c, Jq, sigma, sigma_ac):
            # Vertical spans 
            ax.axvspan(0, self.anode.vars["x"][-1]*10**(6), facecolor='b', alpha=0.1)
            ax.axvspan(self.cathode.vars["x"][0]*10**(6), self.cathode.vars["x"][-1]*10**(6), facecolor='r', alpha=0.1)
            ax.axvspan(self.anode_sf.vars["x"][0]*10**(6), self.anode_sf.vars["x"][-1]*10**(6), facecolor='k', alpha=0.3)
            ax.axvspan(self.cathode_sf.vars["x"][0]*10**(6), self.cathode_sf.vars["x"][-1]*10**(6), facecolor='k', alpha=0.3)
            
            temp_data, phi_data, sigma_ac_data = {"x": np.array([]), "T": np.array([])}, {"x": np.array([]), "phi": np.array([])}, {"x": np.array([]), "sigma_ac": np.array([])}
            
            for model in [self.anode, self.anode_sf, self.electrolyte, self.cathode_sf, self.cathode]:
                bulk = [self.anode, self.electrolyte, self.cathode]
                if model in bulk:
                    temp_data["x"]  = np.append(temp_data["x"], model.vars["x"] * 10**6)
                    temp_data["T"]  = np.append(temp_data["T"], model.vars["T"])
                    
                    phi_data["x"]   = np.append(phi_data["x"], model.vars["x"] * 10**6)
                    phi_data["phi"] = np.append(phi_data["phi"], model.vars["phi"])
                    
                    sigma_ac_data["x"] = np.append(sigma_ac_data["x"], model.vars["x"] * 10**6)
                    sigma_ac_data["sigma_ac"] = np.append(sigma_ac_data["sigma_ac"], model.vars["sigma accumulated"])
                    
                    # plot c, Jq and sigma for each bulk phase separate
                    Jq.plot(model.vars["x"]*10**6, model.vars["Jq"], color="r", linewidth=2)
                    sigma.plot(model.vars["x"]*10**6, model.vars["sigma"], color="r", linewidth=2)
                    
                    # plot c in different color
                    if model == self.electrolyte:
                        c.plot(model.vars["x"]*10**6, model.vars["c"], color="b", linewidth=2)
                    else:
                        c.plot(model.vars["x"]*10**6, model.vars["c"], color="r", linewidth=2)
                else:
                    temp_data["x"] = np.append(temp_data["x"], model.vars["x"] * 10**6) 
                    temp_data["T"] = np.append(temp_data["T"], np.ones(2) * model.vars["T"])
                    
                    # plot sigma surface as dot
                    sigma.plot(np.mean(model.vars["x"])*10**6, model.vars["sigma"], color="r", linewidth=2, marker="*")
               
            # T, phi and sigma accumulated are plotted as one line along whole cell
            T.plot(temp_data["x"], temp_data["T"], color="r", linewidth=2)
            phi.plot(phi_data["x"], phi_data["phi"], color="r", linewidth=2)
            sigma_ac.plot(sigma_ac_data["x"], sigma_ac_data["sigma_ac"], color="r", linewidth=2)
            
            # format x-axes
            ax.set_xlabel(' x / ${\mu m}$', fontsize=12)
            ax.set_xlim(self.anode.vars["x"][0]*10**(6), self.cathode.vars["x"][-1]*10**(6))
            
            # format temperature plot
            T.set_ylabel("T / $K$")
            T.set_title("Temperature profile", fontsize=13)
            y_ticks = T.get_yticks()
            T.set_yticks(y_ticks)
            T.set_yticklabels([f"{tick:.4f}" for tick in y_ticks])
            
            # format electric potential plot
            phi.set_ylabel("$\phi$ / $V$")
            phi.set_title("Potential profile", fontsize=13)
            
            # format concentration plot
            c.set_ylabel("c / $mol m^{-3}$")
            c.set_title("Concentration profile", fontsize=13)
            lines = (Line2D([0], [0], color = "r", linestyle="-"), Line2D([0], [0], color = "b", linestyle="-"))
            labels = ("Li", "LiPF$_6$")
            c.legend(lines, labels)
            
            # format heatflux plot
            Jq.set_ylabel("J'$_q$ / $W m^{-2}$")
            Jq.set_title("Measurable heat flux", fontsize=13)
            
            # format local entropy plot
            sigma.set_ylabel("$\sigma$ / $Wm^{-2}K^{-1}$")
            sigma.set_title("Local entropy production", fontsize=13)
            
            # format accumulated entropy plot
            sigma_ac.set_ylabel("$\sigma$ / $Wm^{-2}K^{-1}$")
            sigma_ac.set_title("Accumulated entropy production", fontsize=13)
            
    
    def plot_single(self):
        """
        plot properties
        """
        fig, ax = plt.subplots(dpi=200)
        
        # Vertical spans 
        ax.axvspan(0, self.anode.vars["x"][-1]*10**(6), facecolor='b', alpha=0.1)
        ax.axvspan(self.cathode.vars["x"][0]*10**(6), self.cathode.vars["x"][-1]*10**(6), facecolor='r', alpha=0.1)
        ax.axvspan(self.anode_sf.vars["x"][0]*10**(6), self.anode_sf.vars["x"][-1]*10**(6), facecolor='k', alpha=0.3)
        ax.axvspan(self.cathode_sf.vars["x"][0]*10**(6), self.cathode_sf.vars["x"][-1]*10**(6), facecolor='k', alpha=0.3)
        
        for model in [self.anode, self.electrolyte, self.cathode]:
            ax.plot(model.vars["x"]*10**6, model.vars["T"], color="r", linewidth=2)
        
        for model in [self.anode_sf, self.cathode_sf]:
            ax.plot(model.vars["x"]*10**6, np.ones(2)*model.vars["T"], color="r", linewidth=2)
        
        ax.set_title("Temperature profile", fontsize=14)
        ax.set_xlim(self.anode.vars["x"][0]*10**(6), self.cathode.vars["x"][-1]*10**(6))
        
        y_ticks = ax.get_yticks()
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:.4f}" for tick in y_ticks])
        
        ax.set_xlabel(' x / ${\mu m}$', fontsize=12)
        ax.set_ylabel("Temperature / K", fontsize=12)
    

#%% submodel
class Submodel:
    def  __init__(self, model, name, mass_trans = True):
        self.mass_trans = mass_trans
        self.params = model.parameter[name]
        self.name = name
        self.vars = {}
    
    def init_mesh(self, init_x, nx):
        """
        initialise mesh for submodel by specifing the initial x value and the number of nodes

        Parameters
        ----------
        init_x : float
            initial x-value of x-vector
        nx : int
            number of nodes
        """
        self.vars["x"] = np.linspace(init_x, init_x + self.params["length"], nx)
        
    def integrate(self, dydx, y0=0):
        """
        integrate dydx along x-axes

        Parameters
        ----------
        dydx : string
            name of the variable
        y0 : float, optional
            initial value where integration starts The default is 0.
        Returns
        -------
        y : float
            integrated value y
        """
        dx = np.gradient(self.vars["x"])
        
        return np.cumsum(self.vars[dydx]*dx) + y0
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
        define more parameters that can be calculated from initial parameters
        """
        L = self.params["length"]
        OCP = self.params["OCP"]
        j0 = self.params["exchange current density"]
        lambda_ = self.params["thermal conductivity"]
        k = self.params["correction factor"]

        self.params["gibbs energy"] = - OCP * F
        self.params["overpotential"] = self.__overpotential(j0)
        self.params["thermal conductivity"] = lambda_ / (k*L) 
        
    def dphi(self, i, o):
        """
        calculates potential jump accross surface

        Parameters
        ----------
        i : object
            instance of the bulk phase on the left-hand-side
        o : object
            instance of the bulk phase on the right-hand-side

        Returns
        -------
        dphi : float
        """
        pi_i, pi_o = i.params["peltier coefficient"], o.params["peltier coefficient"]
        eta = self.params["overpotential"]
        dG = self.params["gibbs energy"]
        
        dT_is, dT_so = self.vars["dT_is"], self.vars["dT_so"] 
        T_io, T_oi = i.vars["T"][-1], o.vars["T"][0]
        
        return -pi_i/(T_io*F) * dT_is - pi_o/(T_oi*F) * dT_so - eta - dG/F
    
    def sigma(self, i, o):
        """
        calculates entropy production accross surface

        Parameters
        ----------
        i : object
            instance of the bulk phase on the left-hand-side
        o : object
            instance of the bulk phase on the right-hand-side

        Returns
        -------
        sigma : float
        """
        Jq_io, Jq_oi = i.vars["Jq"][-1], o.vars["Jq"][0]
        T_io, T_oi = i.vars["T"][-1], o.vars["T"][0]
        dT_is, dT_so, T_s = self.vars["dT_is"], self.vars["dT_so"], self.vars["T"]
        dphi = self.vars["dphi"]
        dG = self.params["gibbs energy"]
        
        return -Jq_io/T_io * dT_is - Jq_oi/T_oi * dT_so - j/T_s*(dphi + dG/F)

#%%% electrode
class Electrode(Submodel):
    def __init__(self, params, name, mass_trans):
        super().__init__(params, name, mass_trans)
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
    
    def Jq(self):
        """
        calculates heat flux

        Returns
        -------
        Jq : np.array()
            heat flux.
        """
        pi = self.params["peltier coefficient"]
        lambda_ = self.params["thermal conductivity"]
        
        dTdx = self.vars["dTdx"]
        
        return -lambda_ * dTdx + pi*j/F
    
    def J_i(self):
        """
        calcualte molar flux

        Returns
        -------
        J_i : np.array()
        """
        D = self.params["diffusion coefficient"]
        dcdx = self.vars["dcdx"]
        
        return - D*dcdx
    
    def dphidx(self):
        """
        calcualtes dphidx

        Returns
        -------
        dphidx : np.array()
        """
        pi = self.params["peltier coefficient"]
        kappa = self.params["electric conductivity"]
        
        T = self.vars["T"]
        dTdx = self.vars["dTdx"]
        
        return pi/F * dTdx/T + j/kappa
    
    def dcdx(self):
        """
        calcualtes dcdx

        Returns
        -------
        dcdx : np.array()
        """
        D = self.params["diffusion coefficient"]
        
        return - j/(D*F)
    
    def dmudx(self):
        """
        calculates dmudx

        Returns
        -------
        dmudx : np.array()
        """
        tdf = self.params["thermodynamic factor"]
        
        T = self.vars["T"]
        c = self.vars["c"]
        dcdx = self.vars["dcdx"]
        
        return tdf * R*T/c * dcdx
    
    def sigma(self):
        """
        calculate local entropy production

        Returns
        -------
        sigma : np.array()
        """
        T      = self.vars["T"]
        dTdx   = self.vars["T"]
        Jq     = self.vars["Jq"]
        J_L    = self.vars["J_L"]
        dmudx  = self.vars["dmudx"]
        dphidx = self.vars["dphidx"]
        dx     = np.gradient(self.vars["x"])
        
        return (- dTdx/T**2 * Jq - dmudx/T * J_L - dphidx/T * j)*dx

#%%% electrolyte
class Electrolyte(Submodel):
    def __init__(self, params, name, mass_trans):
        super().__init__(params, name, mass_trans)
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
    
    def Jq(self):
        """
        calculate and return measurable heat flux

        Returns
        -------
        Jq : np.array()
            heatf lux
        """
        a_q = self.params["a_q"]
        b_q = self.params["b_q"]
        lambda_ = self.params["thermal conductivity"]
        
        T = self.vars["T"]
        dTdx = self.vars["dTdx"]
        
        return -(lambda_ - a_q/T**2)*dTdx + j*b_q/F
    
    def dphidx(self):
        """
        calcualte dphidx

        Returns
        -------
        dphidx : np.array()
        """
        a_phi = self.params["a_phi"]
        b_phi = self.params["b_phi"]
        kappa = self.params["electric conductivity"]
        
        T = self.vars["T"]
        dTdx = self.vars["dTdx"]
        
        return -a_phi/(T*F) * dTdx - b_phi*j/F**2 * T - j/kappa
    
    def __conc_function(self, x, c):
        """
        serves as input function for ivp_solve()
        
        Parameters
        ----------
        x : np.array()
            spatial mesh
        c : np.array()
            concentration along x

        Returns
        -------
        dcdx : np.array()
            returns function dcdx of c and x.
        """
        tdf_LL = self.params["thermodynamic factor LL"]
        tdf_DD = self.params["thermodynamic factor DD"]
        tdf_LD = self.params["thermodynamic factor LD"]
        tdf_DL = self.params["thermodynamic factor DL"]
        
        T = np.mean(self.vars["T"])
        dmuLdx = np.mean(self.vars["dmuLdx"])
        dmuDdx = np.mean(self.vars["dmuDdx"])
        
        return (dmuLdx - tdf_LD/tdf_DD*dmuDdx)/(tdf_LL - tdf_LD*tdf_DL/tdf_DD) * c/(R*T)
    
    def c(self):
        """
        solves for c and dcdx

        Returns
        -------
        c : np.array()
            concentration along x.
        dcdx : np.array()
        """
        x = self.vars["x"]
        c0 = self.params["initial concentration L"]
        
        sol = solve_ivp(self.__conc_function, (x[0], x[-1]), [c0], t_eval=x)
        c = sol.y[0]
        
        return c
    
    def dcdx(self):
        """
        cacluates dcdx

        Returns
        -------
        dcdx : np.array()
        """
        dc = np.gradient(self.vars["c"])
        dx = np.gradient(self.vars["x"])
        
        return dc/dx
    
    def dmudx(self, component):
        """
        calculates dmudx for selected component in electrolyte

        Parameters
        ----------
        component : string
            can be for the electrolyte "L" or "D" 
            "L" : Lithium, "D" : DEC

        Returns
        -------
        dmudx : np.array()
        """
        a = self.params[f"a_{component}"]
        b = self.params[f"b_{component}"]
        
        T = self.vars["T"]
        dTdx = self.vars["dTdx"]
        
        return a/T * dTdx - b*j/F * T
    
    def sigma(self):
        """
        calculate local entropy production

        Returns
        -------
        sigma : np.array()
        """
        T      = self.vars["T"]
        dTdx   = self.vars["T"]
        Jq     = self.vars["Jq"]
        dphidx = self.vars["dphidx"]
        dx     = np.gradient(self.vars["x"])
        
        return (- dTdx/T**2 * Jq - dphidx/T * j)*dx
    
#%% main
model = LiionModel(params_LCO)  
model.init_mesh({"Anode":       100, 
                 "Electrolyte":  20,
                 "Cathode":     100}) 
model.boundary_conditions(Tamb, Tamb)
model.solve()
model.plot()

        
        