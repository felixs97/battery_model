#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:19:31 2023

@author: felix
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from params_sys import j, F, Tamb, R
import params_LCO, params_LFP

#%% cell model
class LiionModel:
#%%% private methods
    def __init__(self, name, param_set):
        self.parameter = param_set.parameter
        self.name = name
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
        
        if sf.name == "Anode Surface":
            dT_is = lambda_i/lambda_s * dTdx_io
            #dT_so = dT_is + pi_s*j/F/lambda_s - eta*j/lambda_s
            dT_so = ((lambda_s - pi_i*j/(F*T_io))*dT_is + pi_s*j/F - eta*j) / (pi_o*j/(F*T_io) + lambda_s)
            dTdx_oi = (lambda_s * dT_so + j/F*(b_q - pi_o))/(lambda_o - a_q/T_io**2)
        else:
            dT_is = ((lambda_i - a_q/T_io**2)*dTdx_io + j/F*(pi_i - b_q))/lambda_s
            #dT_so = dT_is + pi_s*j/(lambda_s*F) - eta*j/lambda_s
            dT_so = ((lambda_s - pi_i*j/(F*T_io))*dT_is + pi_s*j/F - eta*j) / (pi_o*j/(F*T_io) + lambda_s)
            dTdx_oi = lambda_s/lambda_o * dT_so

        
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
        sol = solve_ivp(bulk.temp_function, (x[0], x[-1]), S0, t_eval=x, rtol=10**(-9), atol=10**(-9))
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
        
        self.electrolyte.vars["cL"] = self.electrolyte.c("L")
        self.electrolyte.vars["dcLdx"] = self.electrolyte.gradient("cL")
        self.electrolyte.vars["cD"] = self.electrolyte.c("D")
        self.electrolyte.vars["dcDdx"] = self.electrolyte.gradient("cD")
        self.electrolyte.vars["cE"] = np.ones(len(self.electrolyte.vars["x"]))*(self.electrolyte.params["initial concentration E"])
        
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
        
        self.anode.vars["sigma accumulated"] = self.anode.integrate("sigma", 0)
        self.electrolyte.vars["sigma accumulated"] = self.electrolyte.integrate("sigma", (self.anode.vars["sigma accumulated"][-1] + self.anode_sf.vars["sigma"]))
        self.cathode.vars["sigma accumulated"] = self.cathode.integrate("sigma", (self.electrolyte.vars["sigma accumulated"][-1] + self.cathode_sf.vars["sigma"]))

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
        
    def consistency_check(self, show_subsystems=False):
        """
        calculate and print entropy for each phase and for whole cell in two different ways to check consistency
        both calulations should result in the same 
        """

        write_out = 'consistency_check.txt'
        
        S_ae = self.anode_sf.partial_molar_entropy(self.anode, self.electrolyte)
        S_ce = self.cathode_sf.partial_molar_entropy(self.electrolyte, self.cathode)
        S_ao = self.anode.partial_molar_entropy(S_ae)
        S_co = self.cathode.partial_molar_entropy(S_ce)
        
        Js_ao = self.anode.vars["Jq"][0]/self.anode.vars["T"][0] + self.anode.vars["J_L"][0]*S_ao 
        Js_co = self.cathode.vars["Jq"][-1]/self.cathode.vars["T"][-1] + self.cathode.vars["J_L"][-1]*S_co
    
        dJs, sigma = Js_co - Js_ao, self.cathode.vars["sigma accumulated"][-1]
        print("*************** Consistency Check ***************\n")
        
        print(f"Entropy fluxes difference:       {dJs:.5f} W/m2/K")
        print(f"Entropy production accumulated:  {sigma:.5f} W/m2/K")
        print("\n")
        
        with open(write_out, "w") as o:
            o.write("*************** Consistency Check ***************\n\n")
            o.write(f"Entropy fluxes difference:       {dJs:.5f} W/m2/K\n")
            o.write(f"Entropy production accumulated:  {sigma:.5f} W/m2/K\n")
            o.write("\n\n")

        if show_subsystems:
            Js_ae = self.anode.vars["Jq"][-1]/self.anode.vars["T"][-1] + self.anode.vars["J_L"][-1]*S_ae
            Js_ea = self.electrolyte.vars["Jq"][0]/self.electrolyte.vars["T"][0]
            Js_ec = self.electrolyte.vars["Jq"][-1]/self.electrolyte.vars["T"][-1]
            Js_ce = self.cathode.vars["Jq"][0]/self.cathode.vars["T"][0] + self.cathode.vars["J_L"][0]*S_ce
            
            dJs_anode = Js_ae - Js_ao
            dJs_anode_sf = Js_ea - Js_ae
            dJs_electrolyte = Js_ec - Js_ea
            dJs_cathode_sf = Js_ce - Js_ec
            dJs_cathode = Js_co - Js_ce
            
            sigma_anode = self.anode.integrate("sigma")
            sigma_electrolyte = self.electrolyte.integrate("sigma")
            sigma_cathode = self.cathode.integrate("sigma")
            sigma_anode_sf = self.anode_sf.vars["sigma"]
            sigma_cathode_sf = self.cathode_sf.vars["sigma"]
            
            print("Anode")
            print(f"Entropy fluxes difference:       {dJs_anode:.9f} W/m2/K")
            print(f"Entropy production:              {sigma_anode[-1]:.9f} W/m2/K")
            print("\n")
            
            print("Anode Surface")
            print(f"Entropy fluxes difference:       {dJs_anode_sf:.9f} W/m2/K")
            print(f"Entropy production:              {sigma_anode_sf:.9f} W/m2/K")
            print("\n")
            
            print("Electrolyte")
            print(f"Entropy fluxes difference:       {dJs_electrolyte:.9f} W/m2/K")
            print(f"Entropy production:              {sigma_electrolyte[-1]:.9f} W/m2/K")
            print("\n")
            
            print("Cathode Surface")
            print(f"Entropy fluxes difference:       {dJs_cathode_sf:.9f} W/m2/K")
            print(f"Entropy production               {sigma_cathode_sf:.9f} W/m2/K")
            print("\n")
            
            print("Cathode")
            print(f"Entropy fluxes difference:       {dJs_cathode:.9f} W/m2/K")
            print(f"Entropy production:              {sigma_cathode[-1]:.9f} W/m2/K")
            print("\n")
        
            print("Partial Molar Entropies where determined, using the Peltier heat, Peltier coefficient and the local temperatures")
            print(f"Partial Molar Entropy of Anode Surface:              {S_ae:.1f} J/mol/K")
            print(f"Partial Molar Entropy of Anode on left-hand-side:    {S_ao:.1f} J/mol/K")
            print(f"Partial Molar Entropy of Cathode Surface:            {S_ce:.1f} J/mol/K")
            print(f"Partial Molar Entropy of Cathode on right-hand-side: {S_co:.1f} J/mol/K")
        
            with open(write_out, "a") as o:
                o.write("Anode\n")
                o.write(f"Entropy fluxes difference:       {dJs_anode:.9f} W/m2/K\n")
                o.write(f"Entropy production:              {sigma_anode[-1]:.9f} W/m2/K\n")
                o.write("\n\n")
                
                o.write("Anode Surface\n")
                o.write(f"Entropy fluxes difference:       {dJs_anode_sf:.9f} W/m2/K\n")
                o.write(f"Entropy production:              {sigma_anode_sf:.9f} W/m2/K\n")
                o.write("\n\n")
                
                o.write("Electrolyte\n")
                o.write(f"Entropy fluxes difference:       {dJs_electrolyte:.9f} W/m2/K\n")
                o.write(f"Entropy production:              {sigma_electrolyte[-1]:.9f} W/m2/K\n")
                o.write("\n\n")
                
                o.write("Cathode Surface\n")
                o.write(f"Entropy fluxes difference:       {dJs_cathode_sf:.9f} W/m2/K\n")
                o.write(f"Entropy production               {sigma_cathode_sf:.9f} W/m2/K\n")
                o.write("\n\n")
                
                o.write("Cathode\n")
                o.write(f"Entropy fluxes difference:       {dJs_cathode:.9f} W/m2/K\n")
                o.write(f"Entropy production:              {sigma_cathode[-1]:.9f} W/m2/K\n")
                o.write("\n\n")
        
                o.write("Partial Molar Entropies where determined, using the Peltier heat, Peltier coefficient and the local temperatures\n")
                o.write(f"Partial Molar Entropy of Anode Surface:              {S_ae:.1f} J/mol/K\n")
                o.write(f"Partial Molar Entropy of Anode on left-hand-side:    {S_ao:.1f} J/mol/K\n")
                o.write(f"Partial Molar Entropy of Cathode Surface:            {S_ce:.1f} J/mol/K\n")
                o.write(f"Partial Molar Entropy of Cathode on right-hand-side: {S_co:.1f} J/mol/K\n")
  
    def plot(self):
        """
        plots Temperature, Potential, Concentration, Heat Flux and Entropy production

        Returns
        -------
        None.

        """

        fig, ((T, phi, c), (Jq, sigma, sigma_ac)) = plt.subplots(2, 3, figsize=(14, 7), layout="constrained", dpi=100)
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
                    #sigma.plot(model.vars["x"]*10**6, model.vars["dJsdx"], color="b", linewidth=2, linestyle = "--")
                    # plot c in different color
                    if model == self.electrolyte:
                        c.plot(model.vars["x"]*10**6, model.vars["cL"], color="b", linewidth=2)
                        c.plot(model.vars["x"]*10**6, model.vars["cD"], color="g", linewidth=2)
                        c.plot(model.vars["x"]*10**6, model.vars["cE"], color="orange", linewidth=2)
                    else:
                        c.plot(model.vars["x"]*10**6, model.vars["c"], color="r", linewidth=2)
                else:
                    temp_data["x"] = np.append(temp_data["x"], model.vars["x"] * 10**6) 
                    temp_data["T"] = np.append(temp_data["T"], np.ones(2) * model.vars["T"])
               
            # T, phi and sigma accumulated are plotted as one line along whole cell
            T.plot(temp_data["x"], temp_data["T"], color="r", linewidth=2)
            phi.plot(phi_data["x"], phi_data["phi"], color="r", linewidth=2)
            sigma_ac.plot(sigma_ac_data["x"], sigma_ac_data["sigma_ac"], color="r", linewidth=2)
            
            # format x-axes
            ax.set_xlabel(r' $x$ / $\mu$m', fontsize=12)
            ax.set_xlim(self.anode.vars["x"][0]*10**(6), self.cathode.vars["x"][-1]*10**(6))
            
            # format temperature plot
            T.set_ylabel("$T$ / K")
            T.set_title("Temperature profile", fontsize=13)
            y_ticks = T.get_yticks()
            T.set_yticks(y_ticks)
            T.set_yticklabels([f"{tick:.4f}" for tick in y_ticks])
            
            # format electric potential plot
            phi.set_ylabel(r"$\phi$ / V")
            phi.set_title("Potential profile", fontsize=13)
            
            # format concentration plot
            c.set_ylabel("$c$ / mol m$^{-3}$")
            c.set_title("Concentration profile", fontsize=13)
            lines = (Line2D([0], [0], color = "r", linestyle="-"), Line2D([0], [0], color = "b", linestyle="-"), Line2D([0], [0], color = "g", linestyle="-"), Line2D([0], [0], color = "orange", linestyle="-"))
            labels = ("Li", "LiPF$_6$", "DEC", "EC")
            c.legend(lines, labels)
            
            # format heatflux plot
            Jq.set_ylabel("$J'_q$ / W m$^{-2}$")
            Jq.set_title("Measurable heat flux", fontsize=13)
            
            # format local entropy plot
            sigma.set_ylabel(r"$\sigma$ / W m$^{-3}$ K$^{-1}$")
            sigma.set_title("Local entropy production", fontsize=13)
            
            # format accumulated entropy plot
            sigma_ac.set_ylabel(r"$\sigma$ / W m$^{-2}$ K$^{-1}$")
            sigma_ac.set_title("Accumulated entropy production", fontsize=13)

        fig.savefig('all_results.png')
        plt.show()
            
    
    def plot_single(self, quantity):
        """
        plot a single selected quantity
        possible quantities: T, phi, c, Jq, sigma, sigma accumulated

        Parameters
        ----------
        quantity : string
            quantity to plot, select from list above
        """
        fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
        #fig, ax = plt.subplots(figsize=(3, 4), dpi=200)

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
                if quantity == "Jq":
                    ax.plot(model.vars["x"]*10**6, model.vars["Jq"], color="r", linewidth=2)
                
                elif quantity == "sigma":
                    ax.plot(model.vars["x"]*10**6, model.vars["sigma"], color="r", linewidth=2)
                
                elif quantity == "c":
                    # plot c in different color
                    if model == self.electrolyte:
                        ax.plot(model.vars["x"]*10**6, model.vars["cL"], color="b", linewidth=2)
                        ax.plot(model.vars["x"]*10**6, model.vars["cD"], color="g", linewidth=2)
                        ax.plot(model.vars["x"]*10**6, model.vars["cE"], color="orange", linewidth=2)
                    else:
                        ax.plot(model.vars["x"]*10**6, model.vars["c"], color="r", linewidth=2)
                
                elif quantity == "dmu" and model == self.electrolyte:
                    ax.plot(model.vars["x"]*10**6, model.integrate("dmuLdx"), color="b", linewidth=2)
                    ax.plot(model.vars["x"]*10**6, model.integrate("dmuDdx"), color="g", linewidth=2)
                    ax.plot(model.vars["x"]*10**6, np.zeros(len(model.vars["x"]*10**6)), color="orange", linewidth=2)
                    
                elif quantity == "dc" and model == self.electrolyte:
                    ax.plot(model.vars["x"]*10**6, model.integrate("dcLdx"), color="b", linewidth=2)
                    ax.plot(model.vars["x"]*10**6, model.integrate("dcDdx"), color="g", linewidth=2)
                    ax.plot(model.vars["x"]*10**6, np.zeros(len(model.vars["x"]*10**6)), color="orange", linewidth=2)
            else:
                temp_data["x"] = np.append(temp_data["x"], model.vars["x"] * 10**6) 
                temp_data["T"] = np.append(temp_data["T"], np.ones(2) * model.vars["T"])
               
        if quantity == "T":
            # T, phi and sigma accumulated are plotted as one line along whole cell
            ax.plot(temp_data["x"], temp_data["T"], color="r", linewidth=2)
            
        elif quantity == "phi":
            ax.plot(phi_data["x"], phi_data["phi"], color="r", linewidth=2)
            ax.plot([0, 74, 74, 86, 86, 153], [0, 0, -0.15, -0.15, 3.3, 3.3], color="b", linewidth=1, linestyle="--")
        
        elif quantity == "sigma accumulated":
            ax.plot(sigma_ac_data["x"], sigma_ac_data["sigma_ac"], color="r", linewidth=2)
        
        # format x-axes
        ax.set_xlabel(r' $x$ / $\mu$m')
        ax.set_xlim(self.anode.vars["x"][0]*10**(6), self.cathode.vars["x"][-1]*10**(6))
        
        # zoom on surfaces:
        #ax.set_xlim(self.anode.vars["x"][-2]*10**(6)+0.5, self.electrolyte.vars["x"][1]*10**(6)-0.4)
        #ax.set_xlim(self.electrolyte.vars["x"][-2]*10**(6)+0.55, self.cathode.vars["x"][1]*10**(6)-0.6)
        
        # zoom on electrolyte
        #ax.set_xlim(self.electrolyte.vars["x"][0]*10**(6), self.electrolyte.vars["x"][-1]*10**(6))
        
        # format temperature plot
        if quantity == "T":
            ax.set_ylabel("$T$ / K")
            #ax.set_ylim(290.0006, 290.0007)
            #ax.set_ylim(289.9988, 289.9989)
            #ax.set_title("Temperature profile", fontsize=13)
            y_ticks = ax.get_yticks()
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f"{tick:.4f}" for tick in y_ticks])
            
        # format electric potential plot
        if quantity == "phi":
            ax.set_ylabel(r"$\phi$ / V")
            #ax.set_title("Potential profile", fontsize=13)
            lines = (Line2D([0], [0], color = "r", linestyle="-"), Line2D([0], [0], color = "b", linestyle="--"))
            labels = ("In Operation", "OCV")
            ax.legend(lines, labels)
        
        # format concentration plot
        if quantity == "c":
            ax.set_ylabel("$c$ / mol m$^{-3}$")
            #ax.set_title("Concentration profile", fontsize=13)
            lines = (Line2D([0], [0], color = "r", linestyle="-"), Line2D([0], [0], color = "b", linestyle="-"), Line2D([0], [0], color = "g", linestyle="-"), Line2D([0], [0], color = "orange", linestyle="-"))
            labels = ("Li", "LiPF$_6$", "DEC", "EC")
            
            #lines = (Line2D([0], [0], color = "b", linestyle="-"), Line2D([0], [0], color = "g", linestyle="-"), Line2D([0], [0], color = "orange", linestyle="-"))
            #labels = ("LiPF$_6$", "DEC", "EC")
            
            ax.legend(lines, labels, loc='upper right')
            #ax.set_ylim(0, 8000)
        
        if quantity == "dmu":
            ax.set_ylabel(r"$\Delta \mu_{i,T}$ / J mol$^{-1}$")
            lines = (Line2D([0], [0], color = "b", linestyle="-"), Line2D([0], [0], color = "g", linestyle="-"), Line2D([0], [0], color = "orange", linestyle="-"))
            labels = ("LiPF$_6$", "DEC", "EC")
            
            ax.legend(lines, labels, loc='lower left')
        
        if quantity == "dc":
            ax.set_ylabel(r"$\Delta c$ / mol m$^{-3}$")
            lines = (Line2D([0], [0], color = "b", linestyle="-"), Line2D([0], [0], color = "g", linestyle="-"), Line2D([0], [0], color = "orange", linestyle="-"))
            labels = ("LiPF$_6$", "DEC", "EC")
            
            ax.legend(lines, labels, loc='lower left')
        
        # format heatflux plot
        if quantity == "Jq":
            ax.set_ylabel("$J'_q$ / W m$^{-2}$")
            #ax.set_title("Measurable heat flux", fontsize=13)
        
        # format local entropy plot
        if quantity == "sigma":
            ax.set_ylabel(r"$\sigma$ / W m$^{-3}$ K$^{-1}$")
            ax.set_title("Local entropy production", fontsize=13)
        
        # format accumulated entropy plot
        if quantity == "sigma accumulated":
            ax.set_ylabel(r"$\sigma$ / W m$^{-2}$ K$^{-1}$")
            #ax.set_title("Accumulated entropy production", fontsize=13)
        plt.show()
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
        
        return cumulative_trapezoid(self.vars[dydx], self.vars["x"], initial=0)  + y0
    
    def gradient(self, y):
        """
        calculate gradient in x of component y

        Parameters
        ----------
        y : string
            name of quantity in dictionary "vars"

        Returns
        -------
        dydx : np.array()
            gradient of y in x
        """
        return np.gradient(self.vars[y], self.vars["x"])
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
    
    def partial_molar_entropy(self, i, o):
        """
        Calcualte partial molar entropy of surface using the peltier heat and peltier coefficents

        Parameters
        ----------
        i : object
            instance of the bulk phase on the left-hand-side.
        o : object
            instance of the bulk phase on the lright-hand-side.

        Returns
        -------
        S_io : float
            partial molar entropy of surface
        """
        
        pi_i = i.params["peltier coefficient"]
        pi_o = o.params["peltier coefficient"]
        pi_sf = self.params["peltier heat"]
        T_i = i.vars["T"][-1]
        T_o = o.vars["T"][0]
        
        if self.name == "Anode Surface":
            S_io = (pi_o - pi_i - pi_sf)/T_i
        else:
            S_io = (pi_sf + pi_i - pi_o)/T_o
            
        return S_io
        
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
        
        return -Jq_io/(T_io*T_s) * dT_is - Jq_oi/(T_oi*T_s) * dT_so - j/T_s*(dphi + dG/F)

#%%% electrode
class Electrode(Submodel):
    def __init__(self, params, name):
        super().__init__(params, name)
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
        cp = self.params["heat capacity L"]

        T, dTdx = S
        #rhs     = - pi*j / (lambda_*F*T) * dTdx - j**2 / (lambda_*kappa)
        rhs = (cp - pi/T)* j/(lambda_*F) * dTdx - j**2 / (lambda_*kappa)
        return[dTdx, rhs]
    
    def partial_molar_entropy(self, S_sf):
        sigma_accum = self.integrate("sigma")[-1]
        Jq_r, Jq_l = self.vars["Jq"][-1],self.vars["Jq"][0]
        T_r, T_l = self.vars["T"][-1],self.vars["T"][0]
        
        if self.name == "Anode":
            S_boundary = (Jq_r/T_r - Jq_l/T_l - sigma_accum)*(F/j) + S_sf
        else: 
            S_boundary = (sigma_accum + Jq_l/T_l - Jq_r/T_r)*(F/j) + S_sf 
        
        return S_boundary 
    
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
        
        return -pi/F * dTdx/T - j/kappa
    
    def dcdx(self):
        """
        calcualtes dcdx

        Returns
        -------
        dcdx : np.array()
        """
        D = self.params["diffusion coefficient"]
        size = len(self.vars["x"])
        if D == 0:
            dcdx = np.zeros(size)
        else:
            dcdx = - j/(D*F) * np.ones(size)
        return dcdx
    
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
        dTdx   = self.vars["dTdx"]
        Jq     = self.vars["Jq"]
        J_L    = self.vars["J_L"]
        dmudx  = self.vars["dmudx"]
        dphidx = self.vars["dphidx"]
        
        return - dTdx/T**2 * Jq - dmudx/T * J_L - dphidx/T * j
#%%% electrolyte
class Electrolyte(Submodel):
    def __init__(self, params, name):
        super().__init__(params, name)
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
    
    def __concL_function(self, x, cL, coeff_fun):
        """
        serves as input function for ivp_solve() to solve for lithium concentration
        
        Parameters
        ----------
        x : np.array()
            spatial mesh
        c : np.array()
            lithium concentration along x

        Returns
        -------
        dcdx : np.array()
            returns function dcdx of c and x.
        """
        tdf_LL = self.params["thermodynamic factor LL"]
        tdf_DD = self.params["thermodynamic factor DD"]
        tdf_LD = self.params["thermodynamic factor LD"]
        tdf_DL = self.params["thermodynamic factor DL"]
        
        T, dmuLdx, dmuDdx = coeff_fun(x)
        
        return (dmuLdx - tdf_LD/tdf_DD*dmuDdx)/(tdf_LL - tdf_LD*tdf_DL/tdf_DD) * cL/(R*T)
    
    def __concD_function(self, x, cD, coeff_fun):
        """
        serves as input function for ivp_solve() to solve for DEC concentration
        
        Parameters
        ----------
        x : np.array()
            spatial mesh
        c : np.array()
            DEC concentration along x

        Returns
        -------
        dcdx : np.array()
            returns function dcdx of c and x.
        """
        tdf_DD = self.params["thermodynamic factor DD"]
        tdf_DL = self.params["thermodynamic factor DL"]
        
        T, dmuLdx, dmuDdx, dcLdx, cL = coeff_fun(x)

        return cD*(dmuDdx/(R*T*tdf_DD) - tdf_DL/tdf_DD/cL)
    
    def __conL_coefficients(self, x):
        """
        return postion dependent parameters for solve_ivp function
        for lithium function

        Parameters
        ----------
        x : float
            x value for current step in solve_ivp

        Returns
        -------
        return value that is closest to current x-position
        """
        idx = np.argmin(np.abs(self.vars["x"] - x))
        return self.vars["T"][idx], self.vars["dmuLdx"][idx], self.vars["dmuDdx"][idx]
    
    def __conD_coefficients(self, x):
        """
        return postion dependent parameters for solve_ivp function
        for DEC function

        Parameters
        ----------
        x : float
            x value for current step in solve_ivp

        Returns
        -------
        return value that is closest to current x-position
        """
        idx = np.argmin(np.abs(self.vars["x"] - x))
        return self.vars["T"][idx], self.vars["dmuLdx"][idx], self.vars["dmuDdx"][idx], self.vars["dcLdx"][idx], self.vars["cL"][idx]
    
    def c(self, component):
        """
        solves for c 

        Returns
        -------
        c : np.array()
            concentration along x.
        dcdx : np.array()
        """
        x = self.vars["x"]
        
        if component == "L":
            c0 = self.params["initial concentration L"]
            sol = solve_ivp(self.__concL_function, (x[0], x[-1]), [c0], args=(self.__conL_coefficients,), t_eval=x)
        else:
            c0 = self.params["initial concentration D"]
            sol = solve_ivp(self.__concD_function, (x[0], x[-1]), [c0], args=(self.__conD_coefficients,), t_eval=x)
       
        c = sol.y[0]
        return c
    
    def dcdx(self):
        """
        cacluates dcdx

        Returns
        -------
        dcdx : np.array()
        """
        return np.gradient(self.vars["c"], self.vars["x"])
    
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
        
        return -a/T * dTdx + b*j/F * T
    
    def sigma(self):
        """
        calculate local entropy production

        Returns
        -------
        sigma : np.array()
        """
        T      = self.vars["T"]
        dTdx   = self.vars["dTdx"]
        Jq     = self.vars["Jq"]
        dphidx = self.vars["dphidx"]
        
        return - dTdx/T**2 * Jq - dphidx/T * j
    

