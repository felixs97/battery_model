# battery_model
modeling lithium-ion battery cell using non-equilibrium thermodynamics

## Files and Their Functions

### `main.py`
This is the main file that sets up and runs the simulation. The main steps are:
1. Importing the necessary modules and parameters.
2. Instantiating the `LiionModel` with a scenario name and material properties.
3. Initializing the mesh for different battery cell layers.
4. Setting the boundary temperature conditions.
5. Solving the model.
6. Plotting the results.

```python
from params_sys import Tamb
import params_LFP
import classes as c

model = c.LiionModel("Baseline Scenario", params_LFP)
model.init_mesh({
    "Anode": 100,
    "Electrolyte": 20,
    "Cathode": 100
})
model.boundary_conditions(Tamb, Tamb)
model.solve()
model.plot()
```

### `params_sys.py`
Containig physical constants and system parameters like ambient temperature

### `params_LFP.py`
Dictionary with all material properties of cell to be modeled. 
In this case using a Graphite-Anode and a LFP-Cathode with a ternary electrolyte mixture of EC, DEC and LiPF_6

### `classes.py`
Methods and Functions are defined to run the model the main file
The `LiionModel` class functions as the overarching container, combining all submodels. Users initiate a simulation by creating an instance of this class, triggering the initialization of the associated submodels.
Upon initializing a submodel, a name is assigned to facilitate access to the corresponding parameter set. Additionally, a variables dictionary (`vars`) is initialized to store calculated values. Since each submodel is governed by distinct equations, they are defined in separate classes, each inheriting from the Submodel base class.
Within each submodel class (e.g., `Surface`, `Electrode`, `Electrolyte`), functions are defined to solve the governing equations derived in earlier sections. The entire system is solved through the `solve()` method within the `LiionModel` class.
The solving process begins with determining the temperature distribution across the cell. This is achieved by solving the system of equations using the shooting method, as described earlier. The left boundary temperature condition and an initial guess for its derivative are applied to solve the system from left to right. The resulting temperature on the right-hand side is then compared to the specified right boundary temperature. An optimization process is applied to ensure align- ment with the boundary condition. Once the temperature distribution is known, subsequent quantities are calculated accordingly.
