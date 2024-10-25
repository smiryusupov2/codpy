"""
=============================================
Experiment: Random Data Generation and Plotting
=============================================

This experiment demonstrates how to generate random data using a custom function
and plot it using the CodPy library. We use `data_random_generator` to create data
in different cartesian coordinates and visualize the results using `multi_plot`.

The function `my_fun(x)` is used to create a sinusoidal signal, and the data is
generated in different coordinate systems to see how the function behaves.

**Steps:**

1. Define the sinusoidal function.

2. Use `data_random_generator` to generate the data.

3. Plot the generated data using `multi_plot`.

Below is the code for this experiment:
"""

# Importing necessary modules
import os
import sys

current_dir = os.getcwd()
sys.path.append(current_dir)
import matplotlib.pyplot as plt
import numpy as np

from codpy_old.codpy.com.generators import *
from codpy_old.codpy.data.data_generators import data_random_generator
from codpy_old.codpy.pred import *


def my_fun(x):
    """
    A sinusoidal function that generates a sum of sines based on the input `x`.
    """
    from math import pi

    sinss = np.cos(2 * x * pi)
    if x.ndim == 1:
        sinss = np.prod(sinss, axis=0)
        ress = np.sum(x, axis=0)
    else:
        sinss = np.prod(sinss, axis=1)
        ress = np.sum(x, axis=1)
    return ress + sinss


def labelxfxzfz():
    """
    This function generates random data using `data_random_generator` and plots
    the data using `multi_plot`. It creates data for three sets of coordinates
    and visualizes the results.
    """
    data_random_generator_ = data_random_generator(
        fun=my_fun, types=["cart", "cart", "cart"]
    )
    x, fx, y, fy, z, fz = data_random_generator_.get_data(D=1, Nx=100, Ny=100, Nz=100)

    # Plot the data for x and z
    multi_plot([(x, fx), (z, fz)], plot1D, mp_nrows=1, mp_figsize=(12, 3))


# Call the function to generate and plot the data
labelxfxzfz()
plt.show()


#########################################################################
# **Expected Output:**
# The plot should show a sinusoidal pattern for the data generated in the cartesian coordinate system.
# The two curves for `x` and `z` should exhibit the sinusoidal variations defined by `my_fun(x)`.
