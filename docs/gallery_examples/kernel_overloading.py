"""
Kernel Overloading Example
==========================

This example demonstrates kernel overloading using CodPy.
"""

# import libraries
import numpy as np
from codpydll import *

import codpy.core as core


# define the class
class my_kernel(core.cd.kernel):
    # An example of overloading codpy kernel with a user-defined expression.

    def __init__(self, bandwith=1.0, **kwargs):
        core.cd.kernel.__init__(self)
        self.bandwidth_ = bandwith

    def k(self, x, y):
        out = np.linalg.norm((x - y) * self.bandwidth_)
        return out * out * 0.5

    def grad(self, x, y):
        return y * self.bandwidth_


# %% [markdown]
# Generate data for the kernel.

# %%
core.KerInterface.set_verbose(True)
x, y = np.random.randn(3, 2), np.random.randn(3, 2)

# %% [markdown]
# Create a kernel object and display the result.

# %%
my_kernel1 = my_kernel(1)
my_kernel2 = my_kernel(2)
result_1 = my_kernel1.k(x[0], y[0])
print("Result 1:", result_1)

# %% [markdown]
# Set the kernel and display the next result.
# my_kernel2 and my_kernel_ptr2 are the same object.

# %%
my_kernel.set_kernel_ptr(my_kernel2)
my_kernel_ptr2 = core.KerInterface.get_kernel_ptr()
result_2 = my_kernel_ptr2.k(x[0], y[0])
print("Result 2:", result_2)

# %% [markdown]
# Compute the Gram matrix with my_kernel2 and display it.

# %%
gram_matrix = core.KerOp.knm(x, y)
print("Gram Matrix:", gram_matrix)

# %% [markdown]
# You can switch kernel as follow.
my_kernel.set_kernel_ptr(my_kernel1)
print(core.KerOp.knm(x, y))
pass

# %%
