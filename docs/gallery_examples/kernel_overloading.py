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

    def __init__(self, **kwargs):
        core.cd.kernel.__init__(self)
        self.bandwidth_ = float(kwargs.get("bandwidth", 1.0))

    @staticmethod
    def create(kwargs={}):
        return my_kernel(**kwargs)

    @staticmethod
    def register():
        cd.kernel.register("my_kernel", my_kernel.create)

    def k(self, x, y):
        out = np.linalg.norm((x - y) * self.bandwidth_)
        return out * out * 0.5

    def grad(self, x, y):
        return y * self.bandwidth_


# %% [markdown]
# Generate data for the kernel.

# %%
x, y = np.random.randn(3, 2), np.random.randn(3, 2)

# %% [markdown]
# Create a kernel object and display the result.

# %%
my_kernel_ = my_kernel.create()
result_1 = my_kernel_.k(x[0], y[0])
print("Result 1:", result_1)

# %% [markdown]
# Set the kernel and display the next result.

# %%
my_kernel.set_kernel_ptr(my_kernel_)
my_kernel_ptr = core.kernel_interface.get_kernel_ptr()
result_2 = my_kernel_.k(x[0], y[0])
print("Result 2:", result_2)

# %% [markdown]
# Compute the Gram matrix and display it.

# %%
gram_matrix = core.op.Knm(x, y)
print("Gram Matrix:", gram_matrix)

# %% [markdown]
# Register the kernel (no output for this step).

# %%
my_kernel.register()
my_kernel_ptr = core.factories.get_kernel_factory()["my_kernel"]({"bandwidth": "2."})


# !!!!!!!! a corriger
# print(my_kernel_ptr.k(x[0], y[0]))
# my_kernel.set_kernel_ptr(my_kernel_ptr)
# print(core.op.Knm(x, y))
