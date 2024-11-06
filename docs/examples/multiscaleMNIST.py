"""
Multiscale MNIST Examples
==========================

We illustrate the class :class:`codpy.multiscale_kernel.MultiScaleKernel`, applying it to the `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_ problem.
The methodology is similar to the gallery example :ref:`MNIST <MNIST>`_
"""

import os
import pandas as pd
import numpy as np
import random
# We use a custom hot encoder for performances reasons.
from codpy.data_processing import hot_encoder
# Standard codpy kernel class.
from codpy.kernel import Kernel
# A multi scale kernel method.
from codpy.multiscale_kernel import *
from sklearn.metrics import confusion_matrix

os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"

# %% [markdown]
def get_MNIST_data(N=-1):
    import tensorflow as tf

    (x, fx), (z, fz) = tf.keras.datasets.mnist.load_data()
    x, z = x / 255.0, z / 255.0
    x, z, fx, fz = (
        x.reshape(len(x), -1),
        z.reshape(len(z), -1),
        fx.reshape(len(fx), -1),
        fz.reshape(len(fz), -1),
    )
    fx, fz = (
        hot_encoder(pd.DataFrame(data=fx), cat_cols_include=[0], sort_columns=True),
        hot_encoder(pd.DataFrame(data=fz), cat_cols_include=[0], sort_columns=True),
    )
    x, fx, z, fz = (x, fx.values, z, fz.values)
    if N != -1:
        indices = random.sample(range(x.shape[0]), N)
        x, fx = x[indices], fx[indices]

    return x, fx, z, fz


# %% [markdown]
def show_confusion_matrix(z, fz, predictor=None, cm=True):
    f_z = predictor(z)
    fz, f_z = fz.argmax(axis=-1), f_z.argmax(axis=-1)
    out = confusion_matrix(fz, f_z)
    if cm:
        print("confusion matrix:")
        print(out)
    print("score MNIST:", np.trace(out) / np.sum(out))
pass

# %% [markdown]
# Run codpy silently on/off.
core.kernel_interface.set_verbose(False)

# %% [markdown]
# The training set is `x,fx`, the test set is `z,fz`.
N_clusters=100
N_MNIST_pics=-1
x, fx, z, fz = get_MNIST_data(N_MNIST_pics)

# %% [markdown]
# Select a multi scale kernel method where the centers are given by a k-mean algorithm.
N_partition=5
predictor = MultiScaleKernelClassifier(x=x,fx=fx,N=N_partition,method=MiniBatchkmeans)
print("Reproductibility test:")
show_confusion_matrix(x, fx, predictor,cm=False)
print("Performance test:")
show_confusion_matrix(z, fz, predictor,cm=False)


# %% [markdown]
# Select a multi scale kernel where the centers are given by a greedy search algorithm.
predictor = MultiScaleKernelClassifier(x=x,fx=fx,N=N_partition,method=GreedySearch)
print("Reproductibility test:")
show_confusion_matrix(x, fx, predictor,cm=False)
print("Performance test:")
show_confusion_matrix(z, fz, predictor,cm=False)
