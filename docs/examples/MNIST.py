"""
MNIST Examples
==========================

We illustrate some basic considerations while manipulating the class :class:`codpy.kernel.Kernel` classes , applying it to the `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_ problem.
It illustrates the following interpolation / extrapolation method

    $$f_{k,\\theta}(\cdot) = K(\cdot, Y) \\theta, \quad \\theta = K(X, Y)^{-1} f(X),$$
where
    - $K(X, Y)$ is the Gram matrix, see :func:`codpy.kernel.Kernel.Knm`
    - $K(X, Y)^{-1} = (K(Y, X)K(X, Y))^{-1}K(Y,X)$ is computed as a least-square method without regularization terms, see :func:`codpy.kernel.Kernel.get_knm_inv`.

This notebook illustrates various choices+ for the set $Y$
"""


import os
import pandas as pd
import numpy as np
import random
# We use a custom hot encoder for performances reasons.
from codpy.data_processing import hot_encoder
# Standard codpy kernel class.
from codpy.kernel import Kernel,KernelClassifier
import codpy.core as core
# A multi scale kernel method.
from sklearn.metrics import confusion_matrix
from codpy.clustering import *

os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"

# %% [markdown]
# We pick-up MNIST data using tensorflow (tf). pip install tf on your installation prior running this notebook !
# 
# Note : 
# 
#            - The MNIST corresponds to features $X\in \mathbb{R}^{60000,784}$. Each image, represented as $28\times 28$ black and white pixel is a feature described as a vector in dimension $D=784$.
#
#            - We hot encode the MNIST classes : $f(X) \in \mathbb{R}^{60000,10}$ is defined as 
#                        $$f(x) = (\delta_i(c(x)), \quad i=1,\ldots,10,$$
#               where $c(x)$ is the indice of the label of the class, and $\delta_i(j) = \{i==j\}$.
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
# We perform basic tests on MNIST results : confusion matrix and scores.

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
# Set variables and pick MNIST data for the test.
# N_MNIST_pics is used to pick a smaller set than the original one.
# The training set is `x,fx`, the test set is `z,fz`.
N_clusters=100
N_MNIST_pics=5000
x, fx, z, fz = get_MNIST_data(N_MNIST_pics)


# %% [markdown]
# First pick $Y$ at random. Output confusion matrix for two sets : the training set $X$ and the test set $Z$ 
indices = np.random.choice(range(x.shape[0]),size=N_clusters)
y,fy = x[indices],fx[indices]
predictor = KernelClassifier(x=x, y=y,fx=fx)
print("Output with the training set - reproductibility test:")
show_confusion_matrix(x, fx, predictor,cm=False)
print("Output with the test set :")
show_confusion_matrix(z, fz, predictor)
print("Random Discrepancy(x,y):", predictor.discrepancy(y))

# %% [markdown]
# Select $Y$ having a lowest discrepancy with a greedy algorithm.
y = GreedySearch(x,N=N_clusters).cluster_centers_
predictor = KernelClassifier(x=x,y=y,fx=fx)
print("GreedySearch Reproductibility test:")
show_confusion_matrix(x, fx, predictor,cm=False)
print("GreedySearch Performance test:")
show_confusion_matrix(z, fz, predictor,cm=False)
print("GreedySearch Discrepancy(x,y):", predictor.discrepancy(predictor.get_y()))

# %% [markdown]
# Select $Y$ adapted to $f(x)$ using a greedy algorithm.
predictor = KernelClassifier(x=x,fx=fx).greedy_select(N=N_clusters,all=True,fx=fx)
print("greedy_select Reproductibility test:")
show_confusion_matrix(x, fx, predictor,cm=False)
print("greedy_select Performance test:")
show_confusion_matrix(z, fz, predictor,cm=False)
print("greedy_select Discrepancy(x,y):", predictor.discrepancy(predictor.get_y()))

# %% [markdown]
# Select $Y$ adapted to $f(x)$ using a greedy algorithm.
y = SharpDiscrepancy(x,N=N_clusters,set_kernel=core.kernel_setter(kernel="gaussian", map="standardmean")).cluster_centers_
predictor = KernelClassifier(x=x,y=y,fx=fx)
print("SharpDiscrepancy Reproductibility test:")
show_confusion_matrix(x, fx, predictor,cm=False)
print("SharpDiscrepancy Performance test:")
show_confusion_matrix(z, fz, predictor,cm=False)
print("SharpDiscrepancy Discrepancy(x,y):", predictor.discrepancy(predictor.get_y()))

# %% [markdown]
# Select $Y$ using a k-means algorithm.
y = MiniBatchkmeans(x,N=N_clusters).cluster_centers_
predictor = KernelClassifier(x=x,y=y,fx=fx)
print("kmeans Reproductibility test:")
show_confusion_matrix(x, fx, predictor,cm=False)
print("kmeans Performance test:")
show_confusion_matrix(z, fz, predictor,cm=False)
print("kmeans Discrepancy(x,y):", predictor.discrepancy(predictor.get_y()))