CodPy description
===============

Introduction
------------

Codpy is a kernel based, open source software library for high performance numerical computation, relying on the [RKHS](https://en.wikipedia.org/wiki/Reproducingkernel_Hilbert_space) theory.
It contains a set of core tools that we use for machine Learning, statistics and numerical simulations. As a machine learning platform, it enjoys some interesting properties :

 * It is a numerically efficient machine learning platform. We provide benchmark tools to compare codpy to other popular machine learning platforms.
 * It is a white box method. Any learning machine has access to worst-error bounds computations. These allow to compute confidence levels of prediction on any test set. Moreover, reproducibility properties of kernel methods allow to fully understand and explain obtained results.
 * Each learning machine has access to all classical differential operators. These properties allow us to use this library with any PDE (partial differential equations) approach.
 * Each learning machine has access to optimal transport tools, much needed for statistics.

General setting
------------
We introduce the main components of our CodPy algorithms, which are designed to tackle problems in the field of machine learning. 

* The first component is the selection of a reproducing-kernel space, along with the use of transformation maps that allow us to customize basic kernels for specific problems. 

* The second component involves the definition of mesh-free discrete differential operators, which are relevant for machine learning applications.

These two key components form the basis of our machine learning algorithms, as well as our methods for handling problems that involve partial differential operators. In the subsequent chapters of this monograph, we delve into more advanced algorithms and explore a range of applications.

For a description of the framework of interest we need some notation. A set of $N_x$ variables in $D$ dimensions is given, denoted by $X \in \mathbb{R}^{N_x, D}$, together with a $D_f$-dimensional vector-valued data function $f(X) \in \mathbb{R}^{N_x , D_f}$ which represents the *training values* associated with the *training set* $X$. The input data therefore consists of 

$$(X,f(X)) := \{x^n, f(x^n)\}_{n = 1,\dots,N_x}, \qquad X \in \mathbb{R}^{N_x , D}, \qquad f(X) \in \mathbb{R}^{N_x , D_f}.$$

We are interested in predicting the so-called *test values* $f_Z \in \mathbb{R}^{N_z , D_f}$ on a new set of variables called the *test set* $Z \in \mathbb{R}^{N_z , D}$: 
$$(Z,f_Z) := \{z^n, f_z^n\}_{n = 1,\dots,N_z}, \qquad Z \in \mathbb{R}^{N_z , D}, \qquad f_Z \in \mathbb{R}^{N_z , D_f}.$$

Python Function: Projection Operator
-------------------------------------

The Python function in our framework that describes the projection operator $\mathcal{P}_{k}$ is 

$$f_z = \text{op.projection}(X,Y,Z, f(X)=[], k=None, rescale = False) \in \mathbb{R}^{N_z , D_f}.$$
The function has the following optional values:

- The function $f(X)$ is optional, which allows the user to retrieve the whole matrix $\mathcal{P}_{k}(X,Y,Z) \in \mathbb{R}^{N_z , N_x}$ if desired.
- The kernel $k$ is optional, which provides the user with the freedom to re-use an already input kernel.
- The optional value \textit{rescale} is defaulted to false, which allows for calling the map prior to performing the projection operation. 
This step helps in computing the internal states of the map for proper data scaling. For instance, a rescaling computes $\alpha$ according to the set ($X,Y,Z$).

Interpolation and extrapolation Python functions are, simple decorators applied to the operator $\mathcal{P}_{k}$, as seen in the projection operator. 
The main question at this stage is how well the approximation $f_z$ compares to $f(Z)$, which is addressed in the next section. 

$$f_z = \text{op.extrapolation}(X,Z,f(X) = [],\ldots), \quad f_z = \text{op.interpolation}(X,Z,f(X) = [],\ldots)$$