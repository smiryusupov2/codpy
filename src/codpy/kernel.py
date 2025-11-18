import numpy as np
import os
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "4"
from codpydll import *
from scipy.special import softmax
import scipy

import codpy.core as core
import codpy.algs as algs
from codpy.algs import Alg
from codpy.core import DiffOps
from codpy.lalg import LAlg
from codpy.permutation import lsap, map_invertion,Gromov_Monge
from codpy.sampling import get_uniforms,get_normals,get_qmc_uniforms,get_qmc_normals
from codpy.dictionary import cast
import sparse_dot_mkl,pypardiso


class Kernel:
    """
    class to manipulate datas for various kernel-based operations, such as interpolations or extrapolations of functions, or mapping between distributions.
        Note:
            This class is similar to libraries as scikit-learn or XGBoost, in the sense that they use a fit / predict pattern, with the following correspondances and differences.

            - Datas are loaded into memory in the contructor :func:`__init__`, or via :func:`set`
            - For matching distributions, use :func:`map`,
            - The `predict` function is made directly through :func:`__call__`

            It implements the following methods :

            - In the context of functions interpolation / extrapolation

                $$f_{k,\\theta}(\cdot) = K(\cdot, Y) \\theta, \quad \\theta = K(X, Y)^{-1} f(X),$$

                - $K(X, Y)$ is the Gram matrix, see :func:`knm`
                - $K(X, Y)^{-1} = (K(Y, X)K(X, Y) + \epsilon R(Y,Y))^{-1}K(Y,X)$ is computed as a least-square method with optional regularization terms, , see :func:`get_knm_inv`.

            - For matching distributions
                $$f_{k,\\theta}(\cdot) = K(\cdot, Y) K(X, Y)^{-1} f(X\circ \sigma)$$, where $\sigma$ is a permutation.

            - Fitting is done just-in-time (at first prediction), and means computing the parameters $\\theta = K(X, Y)^{-1} f(X)$, together with $\sigma$ for distributions. The function :func:`get_theta()` performs those computations and corresponds to fit in others frameworks.

    """

    def __init__(
        self,
        x=None,
        y=None,
        fx=None,
        theta=None,
        max_nystrom: int = 1000,
        reg: float = 1e-9,
        order: int = None,
        n_batch: int = sys.maxsize,
        set_kernel: callable = None,
        set_clustering: callable = None,
        **kwargs: dict,
    ) -> None:
        """
        Initializes the Kernel class with default or user-defined parameters.

        :param x: A bi-dimensional numpy array.
        :param fx: A bi-dimensional numpy array. If `x` or `fx` is not `None`, then call :func:`set`
        :param max_nystrom: Maximum number of Nystrom samples. Defaults to 1000.
        :type max_nystrom: :class:`int`, optional
        :param reg: Regularization parameter for kernel operations. Defaults to 1e-9.
        :type reg: :class:`float`, optional
        :param order: Polynomial order for polynomial kernel functions. Defaults to ``None`` (no polynomial regression).
        :type order: :class:`int`, optional, order of the polynomial kernel
        :param dim: Dimensionality of the input data. Defaults to 1.
        :type dim: :class:`int`, optional
        :param set_kernel: A custom kernel function initializer. If not provided, defaults to ``self.default_clustering_functor()``.
        :type set_kernel: :class:`callable`, optional
        :param kwargs: Additional keyword arguments for further customization.
        :type kwargs: :class:`dict`
        """
        self.order = order
        if self.order is None:
            if set_kernel is not None and hasattr(set_kernel, "polynomial_order"):
                self.order = set_kernel.polynomial_order
        self.reg = reg
        self.dim_ = None
        self.max_nystrom = int(max_nystrom)
        self.n_batch = n_batch

        if theta is not None:
            self.theta = theta

        if set_clustering is not None:
            self.set_clustering = set_clustering
        else:
            self.set_clustering = self.default_clustering_functor()

        if set_kernel is not None:
            self.set_kernel = set_kernel
        else:
            self.set_kernel = self.default_kernel_functor()
        self.valid = False
        self.kernels = {}
        if x is not None or fx is not None:
            self.set(x=x, y=y, fx=fx, theta=theta,**kwargs)
        else:
            self.x, self.y, self.fx = None, None, None

    def is_valid(self):
        return self.valid

    def dim(self) -> int:
        """
        return the dimension of the training set
        """
        return self.dim_

    def __set_dim(self, dim):
        """
        return the dimension of the training set
        """
        self.dim_ = dim
        return self

    def default_clustering_functor(self) -> callable:
        """
        Initialize and return the default clustering method for large dataset partitioning.
        :returns: A clustering of the set x into N clusters.
        :rtype: :class:`callable`
        """
        from codpy.clustering import BalancedClustering, MiniBatchkmeans

        return lambda x, N, **kwargs: BalancedClustering(MiniBatchkmeans, x=x, N=N)

    def default_kernel_functor(self) -> callable:
        """
        Initialize and return a default kernel function.

        This method provides a default kernel initialization. We picked up a quite simple but robust kernel functor

            >>> core.kernel_setter("maternnorm", "standardmean", 0, 1e-9)

        defining the `maternnorm` kernel with the `standardmean` map. It sets a polynomial order of 0 and a regularization value of 1e-9.

        :returns: The initialized default kernel function using :func:`core.kernel_setter`.
        :rtype: :class:`callable`

        Example:
            >>> default_kernel = kernel.default_kernel_functor()
        """
        out = core.kernel_setter("maternnorm", "standardmean", 0, 1e-9)
        if self.get_order() is None:
            return out
        else:

            class piped:
                def __init__(self, fun, order, reg):
                    self.fun, self.order, self.reg = fun, order, reg

                def __call__(self, **kwargs):
                    self.fun()
                    linearkernel = core.KernelSetters.kernel_helper(
                        setter=core.KernelSetters.set_linear_regressor_kernel,
                        polynomial_order=self.order + 1,
                        regularization=self.reg,
                        set_map=None,
                    )
                    KerInterface.pipe_kernel_fun(linearkernel, self.reg)

            return piped(out, self.get_order(), self.reg)

    def set_custom_kernel(
        self,
        kernel_name: str,
        map_name: str,
        poly_order: int = 0,
        reg: float = None,
        bandwidth: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Provide a downlink to internal codpy kernel with flexible parameters.

        :param kernel_name: Name of the kernel function to use (e.g., ``'gaussian'``).
        :type kernel_name: :class:`str`
        :param map_name: Name of the mapping function (e.g., ``'standardmin'``).
        :type map_name: :class:`str`
        :param poly_order: The polynomial order if using a polynomial kernel. Defaults to 0.
        :type poly_order: :class:`int`, optional
        :param reg: Regularization parameter. If not provided, uses the instance's `reg` value.
        :type reg: :class:`float`, optional
        :param bandwidth: Bandwidth for kernel functions that require it. Defaults to 1.0.
        :type bandwidth: :class:`float`, optional

        :returns: ``None``
        """
        reg = reg if reg is not None else self.reg
        kernel_function = core.kernel_setter(
            kernel_name, map_name, poly_order, reg, bandwidth
        )
        ## tester!!!!!!!!!!!!!!!!!!!
        self = Kernel(set_kernel=kernel_function)

    def get_order(self, **kwargs) -> int:
        """
        Retrieve the polynomial order for the kernel.

        :returns: The polynomial order if available, otherwise ``None``.
        :rtype: :class:`int` or :class:`None`

        Example:
            >>> order = kernel.get_order()
        """
        if not hasattr(self, "order"):
            self.order = None
        return self.order

    def knm(
        self, x: np.ndarray = None, y: np.ndarray = None, fy: np.ndarray = [], **kwargs
    ) -> np.ndarray:
        """
        Compute the kernel matrix $K(X, Y)=k(x^i, y^j)_{i,j}$, where the kernel function $k$ is defined at class initialization, see :attr:`self.set_kernel`.

        :param x: Input data points :math:`(N, D)`, where :math:`N` is the number of points and :math:`D` is the dimensionality.
        :type x: :class:`numpy.ndarray`
        :param y: Secondary data points :math:`(M, D)`, where :math:`M` is the number of points and :math:`D` is the dimensionality.
        :type y: :class:`numpy.ndarray`
        :param fy: Optional matrix values for optimization purposes. If not None, perform and return the multiplication $K(X, Y)f_y$.
        :type fy: :class:`numpy.ndarray`, optional

        :returns: The computed kernel matrix :math:`K` of size :math:`(N, M)`.
        :rtype: :class:`numpy.ndarray`

        Example:
            >>> x_data = np.array([...])
            >>> y_data = np.array([...])
            >>> kernel_matrix = Kernel(x=x_data,y=y_data).knm()
        """
        if x is None:
            x = self.get_x()
        if y is None:
            y = self.get_y()
        if self.get_map() is not None:
            x, y = self.get_map()(x), self.get_map()(y)

        return core.KerOp.knm(x=x, y=y, fy=fy, kernel_ptr=self.get_kernel()).astype(x.dtype)

    def dnm(
        self, x: np.ndarray = None, y: np.ndarray = None, fy: np.ndarray = [], distance = [], **kwargs
    ) -> np.ndarray:
        """
        Compute the kernel matrix $D(X, Y)=k(x^i, y^j)_{i,j}$, where the kernel function $k$ is defined at class initialization, see :attr:`self.set_kernel`.

        :param x: Input data points :math:`(N, D)`, where :math:`N` is the number of points and :math:`D` is the dimensionality.
        :type x: :class:`numpy.ndarray`
        :param y: Secondary data points :math:`(M, D)`, where :math:`M` is the number of points and :math:`D` is the dimensionality.
        :type y: :class:`numpy.ndarray`
        :param fy: Optional matrix values for optimization purposes. If not None, perform and return the multiplication $K(X, Y)f_y$.
        :type fy: :class:`numpy.ndarray`, optional

        :returns: The computed kernel matrix :math:`K` of size :math:`(N, M)`.
        :rtype: :class:`numpy.ndarray`

        Example:
            >>> x_data = np.array([...])
            >>> y_data = np.array([...])
            >>> kernel_matrix = Kernel(x=x_data,y=y_data).knm()
        """
        if x is None:
            x = self.get_x()
        if y is None:
            y = self.get_y()
        if self.get_map() is not None:
            x, y = self.get_map()(self.get_x()), self.get_map()(self.get_y())
        return core.KerOp.dnm(
            x=x, y=y, fy=fy, kernel_ptr=self.get_kernel(), distance=distance
        )

    def get_knm_inv(
        self,
        epsilon: float = None,
        epsilon_delta: np.ndarray = None,
        reg=None,
        **kwargs,
    ) -> np.ndarray:
        """
        Retrieve the inverse of the kernel matrix :math:`K^{-1}(x, y)` using least squares computations.

        :param epsilon: Regularization parameter for the inverse computation. Defaults to None.
        :type epsilon: :class:`float`, optional
        :param epsilon_delta: Delta values for adjusting regularization. Defaults to None.
        :type epsilon_delta: :class:`numpy.ndarray`, optional

        :returns: The inverse kernel matrix or the product with function values if provided.
        :rtype: :class:`numpy.ndarray`

        Note:

            - If the regularization parameter (``reg``) is empty:
                - If ``fx`` is empty: Returns a ``NumPy`` array of size $(N, M)$, representing the least square inverse of $K(x, y)$.
                - If ``fx`` is provided: Returns the product of $K^{-1}(x, y)$ and $f(x)$. This allows performance and memory optimizations.
            - If the regularization parameter (``reg``) is provided:
                - If ``fx`` is empty: Returns a ``NumPy`` array of size $(N, M)$, computed as $(K(y, x) K(x, y) + \epsilon)^{-1} K(y, x)$
                - If ``fx`` is provided: Returns the product of :math:`K^{-1}(x, y)` and :math:`f(x)`.

        Example:

            >>> x_data = np.random.rand(100, 10)
            >>> y_data = np.random.rand(80, 10)
            >>> fx_data = np.random.rand(80, 5)
            >>> inv_kernel = kernel.get_knm_inv(x=x_data, y=y_data, fx=fx_data)
        """
        if not hasattr(self, "knm_inv"):
            self.knm_inv = None

        if self.knm_inv is None:
            if epsilon is None:
                epsilon = self.reg
            if epsilon_delta is None:
                epsilon_delta = []
            else:
                epsilon_delta = epsilon_delta * self.get_Delta()
            if self.get_map() is not None:
                x, y = self.get_map()(self.get_x()), self.get_map()(self.get_y())
            else:
                x, y = self.get_x(), self.get_y()
            if reg is None:
                reg = self.reg
            self._set_knm_inv(
                core.KerOp.knm_inv(
                    x=x,
                    y=y,
                    order=self.order,
                    reg=reg,
                    reg_matrix=epsilon_delta,
                    kernel_ptr=self.get_kernel(),
                    **kwargs,
                )
            )
        return self.knm_inv

    def get_knm(self, **kwargs) -> np.ndarray:
        """
        Retrieve or compute the Gram matrix $K(x, y)$ for the kernel.

        :returns: The Gram matrix $K(x,y)$.
        :rtype: :class:`numpy.ndarray`
        """
        if self.get_map() is not None:
            x, y = self.get_map()(self.get_x()), self.get_map()(self.get_y())
        else:
            x, y = self.get_x(), self.get_y()
        if not hasattr(self, "knm_") or self.knm_ is None:
            self._set_knm(core.KerOp.knm(x=x, y=y, kernel_ptr=self.get_kernel()))
        return self.knm_

    def _set_knm_inv(self, k):
        self.knm_inv = k
        self.set_theta(None)

    def _set_knm(self, k):
        self.knm_ = k

    def get_x(self, **kwargs) -> np.ndarray:
        """
        Retrieve the input data ``x``.

        :returns: The input data or ``None`` if not set.
        :rtype: :class:`numpy.ndarray` or :class:`None`
        """
        if not hasattr(self, "x"):
            self.x = None
        return self.x

    def density(self, x, **kwargs):
        """
        Return an unnormalized model of the density of the law $p_k(x | X)$, $X=(x^1,\cdots,x^N)$ being the training set and $k$ the kernel.
        This model comes from the kernel extrapolation operator $f_k(x)=k(x,X)k(X,X)^{-1}f(X)$
        $\int f(x) dp(x | X) = \int f(x) p_k(x | X) dx$
        The output is a distribution having input size $(x.shape[0])$.

        This distribution is valued as $\sum_X k(x,X)k(X,X)^{-1}$,
        Take care that this quantity can be negative, and might lead to biaised estimations, depending on the used kernels.
        For unbiaised densities estimations, use NadarayaWatson instead.
        """
        out = LAlg.prod(self.knm(x, self.get_x()), self.get_knm_inv())
        out = out.sum(axis=1) / out.shape[1]
        return out

    def set_x(self, x: np.ndarray, rescale=True, **kwargs) -> None:
        """
        Set the input data ``x`` for the kernel and update related internal states.

        This method sets the input data and optionally recalculates the kernel matrices.

        :param x: Input data points to be set.
        :type x: :class:`numpy.ndarray`
        """
        self.x = x.copy()
        self.__set_dim(x.shape[1])
        self.set_y()
        self._set_knm_inv(None)
        self._set_knm(None)
        self.set_theta(None)
        if rescale:
            self.rescale()
    def set_random_theta(self, **kwargs) -> None:
        self.set_theta(np.random.uniform(size=[self.x.shape[0],self.fx.shape[1]]))
    def set_y(self, y: np.ndarray = None, **kwargs) -> None:
        """
        Set the target data ``y`` for the kernel. If no target data is provided, ``y`` is set equal to ``x``.

        If interpolation/extrapolation is used, the following formula is applied:

        $$
        f_{\\theta}(.) = K(., Y)\\theta, \\quad \\theta = K(X, Y)^{-1} f(X).
        $$

        :param y: Target data points. If None, ``y`` is set equal to ``x``.
        :type y: :class:`numpy.ndarray`, optional
        """
        if y is None:
            self.y = self.get_x()
        else:
            self.y = y.copy()
        self._set_knm_inv(None)
        self._set_knm(None)

    def get_error_field(self, z=None, theta = None,fx= None, **kwargs):
        if z is None: z = self.get_x()
        if theta is None: theta = self.get_theta()
        if fx is None: fx = self.get_fx()
        return self(z=z, theta = theta)-fx
    def get_error(self, z=None, theta = None,fx= None, **kwargs):
        out = self.get_error_field(z=z,theta=theta,fx= fx, **kwargs)
        return (out*out).sum()*.5


    def get_y(self, **kwargs) -> np.ndarray:
        """
        Retrieve the target data ``y``.

        :returns: The target data or the input data ``x`` if ``y`` is not set.
        :rtype: :class:`numpy.ndarray`
        """
        if not hasattr(self, "y") or self.y is None:
            self.set_y()
        return self.y

    def get_fx(self, **kwargs) -> np.ndarray:
        """
        Retrieve the function values ``fx`` for the input data.

        :returns: The function values or ``None`` if not set.
        :rtype: :class:`numpy.ndarray` or :class:`None`
        """
        if not hasattr(self, "fx"):
            self.fx = None
        if self.fx is None and self.get_theta() is not None:
            self.fx = LAlg.prod(self.get_knm(), self.get_theta())
        return self.fx

    def set_fx(self, fx: np.ndarray, **kwargs) -> None:
        """
        Set the function values ``fx`` for the input data.

        :param fx: Function values corresponding to the input data.
        :type fx: :class:`numpy.ndarray`
        """
        if fx is not None:
            self.fx = fx.copy()
        else:
            self.fx = None
        self.set_theta(None)

    def set_theta(self, theta: np.ndarray, **kwargs) -> None:
        """
        Set the coefficient ``theta`` for the kernel regression.

        The coefficient is computed by the formula:

        .. math::
            \\theta = K(X, Y)^{-1} f(X)

        :param theta: Coefficients for kernel regression.
        :type theta: :class:`numpy.ndarray`
        """
        self.theta = theta
        return self

    def get_theta(self, **kwargs) -> np.ndarray:
        """
        Retrieve the coefficient ``theta`` for kernel regression.
        :returns: The regression coefficient ``theta``.
        :rtype: :class:`numpy.ndarray`
        """
        if not hasattr(self, "theta") or self.theta is None:
            if self.fx is None:
                self.theta = None
            else:
                # Compute the regression coefficient `theta` using the kernel matrix inverse and the function values.
                self.theta = LAlg.prod(self.get_knm_inv(**kwargs), self.fx)
        return self.theta

    def get_Delta(self) -> np.ndarray:
        """
        Compute and retrieve the discrete Laplace-Beltrami operator ``Delta``.

        :returns: The Laplace-Beltrami operator.
        :rtype: :class:`numpy.ndarray`
        """

        if not hasattr(self, "Deltas") or self.Delta is None:
            self.Delta = DiffOps.nabla_t_nabla(self.y, self.x, kernel=self.get_kernel())
        return self.Delta

    def greedy_select(
        self, N, x=None, fx=None, all=False, n_batch=1, norm="frobenius", **kwargs
    ):
        """
        Select a subset of points using a greedy Nystrom approximation technique :

        $$Y^{n+1} = Y^{n} \cup \\arg \sup_{x \in X} d(Y^n,x),$$
        to quickly approximate the clustering problem $Y = \\arg \inf_{Y \subset X} d(Y,X),$ where we suppose the following structure
        $d(Y,X) = \sum_i d(Y,x^i)$.


        The selection is typically based on norms such as the discrepancy errors for distributions, Frobenius or classifier type distances.

        :param x: Input data points.
        :type x: :class:`numpy.ndarray`
        :param N: The number of points to select.
        :type N: :class:`int`
        :param fx: Function values corresponding to ``x``.

            - if fx is None,
                $$d(Y,X) = \\frac{1}{N_X} \\sum_{n=1}^{N_x}  k(x^n,\cdot) - \\frac{2}{N_Y} \\sum_{m=1}^{N_Y} k(\cdot,y^m)$$
                This choice corresponds to minimizing the discrepancy error, see :func:`core.KerOp.discrepancy_error()`.
            - if fx is not None, $d(X,Y) = \|f(X)-f_{k,\\theta}(X)\|$
                In which case, we are interested in adaptive mesh or control variate technics.

        :type fx: :class:`numpy.ndarray`, optional
        :param all: If ``True``, all points are selected. Defaults to ``False``.
        :type all: :class:`bool`, optional
        :param norm_: a string to identify the norm used for selection. Can be "frobenius" or "classifier".

            - if "frobenius", $d(X,Y) = \|f(X)-f_{k,\\theta}(X)\|_{\ell2}^2$
            - if "classifier", $d(X,Y) = \|\softmax(f(X))-\softmax(f_{k,\\theta}(X))\|_{\ell_2}^2$ to account for probabilities representation.
            - user-defined functions coming soon.

        :type norm_: :class:`str`, optional
        :param start_indices: an array of indices to set $Y^0$, otherwise the first is chosen randomly.
        :type start_indices: :class:`list`, optional

        :returns: Indices of the selected points.
        :rtype: :class:`numpy.ndarray`

        """
        # Set N to the max_nystrom if it is not specified
        if N is None:
            N = self.max_nystrom

        # Set input data and function values (fx) in the internal state of the object
        if x is not None:
            self.set_x(x)
        else:
            x = self.get_x()
        if fx is not None:
            self.set_fx(fx)
        self.rescale()
        theta = None
        if fx is not None:
            # Apply hybrid greedy Nystrom with error between ||f .-f_{k,\theta}||_A and  wrt a given norm
            # udefined by user
            # to compute Y
            theta, indices = Alg.hybrid_greedy_nystroem(
                x=self.get_x(),
                fx=fx,
                N=N,
                tol=0.0,
                error_type=norm,
                n_batch=n_batch,
                **kwargs,
            )
        else:
            indices = list(Alg.greedy_algorithm(x=self.get_x(), N=N, **kwargs))
            #
        self.indices = indices
        if all is True:
            # if there is a flag all, then
            # f_\theta(.) = K(.,Y)K(X,Y)^{-1}f(X)
            self.y = self.x[indices]
            # self.rescale()
        else:
            # else X = Y is set:
            #  f_\theta(.) = K(.,Y)K(Y,Y)^{-1}f(Y)
            self.kernel = core.KerInterface.get_kernel_ptr()
            self.x = self.x[indices]
            self.y = self.x
            if self.fx is not None:
                self.fx = self.fx[indices]
            # test = self.get_theta()-theta
            self.set_theta(theta)
        return self

    def set(
        self,
        x: np.ndarray = None,
        fx: np.ndarray = None,
        y: np.ndarray = None,
        theta: np.ndarray = None,
        **kwargs,
    ) -> None:
        """
        Set the input data ``x``, function values ``fx``, and target data ``y`` for the kernel.

        :param x: Input data points.
        :type x: :class:`numpy.ndarray`
        :param fx: Function values corresponding to the input data ``x``.
        :type fx: :class:`numpy.ndarray`, optional
        :param y: Target data points. If None, ``y`` is set equal to ``x``.
        :type y: :class:`numpy.ndarray`, optional
        """
        if x is None and fx is None:
            self.x, self.fx = None, None
            return
        if x is not None and fx is None:
            self.set_x(core.get_matrix(x.copy(),dtype=x.dtype), **kwargs)
            self.set_y(y=y)
            self.set_fx(None)
            # self.rescale() # rescaling already done in set_x()

        if x is None and fx is not None:
            if self.get_kernel() is None:
                raise Exception("Please input x at least once")
            if fx.shape[0] != self.get_x().shape[0]:
                raise Exception(
                    "fx of size "
                    + str(fx.shape[0])
                    + "must have the same size as x"
                    + str(self.x.shape[0])
                )
            self.set_fx(core.get_matrix(fx))

        if x is not None and fx is not None:
            (
                self.set_x(x, **kwargs),
                self.set_fx(fx, **kwargs),
                self.set_y(y=y, **kwargs),
            )
            # self.rescale(**kwargs) # done is set_x(..)
            pass
        if theta is not None:
            self.set_theta(theta, **kwargs)
        if not hasattr(self, "x"):
            return self
        if self.n_batch is None or self.n_batch >= self.get_x().shape[0]:
            return self
        self.N = int(self.x.shape[0] / self.n_batch + 1)
        self.clustering = self.set_clustering(
            x=self.get_x(), N=self.N, fx=self.get_fx(), **kwargs
        )

        y, temp_labels = self.clustering.cluster_centers_, self.clustering.labels_
        labels= {}
        def helper(k,v) : 
            labels[k]= set([v]) 
        [helper(k,v) for k, v in enumerate(temp_labels)]
        self.set_y(y)
        self.labels = map_invertion(labels,type_in=dict[int, set[int]])
        fx_proj = self.get_fx() - self(z=x)
        for key in self.labels.keys():
            indices = list(self.labels[key])
            if len(indices) > self.n_batch:
                N = int(len(indices) / self.n_batch) + 1
                self.kernels[key] = Kernel(
                    x=x[indices],
                    fx=fx_proj[indices],
                    n_batch=self.n_batch,
                    clustering=self.clustering,
                    **kwargs,
                )
            else:
                self.kernels[key] = Kernel(x=x[indices], fx=fx_proj[indices], **kwargs)
        # test = fx - self.__call__(z=x) # reproductibility : should be zero if no regularization
        return self

    def map(
        self,
        y: np.ndarray,
        distance: str = "norm22",
        x: np.ndarray = None,
        sub: bool = False,
        **kwargs,
    ) -> None:
        r"""
        Maps the input data points ``x`` to the target data points ``y`` using the kernel and optimal transport techniques.

        :param x: Input data points ($N$, $D_{source}$).
        :type x: :class:`numpy.ndarray`
        :param y: Target data points ($M$, $D_{target}$).
        :type y: :class:`numpy.ndarray`
        :param distance: Distance metric to use in mapping. Defaults to ``None``.
        :type distance: :class:`str`, optional
        :param sub: Whether to apply a sub-permutation. Defaults to False.
        :type sub: :class:`bool`, optional

        :returns: ``None``

        Example:

            >>> x_data = np.array([...])  # Input data with shape (N, D_source)
            >>> y_data = np.array([...])  # Target data with shape (M, D_target)
            >>> kernel.map(x_data, y_data)

        Note:

           This method computes a permutation that maps $x$ to $y$ using the Linear Sum Assignment Problem (LSAP) or a descent method.

            - If the dimensionalities of $x$ and $y$ are the same (:math:`D_{source} = D_{target}`), the classical LSAP algorithm is used.
            - If the dimensionalities differ (:math:`D_{source} \neq D_{target}`), a descent-based method is used to encode the data into a lower-dimensional latent space  before finding the optimal permutation, following principles of discrete optimal transport.
            - This permutation can be used to transform the input data $x$ to approximate the target data $y$.
        """
        # Set the internal state with input data points `x` and function values `y`
        if x is None:
            x = self.get_x()
        else:
            self.set_x(x)
            self.rescale()
        self.set_fx(y,**kwargs)
        # Rescale the input data `x` using the current kernel configuration

        # Check if the dimensionality of `x` and `y` is the same
        if x.shape[1] != y.shape[1]:
            # If the dimensionalities differ, use an encoder to map data into latent space
            # and find the optimal permutation (descent-based method)
            self.set_kernel_ptr()
            Dx = self.dnm(distance=distance)
            Dy = Kernel(x=self.get_fx()).dnm(distance=distance)
            self.permutation = np.array(Gromov_Monge(Dx,Dy,**kwargs))

            # Update `fx` based on the computed permutation
            # Kernel.set_fx(self,fx=y[self.permutation])
            self.fx = self.get_fx()[self.permutation] # not use, set_fx can be overloaded
            # self.permutation = map_invertion(np.array(self.permutation))
            # self.set_x(self.get_x()[self.permutation])
        else:
            # If the d imensionalities are the same, use the LSAP algorithm to compute the permutation
            D = core.KerOp.dnm(
                x=self.get_fx(), y=self.get_x(), distance=distance, kernel_ptr=self.get_kernel()
            )
            self.permutation = lsap(D, bool(sub))  # Solve LSAP to find permutation
            # Update `x` based on the computed permutation (lsap uses different conventions than alg.encoder #to fix )
            self.set_fx(self.get_fx()[self.permutation])
            # self.set_x(self.get_x()[map_invertion(np.array(self.permutation))])
        return self

    def __len__(self) -> int:
        """
        Return the number of input data points ``x``.

        :returns: The number of data points in ``x`` or 0 if ``x`` is not set.
        :rtype: :class:`int`
        """
        if self.x is None:
            return 0
        return self.x.shape[0]

    def copy(self, kernel) -> None:
        def helper(b):
            if b is not None:
                return b.copy()
            else:
                return None

        self.x, self.y, self.theta, self.fx = (
            helper(kernel.x),
            helper(kernel.y),
            helper(kernel.theta),
            helper(kernel.fx),
        )
        self.knm_, self.knm_inv = helper(kernel.knm_), helper(kernel.knm_inv)
        self.kernel = kernel.kernel

    def update(
        self, z: np.ndarray, fz: np.ndarray, eps: float = None, **kwargs
    ) -> None:
        """
        Fit the regressor to new data points ``(z, fz)`` while maintaining the existing kernel structure.

        This method allows fitting a kernel-based regressor that is originally defined on the set ``x`` but
        is updated to match new input values ``z`` and their corresponding function values ``fz``.

        The regression is defined by the formula:

        $$
        f_{k, \\theta}(z) \\approx K(z, X)\\theta = f(z)
        $$

        Where the coefficient `\\theta` is computed as:

        $$
        \\theta = K(z, X)^{-1}f(z)
        $$

        :param z: New input data points to update the regressor.
        :type z: :class:`numpy.ndarray`
        :param fz: Function values corresponding to the new data points `z`.
        :type fz: :class:`numpy.ndarray`
        :param eps: Regularization parameter used in the least squares solution. Defaults to `self.reg` if not provided.
        :type eps: :class:`float`, optional

        :returns: Updates the internal state of the regressor with new `z` and `fz` values.
        :rtype: ``None``
        """
        self.set_kernel_ptr()
        if isinstance(z, list):
            return [self.update(z_, fz_, **kwargs) for z_, fz_ in zip(z, fz)]
        z = core.get_matrix(z)
        if self.x is None:
            return None

        # Compute the kernel matrix `K(z, X)` where `X` is the current input dataset
        knm = core.KerOp.knm(x=z, y=self.get_y())
        # At this point, we are trying to solve the following least squares problem:
        #
        # $$ \theta = \text{argmin} \| K(z, X)\theta - f(z) \|^2 + \text{reg} \|\theta\|^2 $$
        #
        # `eps` (regularization) controls the trade-off between fitting the data and controlling the magnitude of `theta`.

        if eps is None:
            eps = self.reg
        # if self.theta is not None: fzz += eps*LAlg.prod(knm,self.theta)
        # Solve the least-squares problem with regularization:
        # $$ \theta = \left( K(z, X)^\top K(z, X) + \text{eps} \cdot I \right)^{-1} K(z, X)^\top fzz $$
        self.set_theta(LAlg.lstsq(knm, fz, eps=eps))

        return self

    def add(
        self, y: np.ndarray = None, fy: np.ndarray = None, kernel_ptr=None, min_distance = None, **kwargs
    ) -> None:
        """
        Augments the training set by adding new data points and their corresponding function values.

        This method optimizes the computation for training set augmentation by efficiently updating
        the kernel matrix and applying a block-inversion algorithm, which reduces the overall complexity
        compared to recalculating the full kernel matrix.

        :param y: New data points to be added to the training set.
        :type y: :class:`numpy.ndarray`
        :param fy: Function values corresponding to the new data points `y`.
        :type fy: :class:`numpy.ndarray`

        :returns: This method updates the internal state of the class, modifying the training set with the new data points and their function values.
        :rtype: ``None``

        Note:
            The kernel matrix $K([X,Y], [X,Y])$ is of size $\\mathbb{R}^{(N_X+N_Y) \\times (N_X+N_Y)}$,
            and directly computing its inverse has a complexity of $(N_X + N_Y)^3$.

            By using the block-inversion method, the complexity can be reduced to $N_X^3 + N_Y^3$, significantly improving performance.

            The function $f_{k,\\theta}(.)$ is then computed as:

            $$
            f_{k,\\theta}(.) = K(., [X,Y])\\theta, \\quad \\theta = K([X,Y], [X,Y])^{-1} \\begin{bmatrix} f(X) \\; f(Y) \\end{bmatrix}
            $$

            Here, $[.]$ denotes standard matrix concatenation, where $f(X)$ and $f(Y)$ are the function values for the original and new data points, respectively.
        """
        x, fx = core.get_matrix(y.copy()), core.get_matrix(fy.copy())
        if min_distance is not None:
            distances = np.sqrt(core.KerOp.dnm(x,self.get_x(), kernel_ptr=self.get_kernel()))
            mask = np.where(distances.min(1) > min_distance,True,False)
            x,fx = core.get_matrix(y[mask].copy()), core.get_matrix(fy[mask].copy())
            if len(x) == 0 : 
                return self
        if not hasattr(self, "x") or self.x is None:
            self.set(x, fx)
            return
        

        # the method add computes an updated Gram matrix using the already
        # pre-computed Gram matrix K(x,x).
        knm_, knm_inv, y = Alg.add(
            self.get_knm(),
            self.get_knm_inv(),
            self.get_x(),
            x,
            self.get_kernel(),
            self.order,
            0.0,
        )
        if fx is not None and self.get_fx() is not None:
            self.set_fx(np.concatenate([self.get_fx(), fx], axis=0))
        else:
            self.set_fx(fx)
        new_x = np.concatenate([self.get_x(), x], axis=0)
        self.set_x(new_x, rescale=False)
        self.set_y(new_x, rescale=False)
        self.knm_, self.knm_inv = knm_, knm_inv

        return self

    def kernel_distance(self, y: np.ndarray, x=None) -> np.ndarray:
        """
        Compute a MMD-like (Maximum Mean Discrepancy) based distance matrix between the input data ``x`` and the new data ``z``.

        The distance is computed as:

        $$
        D(X,Z) = \Big(d_k(x^i,z^j) \Big)_{i,j},\quad d_k(x,y)= k(x,x) + k(z,z)-2k(x,z)
        $$

        :param z: New input data points.
        :type z: :class:`numpy.ndarray`

        :returns: The computed MMD-based distance matrix.
        :rtype: :class:`numpy.ndarray`
        """
        self.set_kernel_ptr()
        if x is None:
            x = self.x
        return core.KerOp.dnm(x=y, y=x)

    def discrepancy(self, z: np.ndarray) -> float:
        """
        Compute the MMD (Maximum Mean Discrepancy) between the kernel features $x$ and $z$.

        :param z: New input data points.
        :type z: :class:`numpy.ndarray`

        :returns: The computed MMD-based distance matrix.
        :rtype: :class:`numpy.ndarray`
        """
        self.set_kernel_ptr()
        return core.KerOp.discrepancy_error(x=self.get_x(), z=z)

    def set_kernel(self, kernel_ptr) -> callable:
        self.kernel = kernel_ptr

    def get_kernel(self) -> callable:
        """
        Retrieve the current kernel function for the input data.

        This method retrieves a positive semi-definite (PSD) kernel function,
        represented as: $k(S(x), S(y))$, where $S$ is a predefined mapping.

        :returns: The kernel function used by the current model.
        :rtype: callable
        """
        if not hasattr(self, "kernel"):
            self.kernel = self.set_kernel(polynomial_order=self.order)
            # self.order= None
            self.kernel = core.KerInterface.get_kernel_ptr()
        return self.kernel

    def set_kernel_ptr(self, kernel_ptr=None) -> None:
        """
        Set the Codpy interface to use the current kernel function.

        This method updates the Codpy kernel interface with the current kernel
        function, sets the polynomial order+1 (according to the C++ core convention), and applies the regularization
        parameter defined in the object.
        """
        if self.order is None:
            order = 0
        else:
            order = self.order + 1
        if kernel_ptr is not None:
            self.kernel = kernel_ptr
        core.KerInterface.set_kernel_ptr(self.get_kernel(), order, self.reg)

    def set_map(self, map_) -> callable:
        self.map_ = map_
        return self

    def get_map(self) -> callable:
        """
        Retrieve the current mapping function for the input data.
        :returns: The mapping function used by the current model.
        :rtype: callable
        """
        if not hasattr(self, "map_"):
            self.map_ = None
        return self.map_

    def rescale(self) -> None:
        """
        Rescale the input data using the current mapping.

        This method rescales the input data by applying the map function associated
        with the current kernel. It also retrieves and updates the internal kernel
        function based on the rescaled data.

        If ``x`` is set, the rescaling is applied to ``x`` with a maximum number of
        points defined by ``max_nystrom``.
        """
        if self.get_x() is not None:
            # instructs to set the map parameter
            # applied to the data
            if self.get_map() is not None:
                core.KerInterface.rescale(
                    self.get_map()(self.get_x()),
                    max=self.max_nystrom,
                    kernel_ptr=self.get_kernel(),
                )
            else:
                core.KerInterface.rescale(
                    self.get_x(),
                    max=self.max_nystrom,
                    kernel_ptr=self.get_kernel(),
                    order=self.order,
                    reg=self.reg,
                )
            # retrives the kernel
            self.kernel = core.KerInterface.get_kernel_ptr()
            self.set_theta(None)
            self.knm_inv, self.knm_ = None, None
            self.valid = True

    def __call__(self, z: np.ndarray=None,theta=None,second_member=None, **kwargs) -> np.ndarray:
        """
        Predict the output using the kernel for input data ``z``.

        :param z: Input data points for prediction.
        :type z: :class:`numpy.ndarray`

        :returns: The predicted values based on the kernel and function values.
        :rtype: :class:`numpy.ndarray`

        Example:
            >>> z_data = np.array([...])
            >>> prediction = kernel(z_data)

        Note:
            This function is similar to ``predict`` in libraries like scikit-learn or XGBoost.

            - If ``fx`` is defined, the prediction is given by the formula $f_{k, \\theta}(z)$.
            - If ``fx`` is not defined, the function returns the projection operator:

            $$P_{k,\\theta}(z) = K(Z, K) K(X, X)^{-1}$$
        """
        if z is None:
            if hasattr(self,"knm_z_"): 
                if theta is None: return self.knm_z_
                return LAlg.prod(self.knm_z_,theta)
            z = self.get_x()
        else : z = core.get_matrix(z)

        # Don't forget to set the kernel
        if theta is None:
            theta = self.get_theta(**kwargs)

        if theta is None:
            theta = self.get_knm_inv()

        self.knm_z_ = core.KerOp.knm(
            x=z,
            y=self.get_y(),
            fy=None,
            kernel_ptr=self.get_kernel(),
            order=self.order,
            reg=self.reg,
        )

        if not hasattr(self, "set_clustering"):
            if theta is not None: return LAlg.prod(self.knm_z_,theta)
            return self.knm_z_
        if len(self.kernels) == 0:
            if theta is not None: return LAlg.prod(self.knm_z_,theta)
            return self.knm_z_
        mapped_indices = self.clustering(z)
        mapped_indices = cast(mapped_indices,type_in=np.ndarray,type_out=dict[int, set[int]])
        mapped_indices = map_invertion(mapped_indices,type_in=dict[int, set[int]])
        for key in mapped_indices.keys():
            indices = list(mapped_indices[key])
            self.knm_z_[indices] += self.kernels[key](z[indices])
        return self.knm_z_

    def grad(self, z: np.ndarray, **kwargs) -> np.ndarray:
        if z is None:
            return None
        z = core.get_matrix(z)

        # Don't forget to set the kernel
        theta = self.get_theta(**kwargs)

        if theta is None:
            theta = self.get_knm_inv(**kwargs)
        knm = core.DiffOps.nabla_knm(
            x=z, y=self.get_x(), theta=theta, order = self.order, reg = self.reg, kernel=self.get_kernel()
        )

        return knm

    def __and__(self, other):
        return BitwiseANDKernel(self, other)

class Sampler(Kernel):
    """
    an overload of the class Kernel to sample distributions from a latent space:
        :param x: Input distribution.
        :type x: :class:`numpy.ndarray`

        :param latent_generator: an optional generator. Defaulted to numpy.random.normal
    """
    def __init__(self, x, set_kernel=core.kernel_setter("maternnorm", "standardmean",0,1e-9), latent_dim=None,latent_generator=None,**kwargs):
        """
        Initializes the sampler with a kernel mapping object.

        Parameters:
            kernel (Kernel): A kernel-based mapper with a `.map()` method.
        """
        if latent_generator is None:
            if latent_dim is None:
                latent_dim = x.shape[1]
            # self.latent_generator = lambda n: get_uniforms(n, latent_dim,nmax=10) # FIX THIS!!!
            self.latent_generator = lambda n: get_qmc_normals(n, latent_dim,nmax=10)
        else:
            self.latent_generator = latent_generator
        y= self.latent_generator(x.shape[0])
        super().__init__(x=y,fx=None,set_kernel=set_kernel,**kwargs)
        # self.map(y=x,distance=None)
        self.map(y=x,**kwargs)

    def sample(self, N):
        """
        Generates samples using the learned structure.

        Parameters:
            N (int): Number of samples to generate.

        Returns:
            np.ndarray: Sampled data.
        """
        return self(z=self.latent_generator(N))



def get_tensor_probas(policy):
    """
        params:
            policy: array of shape (n,m) representing n probability distributions over m classes
        returns:
            tensor of shape (n,m,m) representing the gradient of the jacobian of the softmax function for each n
    """
    # @np.vectorize
    # def fun(i, j, k):
    #     return policy[i, j] * (float(j == k) - policy[i, k])

    # return np.fromfunction(
    #     fun, shape=[policy.shape[0], policy.shape[1], policy.shape[1]], dtype=int
    # )

    # Faster version
    # policy: shape (n, m)
    # Output: shape (n, m, m)
    n, m = policy.shape

    # Create diagonal matrices with p_ij on the diagonal for each i
    diag_p = np.einsum('ij,jk->ijk', policy, np.eye(m))

    # Outer product of each row with itself
    outer_p = np.einsum('ij,ik->ijk', policy, policy)

    return diag_p - outer_p


class KernelClassifier(Kernel):
    """
    A simple overload of the kernel :class:`Kernel` for proabability handling.
        Note:
            It overloads the prediction method as follows :

                $$\text{softmax} (\log(f)_{k,\\theta})(\cdot)$$
    """

    def set_fx(
        self,
        fx: np.ndarray,
        # clip=Alg.proportional_fitting,
        clip=None,
        **kwargs,
    ) -> None:
        if fx is None:
            fx = np.identity(self.get_x().shape[0])
        else:
            fx = core.get_matrix(fx,dtype=fx.dtype)
        if clip is not None and fx is not None:
            fx = clip(fx)
        fx = np.where(fx < 1e-9, 1e-9, fx) 
        fx = fx / fx.sum(axis=1, keepdims=True) 
        if fx is not None:
            fx = np.log(fx)
        super().set_fx(fx, **kwargs)

    def __call__(self, z = None, **kwargs):
        if self.x is None:
            return None
        knm = super().__call__(z, **kwargs)
        return softmax(knm, axis=1)

    def greedy_select(
        self, N, x=None, fx=None, all=False, norm_="classifier", **kwargs
    ):
        return super().greedy_select(N=N, x=x, fx=fx, all=all, norm_=norm_, **kwargs)

    def grad(self, z: np.ndarray, **kwargs) -> np.ndarray:
        out = super().grad(z)
        if out is None:
            return None
        out = LAlg.prod_vector_matrix(
            out, get_tensor_probas(softmax(self.get_fx(), axis=1))
        )
        return out

    def update(
        self, z: np.ndarray, fz: np.ndarray, eps: float = None, **kwargs
    ) -> None:
        return super().update(z=z, fz=np.log(fz), eps=eps, **kwargs)

class se_error_theta:
    def __init__(self,fun,x,y,theta=None,**kwargs):
        self.fun = fun
        self.x= x
        self.y= y
        if theta is not None:
            self.theta = theta
    def error_field(self,theta=None,z=None,**kwargs):
        if theta is None:
            theta = self.fun.get_theta(**kwargs)
        debug = self.fun(z=z,theta=theta,**kwargs)
        return debug-self.y
    def error(self,theta,**kwargs):
        error = self.error_field(theta,**kwargs)
        return (error*error).sum()*.5
    def grad_error(self,theta,**kwargs):
        return self.fun.grad_theta(second_member=self.error_field(theta,**kwargs),theta=theta,**kwargs)
    def __call__(self,theta,**kwargs):
        return  algs.Alg.gradient_descent(theta,fun=self.error,constraints=None,grad_fun=self.grad_error,**kwargs)

class SparseKernel(Kernel):
    def __init__(self,x,k=10,**kwargs):
        self.k= k
        super().__init__(x=x,**kwargs)
        pass
    def error_field(self,theta,**kwargs):
        if theta.ndim == 1:
            theta= theta.reshape(self.get_fx().shape).astype(self.get_fx().dtype)
        out = self(theta=theta,**kwargs)-self.get_fx()
        return out
    def error(self,theta,second_member=None,**kwargs):
        out = self.error_field(theta=theta,**kwargs)
        return (out*out).sum()*.5

    def grad(self,z=None,k=None,theta=None,second_member=None,**kwargs):
        if k is None: k=self.k
        if z is None: z= self.get_x()
        knm = self.grad_knm(z, self.get_y(),k=k,**kwargs)
        if theta is not None:
            result  = sparse_dot_mkl.dot_product_mkl(knm,theta)
        else:
            result  = sparse_dot_mkl.dot_product_mkl(knm,self.get_theta())
        if second_member is not None:
            result  = LAlg.prod(result,second_member)
        return result.reshape(z.shape[0],z.shape[1],-1) 
    
    def grad_theta(self,theta,dtype=np.float64,second_member=None,**kwargs):
        knm = self.get_knm(**kwargs)
        if second_member is None:
            second_member = self.error_field(theta=theta,**kwargs)
        out = sparse_dot_mkl.dot_product_mkl(knm,second_member)
        if theta.ndim == 1:
            return out.reshape(theta.shape).astype(dtype)
        else:
            return out
    def get_theta(self,method="bfgs", **kwargs) -> np.ndarray:
        if not hasattr(self, "theta") or self.theta is None:
            if method == "bfgs":
                theta= self.get_fx().flatten().astype(np.float64)
                print("error beg: ",self.error(theta,**kwargs))
                out,fmin,infos = scipy.optimize.fmin_l_bfgs_b(func=self.error,x0=theta,fprime=self.grad_theta)
                print("error end: ",fmin, "infos",infos)
                self.theta = out.astype(self.x.dtype).reshape(self.get_fx().shape)
            else:
                se_error_instance = se_error_theta(self, self.get_x(), self.get_fx())
                self.theta =  se_error_instance(self.fx,**kwargs)
        return self.theta
    def __call__(self,z=None,theta=None,second_member=None,**kwargs):
        if z is None: knm = self.get_knm(**kwargs)
        else: knm = self.knm(z.astype(self.get_y().dtype), **kwargs)
        if theta is None:
            theta = self.get_theta(**kwargs)
        result  = sparse_dot_mkl.dot_product_mkl(knm,theta)
        if second_member is not None:
            return LAlg.prod(result,second_member)
        return result
    def get_index(self,**kwargs):
        if not hasattr(self,"index"):
            self.index = algs.Alg.faiss_knn_index(x=self.get_x(), **kwargs)
        return self.index

    def knm(
        self, z: np.ndarray = None, y: np.ndarray = None, fy: np.ndarray = None, **kwargs
    ) -> np.ndarray:
        assert y is None," Sparse kernel can't estimate the Gram matrix outside of the training set."
        index_x = self.get_index(**kwargs)
        if z is None: 
            z=self.get_x()
            Sx,_ = algs.Alg.faiss_knn(z=z,fun=None, metric="cosine",index=index_x,**kwargs)
        else:
            Sx,_ = algs.Alg.faiss_knn(z=z,fun=None, metric="cosine",index=index_x,**kwargs)
            # index_z = algs.Alg.faiss_knn_index(x=z, **kwargs)
            # Sz,_ = algs.Alg.faiss_knn(z=self.get_x(),fun=None, metric="cosine",index=index_z,**kwargs)
            # Sx = (Sx+Sz.T)*.5

        if fy is not None:
            return sparse_dot_mkl.dot_product_mkl(Sx,fy) 
        return Sx
    
    def grad_knm(self, x=None, z=None, k=None,**kwargs) :
        if k is None: k = self.k
        if x is None: x = self.get_x()
        if z is None: z = self.get_y()
        out = algs.Alg.grad_faiss_knn(x, z,k=k,fun=None, metric="cosine",**kwargs)
        # out = (out + algs.grad_faiss_knn(z, x,k=k,fun=None, metric="cosine",**kwargs))*.5
        return out
    def get_knm(self, **kwargs) -> np.ndarray:
        if not hasattr(self, "knm_") or self.knm_ is None:
            self.knm_ = self.knm(z=self.get_x(),**kwargs)
        return self.knm_     

class SparseKernelClassifier(SparseKernel):
 
    def __call__(self,z=None,theta=None,second_member=None,**kwargs):
        out  = softmax(super().__call__(z=z,theta=theta,second_member=None),axis=1)
        if second_member is not None:
            return LAlg.prod(out,second_member)
        return out
    def error_field(self,theta,**kwargs):
        if theta.ndim == 1:
            theta= theta.reshape(self.get_fx().shape).astype(self.get_fx().dtype)
        out = self(theta=theta,**kwargs)-softmax(self.get_fx(),axis=1)
        return out
    def error(self,theta,second_member=None,**kwargs):
        out = self.error_field(theta=theta,**kwargs)
        return (out*out).sum()*.5
        
    def get_theta(self,method="toto", **kwargs) -> np.ndarray:
        if not hasattr(self, "theta") or self.theta is None:
            if method == "bfgs":
                theta= self.get_fx().flatten().astype(np.float64)
                print("error beg: ",self.error(theta,**kwargs))
                out,fmin,infos = scipy.optimize.fmin_l_bfgs_b(func=self.error,x0=theta,fprime=self.grad_theta)
                print("error end: ",fmin, "infos",infos)
                self.theta = out.astype(self.x.dtype).reshape(self.get_fx().shape)
            else:
                se_error_instance = se_error_theta(self, self.get_x(), softmax(self.get_fx(),axis=1))
                self.theta =  se_error_instance(self.fx,**kwargs)
        return self.theta
      
    def set_fx(
        self,
        fx: np.ndarray,
        clip=None,
        **kwargs,
    ) -> None:
        if fx is None:
            self.fx = None
            return
        else:
            fx = core.get_matrix(fx,dtype=fx.dtype)
        if clip is not None and fx is not None:
            fx = clip(fx)
        fx = np.where(fx < 1e-9, 1e-9, fx) 
        fx = fx / fx.sum(axis=1, keepdims=True) 
        if fx is not None:
            fx = np.log(fx)
        super().set_fx(fx, **kwargs)
    
    def grad_theta(self,theta,dtype=np.float64,second_member=None,**kwargs):
        knm = self.get_knm(**kwargs)
        shape = None
        if theta.ndim == 1:
            shape=theta.shape
            theta= theta.reshape(self.get_fx().shape).astype(self.get_fx().dtype)
        if second_member is None:
            second_member = self.error_field(theta=theta,**kwargs)
        policy = softmax(sparse_dot_mkl.dot_product_mkl(knm, theta), axis=1)  # (n, m)
        # Jacobian of softmax: diag(pi) - pi*pi^T for each sample
        policy_grad = get_tensor_probas(policy).astype(knm.dtype)  # (n, m, m)
        
        # Compute J_pi^T @ e_theta for each sample
        # policy_grad is (256, 10, 10), second_member is (256, 10)
        # We need: for each i (sample), policy_grad[i].T @ second_member[i] 
        # == (10,10) @ (10,) -> (10,)
        
        # Transpose the Jacobians: (256, 10, 10) -> (256, 10, 10) : NEEDED????
        # policy_grad_T = np.transpose(policy_grad, (0, 2, 1))
        
        # Apply J^T to e_theta: (256, 10, 10) @ (256, 10, 1) -> (256, 10)
        j_times_e = np.einsum('nij,nj->ni', policy_grad, second_member)  # (256, 10)
        # Multiply by k^T: (256, 256)^T @ (256, 10) -> (256, 10)
        grad = sparse_dot_mkl.dot_product_mkl(knm.T, j_times_e)
            
        if shape is not None:
            return grad.reshape(shape).astype(dtype)
        else:
            return grad
        
from codpy.kengineering import *

if __name__ == "__main__":
    codpy.core.KerInterface.set_verbose()
    # test derivative with polynomial regression
    # test = Kernel(x=get_matrix([[0.,1.,2.]]).T,fx=get_matrix([[1.,3.,5.]]).T, order=1)
    # print(test(z=[0.,1.,2.,3]))
    # print(test.grad(z=0))

    # test derivative of classifier, that regressor to probabilites
    y = np.random.rand(100, 10)
    x = np.random.rand(100, 15)
    k = KernelClassifier(x=x, fx=y)
    print(k(z=np.random.rand(2, 15)))
    print(k.grad(z=np.random.rand(2, 15)))
    pass
