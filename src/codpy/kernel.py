from typing import Tuple

import numpy as np
from codpydll import *
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

import codpy.core as core
from codpy.algs import alg
from codpy.core import diffops
from codpy.lalg import lalg as lalg
from codpy.permutation import lsap


class Kernel:
    """
    A kernel class to perform various kernel-based operations, such as mapping, transformations.
    """

    def __init__(
        self,
        max_pool: int = 1000,
        max_nystrom: int = 1000,
        reg: float = 1e-9,
        order: int = None,
        dim: int = 1,
        set_kernel: callable = None,
        **kwargs: dict,
    ) -> None:
        """
        Initializes the Kernel class with default or user-defined parameters and sets up the kernel function.

        :param max_pool: Maximum pool size for the kernel operations. Defaults to 1000.
        :type max_pool: :class:`int`, optional
        :param max_nystrom: Maximum number of Nystrom samples. Defaults to 1000.
        :type max_nystrom: :class:`int`, optional
        :param reg: Regularization parameter for kernel operations. Defaults to 1e-9.
        :type reg: :class:`float`, optional
        :param order: Polynomial order for polynomial kernel functions. Defaults to ``None``.
        :type order: :class:`int`, optional
        :param dim: Dimensionality of the input data. Defaults to 1.
        :type dim: :class:`int`, optional
        :param set_kernel: A custom kernel function initializer. If not provided, a default kernel is used.
        :type set_kernel: :class:`callable`, optional
        :param kwargs: Additional keyword arguments for further customization.
        :type kwargs: :class:`dict`
        """
        self.dim = dim
        self.order = order
        self.reg = reg
        self.max_pool = int(max_pool)
        self.max_nystrom = int(max_nystrom)

        if set_kernel is not None:
            self.set_kernel = set_kernel
        else:
            self.set_kernel = self.default_kernel_functor()
        self.x = None
        if kwargs.get("x", None) is not None or kwargs.get("fx", None) is not None:
            self.set(**kwargs)

    def default_kernel_functor(self) -> callable:
        """
        Initialize and return the default kernel function.

        This method provides the default kernel initialization using the `maternnorm` kernel
        with the `standardmean` map. It sets a polynomial order of 0 and a regularization value of 1e-9.

        :returns: The initialized default kernel function using :func:`core.kernel_setter`.
        :rtype: :class:`callable`

        Example:
            >>> default_kernel = kernel.default_kernel_functor()
        """
        return core.kernel_setter("maternnorm", "standardmean", 0, 1e-9)

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
        Set a custom kernel using `core.kernel_helper2` with flexible parameters.

        :param kernel_name: Name of the kernel function to use (e.g., ``'gaussian'``).
        :type kernel_name: :class:`str`
        :param map_name: Name of the mapping function (e.g., ``'standardmean'``).
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

    def _set_polynomial_regressor(
        self, x: np.ndarray = None, fx: np.ndarray = None, **kwargs
    ) -> None:
        """
        Set up the polynomial regressor for the input data ``x`` and function values ``fx``.

        This method fits a polynomial regression model based on the current polynomial order.
        If the order is not defined, or if either ``x`` or ``fx`` is not provided, the polynomial
        variables, kernel, and values are set to ``None``.

        :param x: Input data points.
        :type x: :class:`numpy.ndarray`, optional
        :param fx: Function values corresponding to the input data ``x``.
        :type fx: :class:`numpy.ndarray`, optional
        :param kwargs: Additional keyword arguments for flexibility (not used directly).

        Note:
            - The polynomial order is retrieved using :meth:`get_order`.
            - If the polynomial order, ``x``, or ``fx`` are not provided, the internal polynomial attributes
            (``polyvariables``, ``polynomial_kernel``, and ``polynomial_values``) are reset to ``None``.

        Example:
            >>> kernel._set_polynomial_regressor(x_data, fx_data)
        """
        if x is None or fx is None or self.get_order() is None:
            self.polyvariables, self.polynomial_kernel, self.polynomial_values = (
                None,
                None,
                None,
            )
            return
        order = self.get_order()
        if order is not None and fx is not None and x is not None:
            self.polyvariables = PolynomialFeatures(order).fit_transform(x)
            self.polynomial_kernel = linear_model.LinearRegression().fit(
                self.polyvariables, fx
            )
            self.polynomial_values = self.polynomial_kernel.predict(self.polyvariables)
            self.set_theta(None)

    def get_polynomial_values(self, **kwargs) -> np.ndarray:
        """
        Retrieve the predicted polynomial values based on the current input data.

        This method returns the values obtained from the polynomial regression model.
        If the polynomial values are not yet computed, it calls :meth:`_set_polynomial_regressor`
        to set up the polynomial regressor using the current input data ``x`` and function values ``fx``.

        :param kwargs: Additional keyword arguments for flexibility (not used directly).

        :returns: The predicted polynomial values or ``None`` if the polynomial order is not set.
        :rtype: :class:`numpy.ndarray` or :class:`None`

        Example:
            >>> poly_values = kernel.get_polynomial_values()
        """
        if self.get_order() is None:
            return None
        if not hasattr(self, "polynomial_values") or self.polynomial_values is None:
            self._set_polynomial_regressor(self.get_x(), self.get_fx())
        return self.polynomial_values

    def _get_polyvariables(self, **kwargs) -> np.ndarray:
        """
        Retrieve the polynomial variables transformed from the input data.

        This method returns the polynomial features (variables) created based on the polynomial order.
        If the polynomial variables are not yet set, it calls :meth:`_set_polynomial_regressor` to
        fit the polynomial features using the current input data ``x`` and function values ``fx``.

        :param kwargs: Additional keyword arguments for flexibility (not used directly).

        :returns: The polynomial variables transformed from the input data or ``None`` if the polynomial order is not set.
        :rtype: :class:`numpy.ndarray` or :class:`None`

        Example:
            >>> poly_vars = kernel._get_polyvariables()
        """
        if self.get_order() is None:
            return None
        if not hasattr(self, "polyvariables") or self.polyvariables is None:
            self._set_polynomial_regressor(self.get_x(), self.get_fx())
        return self.polyvariables

    def _get_polynomial_kernel(self, **kwargs) -> linear_model.LinearRegression:
        """
        Retrieve the polynomial kernel (regression model) used for fitting the polynomial features.

        This method returns the polynomial kernel (linear regression model) fitted on the input data.
        If the polynomial kernel is not yet set, it calls :meth:`_set_polynomial_regressor` to
        fit the polynomial regression model using the current input data ``x`` and function values ``fx``.

        :param kwargs: Additional keyword arguments for flexibility (not used directly).

        :returns: The polynomial kernel (linear regression model) or ``None`` if the polynomial order is not set.
        :rtype: :class:`linear_model.LinearRegression` or :class:`None`

        Example:
            >>> poly_kernel = kernel._get_polynomial_kernel()
        """
        if self.get_order() is None:
            return None
        if not hasattr(self, "polynomial_kernel") or self.polynomial_kernel is None:
            self._set_polynomial_regressor(self.get_x(), self.get_fx())
        return self.polynomial_kernel

    def get_polynomial_regressor(
        self, z: np.ndarray, x: np.ndarray = None, fx: np.ndarray = None, **kwargs
    ) -> np.ndarray:
        """
        Set up the polynomial regressor based on the input data and the polynomial order.

        :param z: New input data points for the regressor.
        :type z: :class:`numpy.ndarray`
        :param x: Input data points.
        :type x: :class:`numpy.ndarray`, optional
        :param fx: Function values corresponding to the input data.
        :type fx: :class:`numpy.ndarray`, optional

        :returns: The predicted polynomial values or `None` if unavailable.
        :rtype: :class:`numpy.ndarray` or :class:`None`

        Example:
            >>> z_data = np.random.rand(100, 10)
            >>> pred = kernel.get_polynomial_regressor(z_data)
        """
        if self.get_order() is None:
            return None
        if x is None:
            polyvariables = self._get_polyvariables()
        else:
            polyvariables = PolynomialFeatures(self.order).fit_transform(x)
        if fx is None:
            polynomial_kernel = self._get_polynomial_kernel()
        else:
            polynomial_kernel = linear_model.LinearRegression().fit(polyvariables, fx)
        z_polyvariables = PolynomialFeatures(self.order).fit_transform(z)
        if polynomial_kernel is not None:
            return polynomial_kernel.predict(z_polyvariables)
        return None

    def Knm(
        self, x: np.ndarray, y: np.ndarray, fy: np.ndarray = [], **kwargs
    ) -> np.ndarray:
        """
        Compute the kernel matrix :math:`K(x, y)` using the current kernel settings.

        :param x: Input data points :math:`(N, D)`, where :math:`N` is the number of points and :math:`D` is the dimensionality.
        :type x: :class:`numpy.ndarray`
        :param y: Secondary data points :math:`(M, D)`, where :math:`M` is the number of points and :math:`D` is the dimensionality.
        :type y: :class:`numpy.ndarray`
        :param fy: Optional function values for kernel computation.
        :type fy: :class:`numpy.ndarray`, optional

        :returns: The computed kernel matrix :math:`K` of size :math:`(N, M)`.
        :rtype: :class:`numpy.ndarray`

        Example:
            >>> x_data = np.array([...])
            >>> y_data = np.array([...])
            >>> kernel_matrix = Kernel(x=x_data,y=y_data).Knm()
        """

        self.set_kernel_ptr()
        return core.op.Knm(x=x, y=y, fy=fy)

    def get_knm_inv(
        self, epsilon: float = None, epsilon_delta: np.ndarray = None, **kwargs
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
            epsilon = kwargs.get("epsilon", self.reg)
            epsilon_delta = kwargs.get("epsilon_delta", None)
            if epsilon_delta is None:
                epsilon_delta = []
            else:
                epsilon_delta = epsilon_delta * self.get_Delta()
            self._set_knm_inv(
                core.op.Knm_inv(
                    x=self.get_x(),
                    y=self.get_y(),
                    epsilon=epsilon,
                    reg_matrix=epsilon_delta,
                ),
                **kwargs,
            )
        return self.knm_inv

    def get_knm(self, **kwargs) -> np.ndarray:
        """
        Retrieve or compute the Gram matrix $K(x, y)$ for the kernel.

        :returns: The Gram matrix $K(x,y)$.
        :rtype: :class:`numpy.ndarray`
        """
        if not hasattr(self, "knm") or self.knm is None:
            self._set_knm(core.op.Knm(x=self.x, y=self.y))
        return self.knm

    def _set_knm_inv(self, k):
        self.knm_inv = k
        self.set_theta(None)

    def _set_knm(self, k):
        self.knm = k

    def get_x(self, **kwargs) -> np.ndarray:
        """
        Retrieve the input data ``x``.

        :returns: The input data or ``None`` if not set.
        :rtype: :class:`numpy.ndarray` or :class:`None`
        """
        if not hasattr(self, "x"):
            self.x = None
        return self.x

    def set_x(
        self, x: np.ndarray, set_polynomial_regressor: bool = True, **kwargs
    ) -> None:
        """
        Set the input data ``x`` for the kernel and update related internal states.

        This method sets the input data and optionally recalculates the polynomial regressor and kernel matrices.

        :param x: Input data points to be set.
        :type x: :class:`numpy.ndarray`
        :param set_polynomial_regressor: Whether to recalculate the polynomial regressor after setting the data.
                                        Defaults to ``True``.
        :type set_polynomial_regressor: :class:`bool`, optional
        """
        self.x = x.copy()
        self.set_y()
        if set_polynomial_regressor:
            self._set_polynomial_regressor()
        self._set_knm_inv(None)
        self._set_knm(None)
        self.rescale()

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
        return self.fx

    def set_fx(
        self, fx: np.ndarray, set_polynomial_regressor: bool = True, **kwargs
    ) -> None:
        """
        Set the function values ``fx`` for the input data.

        :param fx: Function values corresponding to the input data.
        :type fx: :class:`numpy.ndarray`
        :param set_polynomial_regressor: Whether to recalculate the polynomial regressor after setting the function values.
                                        Defaults to ``True``.
        :type set_polynomial_regressor: :class:`bool`, optional
        """
        if fx is not None:
            self.fx = fx.copy()
        else:
            self.fx = None
        if set_polynomial_regressor:
            self._set_polynomial_regressor()
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
        if theta is None:
            return
        self.fx = None
        # self.fx =  lalg.prod(self.get_knm(),self.theta)
        # if self.get_order() is not None :
        #     self.fx += self.get_polynomial_regressor(z=self.get_x())

    def get_theta(self, **kwargs) -> np.ndarray:
        """
        Retrieve the coefficient ``theta`` for kernel regression.

        If ``fx`` is not defined, the polynomial regressor is used to adjust the function values.

        :returns: The regression coefficient ``theta``.
        :rtype: :class:`numpy.ndarray`
        """
        if not hasattr(self, "theta") or self.theta is None:
            # If a polynomial order is defined and the function values `fx` are available,
            # compute the residual `fx` by subtracting the polynomial regressor's contribution.
            if self.get_order() is not None and self.get_fx() is not None:
                fx = self.fx - self.get_polynomial_regressor(z=self.get_x())
            else:
                ##
                fx = self.get_fx()
            # If `fx` is still `None`, it means there's no data to compute `theta` from, so set `theta` to `None`.
            if fx is None:
                self.theta = None
            else:
                # Compute the regression coefficient `theta` using the kernel matrix inverse and the function values.
                self.theta = lalg.prod(self.get_knm_inv(), fx)
        return self.theta

    def get_Delta(self) -> np.ndarray:
        """
        Compute and retrieve the discrete Laplace-Beltrami operator ``Delta``.

        :returns: The Laplace-Beltrami operator.
        :rtype: :class:`numpy.ndarray`
        """

        if self.Delta is None:
            self.Delta = diffops.nablaT_nabla(self.y, self.x)
        return self.Delta

    def select(self, x, N, fx=None, all=False, norm_="frobenius", **kwargs):
        """
        Select a subset of points using a greedy Nystrom approximation technique.

        The selection is based on norms such as the Frobenius norm or a classifier norm.

        :param x: Input data points.
        :type x: :class:`numpy.ndarray`
        :param N: The number of points to select.
        :type N: :class:`int`
        :param fx: Function values corresponding to the input data ``x``.
        :type fx: :class:`numpy.ndarray`, optional
        :param all: If ``True``, all points are selected. Defaults to ``False``.
        :type all: :class:`bool`, optional
        :param norm_: Norm used for selection. Can be `"frobenius"` or `"classifier"`.
                    The classifier norm is adapted to ``fx`` representing probabilities in a given label class:
                    $$
                    \\|f - f_{k, \\theta}(.)\\|
                    $$
        :type norm_: :class:`str`, optional
        :returns: Indices of the selected points.
        :rtype: :class:`numpy.ndarray`

        Note:

            `"classifier"` norm is a norm adapted to ``fx`` representing probabilities to be in a given label class: $||f - f_{k,\\theta}(.)||$
            where
            $$\\|A - B\\| = \\left\\|\\frac{A}{\sum A} - \\frac{B}{\sum B}\\right\\|_{\\ell^2}, \quad A \\in \\mathbb{R}^{N,D}, \\in \\mathbb{R}^{N,D}$$
        """
        # Set N to the max_nystrom if it is not specified
        if N is None:
            N = self.max_nystrom

        # Set input data and function values (fx) in the internal state of the object
        self.set_x(x)
        self.set_fx(fx)
        self.rescale()

        # If function values (fx) are provided, apply polynomial regression to `x`
        if self.get_fx() is not None:
            if self.get_polynomial_values() is not None:
                # If polynomial values are available, compute polynomial regression error
                polynomial_values = self.get_polynomial_regressor(z=self.get_x())
                # Subtract polynomial values from `fx` to get the residual error
                fx = self.fx - polynomial_values
            else:
                fx = self.fx
            # Apply hybrid greedy Nystrom with error between ||f .-f_{k,\theta}||_A and  wrt a given norm
            # udefined by user
            # to compute Y
            theta, indices = alg.HybridGreedyNystroem(
                x=self.get_x(), fx=fx, N=N, tol=0.0, error_type=norm_, **kwargs
            )
            #
            if all is False:
                # Select the subset of points based on the greedy Nystrom approximation
                self.set(
                    x=self.x,
                    y=self.x[indices],
                    fx=self.fx,
                    set_polynomial_regressor=False,
                )
            else:
                # If selecting all points, update the internal state to reflect X=Y
                self.set_x(self.x[indices], set_polynomial_regressor=False)
                self.set_fx(self.fx[indices], set_polynomial_regressor=False)
                self.set_theta(theta)
            return indices

        # If the number of points is smaller than or equal to N, return all indices
        if self.x.shape[0] <= N:
            indices = list(range(self.x.shape[0]))
            return indices

        # Otherwise, perform Maximum Mean Discrepancy (MMD)-based selection of points
        indices = [0]  # Start with the first point
        complement_indices = list(range(1, N))  # Remaining points

        # Iteratively select points with the maximum MMD distance
        for _ in range(N - 1):
            Dnm = core.op.Dnm(x[indices], x[complement_indices])  # Compute MMD distance
            indice = np.max(Dnm, axis=0)  # Select index with max distance
            indice = np.argmax(indice)
            indice = complement_indices[indice]
            complement_indices.remove(indice)
            indices.append(indice)
            pass

        # Update the internal state with the selected points
        self.set_x(self.x[indices])
        return indices

    def set(
        self, x: np.ndarray = None, fx: np.ndarray = None, y: np.ndarray = None
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
            return
        if x is not None and fx is None:
            self.set_x(core.get_matrix(x.copy()))
            self.set_y(y=y)
            self.set_fx(None)
            self.rescale()

        if x is None and fx is not None:
            if self.kernel is None:
                raise Exception("Please input x at least once")
            if fx.shape[0] != self.x.shape[0]:
                raise Exception(
                    "fx of size "
                    + str(fx.shape[0])
                    + "must have the same size as x"
                    + str(self.x.shape[0])
                )
            self.set_fx(core.get_matrix(fx))

        if x is not None and fx is not None:
            self.set_x(x), self.set_fx(fx), self.set_y(y=y)
            self.rescale()
        return self

    def map(
        self, x: np.ndarray, y: np.ndarray, distance: str = None, sub: bool = False
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
        self.set_x(x), self.set_fx(y)
        # Rescale the input data `x` using the current kernel configuration
        self.rescale()

        # Check if the dimensionality of `x` and `y` is the same
        if x.shape[1] != y.shape[1]:
            # If the dimensionalities differ, use an encoder to map data into latent space
            # and find the optimal permutation (descent-based method)
            self.permutation = cd.alg.encoder(self.get_x(), self.get_fx())
        else:
            # If the dimensionalities are the same, use the LSAP algorithm to compute the permutation
            D = core.op.Dnm(x=x, y=y, distance=distance)
            self.permutation = lsap(D, bool(sub))  # Solve LSAP to find permutation
        # Update `x` based on the computed permutation
        self.set_x(self.get_x()[self.permutation])
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

    def update_set(
        self, z: np.ndarray, fz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the training set by limiting the data to a maximum pool size.

        This method trims the input data ``z`` and corresponding function values ``fz``
        to the size defined by the ``max_pool`` parameter.

        :param z: Input data points to update.
        :type z: :class:`numpy.ndarray`
        :param fz: Function values corresponding to the input data ``z``.
        :type fz: :class:`numpy.ndarray`

        :returns: The trimmed input data points and corresponding function values, limited by ``max_pool``.
        :rtype: Tuple[:class:`numpy.ndarray`, :class:`numpy.ndarray`]
        """
        return z[-self.max_pool :], fz[-self.max_pool :]

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
            return [self.__call__(x, **kwargs) for x in z]
        z = core.get_matrix(z)
        if self.x is None:
            return None

        # Compute the kernel matrix `K(z, X)` where `X` is the current input dataset
        Knm = core.op.Knm(x=z, y=self.get_y())

        # If a polynomial order is defined, remove the polynomial regression component from `fz`
        if self.order is not None:
            # Compute the residual by subtracting the polynomial regressor's prediction from `fz`
            fzz = fz - self.get_polynomial_regressor(z)
        else:
            fzz = fz
        # err = self(z)-fz
        # err= (err**2).sum()

        # At this point, we are trying to solve the following least squares problem:
        #
        # $$ \theta = \text{argmin} \| K(z, X)\theta - f(z) \|^2 + \text{reg} \|\theta\|^2 $$
        #
        # `eps` (regularization) controls the trade-off between fitting the data and controlling the magnitude of `theta`.

        if eps is None:
            # Use the default regularization value if none is provided
            eps = self.reg
        # if self.theta is not None: fzz += eps*lalg.prod(Knm,self.theta)
        # Solve the least-squares problem with regularization:
        # $$ \theta = \left( K(z, X)^\top K(z, X) + \text{eps} \cdot I \right)^{-1} K(z, X)^\top fzz $$
        self.set_theta(lalg.lstsq(Knm, fzz, eps=eps))
        # err = self(z)-fz
        # err= (err**2).sum()

        # If polynomial regression is involved, update the kernel's internal function approximation
        if self.order is not None:
            # Update `fx` by adding the contribution of the polynomial regressor
            self.fx += self.get_polynomial_regressor(z=self.get_x())

        return self

    def add(self, y: np.ndarray = None, fy: np.ndarray = None) -> None:
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
        # if self.x is not None and x is not None: x=np.concatenate([self.x,x.copy()])[-self.max_pool:]
        # if self.fx is not None and fx is not None: fx=np.concatenate([self.fx,fx.copy()])[-self.max_pool:]
        if not hasattr(self, "x") or self.x is None:
            self.set(x, fx)
            return

        # the method add computes an updated Gram matrix using the already
        # pre-computed Gram matrix K(x,x).
        self.Knm, self.Knm_inv, y = alg.add(
            self.get_knm(), self.get_knm_inv(), self.get_x(), x
        )
        self.set_x(y)
        if fx is not None and self.get_fx() is not None:
            self.set_fx(np.concatenate([fx, self.get_fx()], axis=0))
        else:
            self.set_fx(fx)

        self._set_polynomial_regressor()
        return self

    def kernel_distance(self, z: np.ndarray) -> np.ndarray:
        """
        Compute an MMD (Maximum Mean Discrepancy) based distance matrix between the input data ``x`` and the new data ``z``.

        The distance is computed as:

        $$
        D(X,Z) = K(X,X) + K(Z,Z) - 2K(X,Z)
        $$

        :param z: New input data points.
        :type z: :class:`numpy.ndarray`

        :returns: The computed MMD-based distance matrix.
        :rtype: :class:`numpy.ndarray`
        """
        return core.op.Dnm(x=z, y=self.x)

    def get_kernel(self) -> callable:
        """
        Retrieve the current kernel function for the input data.

        This method retrieves a positive semi-definite (PSD) kernel function,
        represented as: $k(S(x), S(y))$, where $S$ is a predefined mapping.

        :returns: The kernel function used by the current model.
        :rtype: callable
        """
        if not hasattr(self, "kernel"):
            self.set_kernel()
            # self.order= None
            self.kernel = core.kernel_interface.get_kernel_ptr()
        return self.kernel

    def set_kernel_ptr(self) -> None:
        """
        Set the Codpy interface to use the current kernel function.

        This method updates the Codpy kernel interface with the current kernel
        function, sets the polynomial order to zero, and applies the regularization
        parameter defined in the object.
        """
        core.kernel_interface.set_kernel_ptr(self.get_kernel())
        core.kernel_interface.set_polynomial_order(0)
        core.kernel_interface.set_regularization(self.reg)

    def rescale(self) -> None:
        """
        Rescale the input data using the current mapping.

        This method rescales the input data by applying the map function associated
        with the current kernel. It also retrieves and updates the internal kernel
        function based on the rescaled data.

        If ``x`` is set, the rescaling is applied to ``x`` with a maximum number of
        points defined by ``max_nystrom``.
        """
        # instructs to set kernel
        self.set_kernel_ptr()
        if self.get_x() is not None:
            # instructs to set the map parameter
            # applied to the data
            core.kernel_interface.rescale(self.get_x(), max=self.max_nystrom)
            # retrives the kernel
            self.kernel = core.kernel_interface.get_kernel_ptr()

    def __call__(self, z: np.ndarray) -> np.ndarray:
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
        z = core.get_matrix(z)

        # Don't forget to set the kernel
        self.set_kernel_ptr()

        fy = self.get_theta()

        if fy is None:
            fy = self.get_knm_inv()

        Knm = core.op.Knm(x=z, y=self.get_y(), fy=fy)

        if self.order is not None:
            polynomial_regressor = self.get_polynomial_regressor(z)
            Knm += polynomial_regressor

        return Knm
