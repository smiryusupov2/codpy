import functools
import operator

import numpy as np
import itertools
from codpydll import *
import codpy.lalg
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


from codpy.data_conversion import get_matrix

def cartesian_outer_product(x:np.array,y:np.array) -> np.array:
    #perform the commented code but parallel and C++
    out = cd.tools.cartesian_outer_product(x,y)
    return out
    # out = np.zeros([x.shape[0],y.shape[0],x.shape[1]+y.shape[1]])
    # def helper(u):
    #     i,j= u
    #     out[i,j,:] = np.concatenate([x[i],y[j]])

    # iterable = itertools.product(range(x.shape[0]),range(y.shape[0]))
    # list(map(helper, iterable))
    # return out

def cartesian_sum(x:np.array,y:np.array) -> np.array:
    #perform the commented code but parallel and C++
    out = np.zeros([x.shape[0],y.shape[0],x.shape[1]])
    def helper(u):
        i,j= u
        out[i,j,:] = x[i]+y[j]

    iterable = itertools.product(range(x.shape[0]),range(y.shape[0]))
    list(map(helper, iterable))
    return out.reshape(out.shape[0]*out.shape[1],out.shape[2])


def pad_axis(x, y, axis=1):
    if x.shape[axis] == y.shape[axis]:
        return x, y
    if x.shape[axis] < y.shape[axis]:
        y, x = pad_axis(y, x)
        return x, y
    ref_shape = np.array(y.shape)
    ref_shape[axis] = x.shape[axis]
    out = np.zeros(ref_shape)
    slices = [slice(0, y.shape[dim]) for dim in range(y.ndim)]
    out[slices] = y
    return x, out


def format_32(x, **kwargs):
    if x.ndim == 3:
        x = np.swapaxes(x, 1, 2)
        return x.reshape((x.shape[0] * x.shape[1], x.shape[2]))


def format_23(x, shapes, **kwargs):
    if x.ndim == 2:
        out = x.reshape((shapes[0], shapes[2], shapes[1]))
        out = np.swapaxes(out, 1, 2)
        return out


def null_rows(data_frame):
    return data_frame[data_frame.isnull().any(axis=1)]


def flatten(list):
    return functools.reduce(operator.iconcat, list, [])


def softmaxindices(
    mat: np.ndarray,
    axis: int = 1,
    label: np.ndarray = [],
    diagonal: bool = False,
    **kwargs,
):
    """
    Selects indices of maximum values along a specified axis in a matrix, with optional modifications.

    This function computes the indices of the maximum values along the specified axis of the input matrix.
    It can optionally ignore the diagonal elements by setting them to a very large negative number. If provided,
    it can map these indices to corresponding labels.

    Args:
        mat (np.ndarray): A NumPy array (matrix) in which the maximum value indices are to be found.
        **kwargs: Additional keyword arguments.
            axis (int, optional): The axis along which to find the indices of the maximum values. Default is 1.
            label (np.ndarray, optional): An array of labels that can be used to map the indices to.
                                        If provided, the function returns labels instead of indices.
                                        Default is None.
            diagonal (bool, optional): If True, the function ignores the diagonal elements by setting them
                                    to a very large negative number. Default is False.

    Returns:
        np.ndarray or list: An array of indices or labels (if provided) of the maximum values along the specified axis.

    Example:
        >>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> max_indices = softmaxindices(mat)

        max_indices will be [2, 2, 2] for axis=1

        >>> labels = np.array(['a', 'b', 'c'])
        >>> max_labels = softmaxindices(mat, label=labels)

        max_labels will be ['c', 'c', 'c'] for axis=1

    Note:
    This function is useful in scenarios where you need to find the dominant indices or labels from a set of softmax
    probabilities.
    """
    # print(mat[0:5])
    if diagonal:

        def helper(n):
            mat[n, n] = -1e150

        [helper(n) for n in range(mat.shape[0])]
    out = np.argmax(get_matrix(mat), axis)
    if isinstance(label, np.ndarray):
        test = list()

        def helper(i):
            test.append(label[out[i]])

        [helper(i) for i in range(0, len(out))]
        out = test
    return out


def softmaxvalues(mat: np.ndarray, axis: int = 1, diagonal: bool = False, **kwargs):
    if diagonal:

        def helper(n):
            mat[n, n] = -1e150

        [helper(n) for n in range(mat.shape[0])]
    return np.max(get_matrix(mat), axis)


def softminindices(mat: np.ndarray, axis: int = 1, diagonal: bool = False, **kwargs):
    return softmaxindices(-mat, axis, diagonal, **kwargs)


def softminvalues(mat: np.ndarray, axis: int = 1, diagonal: bool = False, **kwargs):
    return -softmaxvalues(-mat, axis, diagonal, **kwargs)


def get_closest_index(mySortedList, myNumber):
    from bisect import bisect_left

    if isinstance(myNumber, list):
        return [get_closest_index(mySortedList, n) for n in myNumber]
    pos = bisect_left(mySortedList, myNumber)
    if pos == len(mySortedList):
        return pos - 1
    return pos


def get_closest_list(mySortedList, myNumber):
    pos = get_closest_index(mySortedList, myNumber)
    if pos == 0:
        return mySortedList[0]
    if pos == len(mySortedList):
        return mySortedList[-1]
    before = mySortedList[pos - 1]
    after = mySortedList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def gather(matrix, indices):
    indices = get_matrix(indices, dtype=int)
    mask = np.full(matrix.shape, False)

    def helper(i):
        mask[i, int(indices[i, 0])] = True

    [helper(i) for i in range(matrix.shape[0])]
    return get_matrix(matrix[mask])


def fill(matrix, values, indices=None, op=None):
    if indices is None:

        def helper(j):
            if op is None:
                matrix[:, j] = values
            else:
                matrix[:, j] = op(matrix[:, j], values)
            [helper(j) for j in range(matrix.shape[1])]
    else:

        def helper(i):
            if op is None:
                matrix[i, int(indices[i])] = values[i]
            else:
                matrix[i, int(indices[i])] = op(matrix[i, int(indices[i])], values[i])

        [helper(i) for i in range(matrix.shape[0])]

class LinearRegression:
    """
    Polynomial Regression based on scikit learn with a similar interface to Kernel class.
    This class fits a polynomial regression model based on the current polynomial order.
    Basic usage : LinearRegression(x,fx,order)(z) provides a polynomial expansion of $f(z) = \beta^0 + \sum_d \beta^1_d z_d+\sum_{d_1,d_2} \beta^2_{d_1,d_2} z_{d_1}z_{d_2}$ + ..
    where the coefficients $\beta^0, \beta^1, ...$ are fitted to $X,f(X)$.
    The gradient is not available (use Kernel class with a linear regressor if needed)
    """
    def __init__(
        self,
        x,
        fx,
        order = 1
    ) -> None:
        self.x, self.fx, self.order = x,fx,order
    def get_order(self):
        return self.order
    def get_x(self):
        return self.x
    def get_fx(self):
        return self.fx
    
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

    def __call__(
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


def fit_to_cov(datas,cov_matrix):
    from scipy.linalg import sqrtm
    import numpy as np
    mean =   np.mean(datas, axis=0)  
    datas -= mean
    datas /= np.std(datas, axis=0)

# Multiply by the Cholesky decomposition of the covariance matrix
    sqrtm_ = sqrtm(cov_matrix)
    transformed_data = datas@sqrtm_
    transformed_data += mean
    #check
    # error = cov_matrix - np.cov(transformed_data.T)
    return transformed_data

if __name__ == "__main__":
    test = LinearRegression(x=get_matrix([[0.,1.,2.]]).T,fx=get_matrix([[1.,3.,5.]]).T, order=1)
    print(test(z=get_matrix([0.,1.,2.,3.])))

    import matplotlib.pyplot as plt
    import scipy.special
    delta_t = .1
    M=50
    MC = np.zeros([M,M])
    SD = np.zeros([M,M])
    import random
    normals = np.random.normal(size=[M,M])
    normals -= np.mean(normals, axis=0)
    erfinv = scipy.special.erfinv(np.linspace(-.5+.5/M, .5-.5/M,M))
    for n in range(1,M):
        MC[:,n] = MC[:,n-1] + np.sqrt(delta_t)*normals[:,n]
        SD[:,n] = erfinv * 4.*np.sqrt(n*delta_t)
    labels = ["Naive MC","Sharp Disc"]
    for i in range(M):
        x,y = MC[i,:], range(M)
        line, = plt.plot(x,y, color="r", alpha=.5)
    line.set_label("Naive MC")
    for i in range(M):
        x,y = SD[i,:], range(M)
        line, = plt.plot(x,y, color="b")
    line.set_label("Sharp Disc.")
    plt.legend()
    plt.title("Brownian motion")
    # test = [1,2]
    # test=map_invertion(test)
    # print(test)
    plt.show()
    # u = f(t) u_0
    # dtu = f'(t) u_0 = Delta u / 2 = f'(t)/(f(t)) Delta u
    # (f'/f)=1/2 f=sqrt(2)
    # ln(f)=t
    # f = exp(t)
    pass
