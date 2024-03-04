from selection import *
from data_conversion import *
from codpydll import *
import numpy as np
from functools import partial, cache



class _codpy_param_getter:
    def get_params(**kwargs) : return kwargs.get('codpy',{})
    def get_kernel_fun(**kwargs): return _codpy_param_getter.get_params(**kwargs)['set_kernel']


class op:
    def projection(x, y, z, fx):
        """
        Performs projection in kernel regression for efficient computation, targeting a lower sampling space.
        Note:
            The performance of the function depends on two ingredients:
            
            * ``kernel`` function
            * ``map``

        Example:

            With NumPy arrays

            >>> xtrain = np.random.randn(100, 1)
            >>> xtest = np.random.randn(100, 1)
            >>> fx_train = x * 2
            >>> fx_test = z * 2
            >>> fx_test_pred = projection(xtrain, xtrain, xtest, fx_train, 
            kernel_fun = "tensornorm", map = "unitcube", 
                   polynomial_order=2)

            With pandas DataFrames

            >>> x_train_df = pd.DataFrame([...])
            >>> y_train_df = pd.DataFrame([...])
            >>> z_test_df = pd.DataFrame([...])
            >>> fx_train_df = pd.DataFrame([...])
            >>> projected_responses = projection(x_train_df, y_train_df, z_test_df, fx_train_df, kernel_fun = "tensornorm", map = "unitcube", 
                   polynomial_order=2)
        """
        def project_dataframe(x, y, z, fx, reg):
            x, y, z = column_selector([x, y, z], **kwargs)
            f_z = cd.op.projection(x, y, z, fx, reg)
            if isinstance(fx, pd.DataFrame):
                f_z = pd.DataFrame(f_z, columns=list(fx.columns))
                if isinstance(z, pd.DataFrame):
                    f_z.index = z.index
            return f_z
        
        def project_array(x, y, z, fx, reg):
            return cd.op.projection(x, y, z, fx, reg)

        params = {'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization)}
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x,y,z, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)

        if isinstance(z, list):
            return [project_dataframe(x, y, zi, fx, reg) if isinstance(x, pd.DataFrame) else project_array(x, y, zi, fx, reg) for zi in z]

        if isinstance(x, pd.DataFrame):
            return project_dataframe(x, y, z, fx, reg)
        
        return project_array(x, y, z, fx, reg)


    def _weighted_projection(**kwargs):
        projection_format_switchDict = { pd.DataFrame: lambda **kwargs :  projection_dataframe(**kwargs) }
        z = kwargs['z']
        if isinstance(z,list):
            def fun(zi):
                kwargs['z'] = zi
                return op.weighted_projection(**kwargs)
            out = [fun(zi) for zi in z]
            return out
        def projection_dataframe(**kwargs):
            x,y,z,fx,weights = kwargs.get('x',[]),kwargs.get('y',x),kwargs.get('z',[]),kwargs.get('fx',[]),np.array(kwargs.get('weights',[]))
            x,y,z = column_selector([x,y,z],**kwargs)
            f_z = cd.op.weighted_projection(get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx),weights)
            if isinstance(fx,pd.DataFrame): f_z = pd.DataFrame(f_z,columns = list(fx.columns), index = z.index)
            return f_z

        kernel.init(**kwargs)
        type_debug = type(kwargs.get('x',[]))

        def debug_fun(**kwargs):
            x,y,z,fx,weights = kwargs.get('x',[]),kwargs.get('y',x),kwargs.get('z',[]),kwargs.get('fx',[]),np.array(kwargs.get('weights',[]))
            f_z = cd.op.weighted_projection(get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx),weights)
            return f_z

        method = projection_format_switchDict.get(type_debug,debug_fun)
        f_z = method(**kwargs)

        return f_z

    def extrapolation(x, z, fx=[]):
        """
        Performs extrapolation in the context of kernel regression.

        This method leverages the kernel regression framework to extrapolate values between data points. 
        Note:
            The performance of the function depends on two ingredients:
            
            * ``kernel`` function
            * ``map``
        """

        return op.projection(x = x, y = x, z = z, fx = fx)

    def interpolation(x, z, fx=[]):
        """
        Performs interpolation in the context of kernel regression.

        This method leverages the kernel regression framework to interpolate values between data points. 
        Note:
            The performance of the function depends on two ingredients:
            
            * ``kernel`` function
            * ``map``
        """
        return op.projection(x = x, y = z, z = z, fx = fx)
    
    def gradient_denoiser(x, z, fx,epsilon):
        """
        A function for performing least squares regression penalized by the norm of the gradient, 
        induced by a positive definite (PD) kernel.

        This functioon initializes with various parameters and sets up a regression framework 
        that includes regularization based on the gradient's magnitude. It is designed to 
        work with gradient norms induced by a PD kernel.
        Returns:
            The denoised output for the input data.

        Example:

        Initialize the denoiser with input data 'x' and optional parameters

        >>> xtrain = np.random.randn(100, 1)
        >>> xtest = np.random.randn(100, 1)
        >>> fx_train = x * 2
        >>> fx_test = z * 2
        
        # Perform denoising on the input data or new data points 'z'
        op.denoiser(xtrain, xtest, fx_train, kernel_fun = "maternnorm", map = "standardmean")
        """
        reg = epsilon*diffops.nablaT_nabla(x = x , y = x, fx = [])
        out = op.extrapolation(x, z = z, fx = fx)    
        # self.reg = self.epsilon*op.nablaT_nabla(**{**kw,**{'fx':[],'y':self.y,'x':self.x}})
        # self.kernel = kernel_setters.kernel_helper(kernel_setters.set_matern_normkernel, 0,1e-8 ,map_setters.set_standard_mean_map)
        # out = op.extrapolation(**{**self.params,**{'z':z}})
        return out
    
    def norm(x, y, z, fx):
        """
        Calculate the kernel-induced norm based on the provided matrices.

        This function computes a norm projection using the kernel initialization parameters. 
        It supports flexible argument input through keyword arguments.

        Args:
            x (list, optional): The first matrix. Defaults to an empty list.
            y (list, optional): The second matrix. Defaults to a list containing `x`.
            z (list, optional): The third matrix. Defaults to an empty list.
            fx (list, optional): The fourth matrix. Defaults to an empty list.
        Note:
            The performance of the function depends on two ingredients:
            
            * ``kernel`` function
            * ``map``
        """
        params = {'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization)}
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x,y,z, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)
        return cd.tools.norm_projection(get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx))

    def coefficients(x, y, fx) -> np.ndarray:
        """
        Computes the regressors or coefficients for kernelized regression, using a specified PD kernel.

        This method initializes the kernel function with the given parameters and then 
        computes the regression coefficients based on the input datasets. 

        Args:
            x (np.array): The input data points for which the regression coefficients are computed.
            y (np.array): Internal parameter, can be y = x.
            fx (np.array, optional): Responses associated with 'x'.
        :param kernel_fun: The name of the kernel function to use. Options include ``'gaussian'``, ``'tensornorm'``, etc.
        :type kernel_fun: :class:`str`, optional
        :param map: The name of the mapping function to apply. Options include ``'linear'``, ``'affine'``, etc.
        :type map: :class:`str`, optional
        :param polynomial_order: The polynomial order for the kernel function. Defaults to ``2``.
        :type polynomial_order: :class:`float`, optional
        :param regularization: Regularization parameter for the kernel. Defaults to ``1e-8``.
        :type regularization: :class:`numpy.ndarray`, optional
        :param rescale: Whether to rescale the data.
        :type rescale: :class:`bool`, optional
        :param rescale_params: Parameters for data rescaling. Defaults to ``{'max': 1000, 'seed': 42}``.
        :type rescale_params: :class:`dict`, optional
        :param kwargs: Arbitrary keyword arguments.
        :type kwargs: dict

        Returns:
            np.array: The computed regression coefficients or regressors that model the 
            relationship between the input data 'x' and the target 'y'.


        Available Kernels:

            - ``'gaussian'``: Gaussian kernel for smooth, continuous functions.
            - ``'tensornorm'``: Tensor norm kernel suitable for multidimensional data.
            - ``'absnorm'``: Absolute norm kernel for robust performance in varied datasets.
            - ``'matern'``: Matérn kernel useful in spatial statistics.
            - ``'multiquadricnorm'``: Multi-quadric norm kernel for flexible shape adaptation.
            - ``'multiquadrictensor'``: Multi-quadric tensor kernel, a tensor-based variant offering flexible shape adaptation.
            - ``'sincardtensor'``: Sinc cardinal tensor kernel, suitable for periodic and oscillatory data.
            - ``'sincardsquaretensor'``: Sinc cardinal square tensor kernel, enhancing the sinc cardinal tensor kernel for certain data types.
            - ``'dotproduct'``: Dot product kernel, useful for linear classifications and regressions.
            - ``'gaussianper'``: Gaussian periodic kernel, ideal for modeling periodic functions.
            - ``'maternnorm'``: Matérn norm kernel, a variation of the Matérn kernel for normalized spaces.
            - ``'scalarproduct'``: Scalar product kernel, a simple yet effective kernel for dot products.

        Available Maps:

            - ``'linear'``: Linear map for straightforward transformations.
            - ``'affine'``: Affine map for linear transformations with translation.
            - ``'log'``: Logarithmic map for non-linear scaling.
            - ``'exp'``: Exponential map for rapidly increasing scales.
            - ``'scalestd'``: Standard scaling map that normalizes data by removing the mean and scaling to unit variance.
            - ``'erf'``: Error function map, useful for data normalization with a non-linear scale.
            - ``'erfinv'``: Inverse error function map, providing the inverse transformation of the error function.
            - ``'scalefactor'``: Scaling factor map that applies a uniform scaling defined by a bandwidth or scale factor.
            - ``'bandwidth'``: Helper for setting the scale factor map with a specified bandwidth, typically used in kernel methods.
            - ``'grid'``: Grid map that projects data onto a grid, useful for discretizing continuous variables or spatial data.
            - ``'unitcube'``: Unit cube map that scales data to fit within a unit cube, ensuring all features are within the range [0,1].
            - ``'meandistance'``: Mean distance map that scales data based on the mean distance between data points.
            - ``'mindistance'``: Minimum distance map that scales data based on the minimum distance between data points.
            - ``'standardmin'``: Standard minimum map pipeline combining minimum distance scaling with other transformations.
            - ``'standardmean'``: Standard mean map pipeline combining mean distance scaling with other transformations.
        """
        params = {'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization)}
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x,y,x, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)
        return cd.op.coefficients(get_matrix(x),get_matrix(y),get_matrix(fx), [])
    # @cache
    def Knm(x, y, fx = []) -> np.ndarray:
        """
        Computes the kernel matrix induced by a positive definite (pd) kernel.

        This method calculates the kernel matrix k(x_i,y_j) using the input kernel function. 

        Args:
            x (:class:`numpy.ndarray`): Input data points for the gradient computation. np.array of size N , D.
            y (:class:`numpy.ndarray`): Secondary data points used in the kernel computation. np.array of size M , D.
            fx (:class:`numpy.ndarray`): optional-Function values or responses at the data points in `x`. np.array of size M , Df.

        Returns:
            - if fx is empty: matrix np.array of size (NxM) The computed kernel matrix, representing the kernel-induced distances or similarities between the data points in 'x' and 'y'.
            - prod(Knm(x,y),fx) else. This allow performance and memory optimizations.

        """
        return cd.op.Knm(x, y, fx)

    def Knm_inv(x, y, fx=[],reg=[]):
        """
        Args:
            x (:class:`numpy.ndarray`): Input data points for the gradient computation. np.array of size N , D.
            y (:class:`numpy.ndarray`): Secondary data points used in the kernel computation. np.array of size M , D.
            fx (:class:`numpy.ndarray`): optional-Function values or responses at the data points in `x`. np.array of size M , Df.
        Returns:
            - if reg is empty:
                - if fx is empty: matrix np.array of size (NxM), that is the least square inverse of Knm(x,y).
                - prod(Knm_inv(x,y),fx) else. This allow performance and memory optimizations. The output corresponds then to the coefficient of fx in the kernel induced basis.
            - else: 
                - if fx is empty: matrix np.array of size (NxM), that corresponds to the least square computation (Knm(y,x)Knm(x,y)+reg)^{-1}Knm(y,x).
                - prod(Knm_inv(x,y),fx) else. This allow performance and memory optimizations. The output corresponds then to the coefficient of fx in the kernel induced basis.
        """
        return cd.op.Knm_inv(get_matrix(x),get_matrix(y),get_matrix(fx),get_matrix(reg))
    
    def Dnm(x, y, distance = None, kernel_fun = "tensornorm", map = "unitcube", 
                   polynomial_order=2, regularization: float = 1e-8, rescale = False, 
                   rescale_params: dict = {'max': 1000, 'seed':42}, verbose = False,
                   **kwargs) -> np.ndarray:
        """
        Computes a distance matrix induced by a positive definite (pd) kernel.

        This function calculates the distance matrix between sets of data points
        x and y based on a specified pd kernel.

        Args:
            x (np.array): The first set of data points.
            y (np.array): The second set of data points. If not provided, defaults to x.
            distance (function, optional): a name of distance function.
        :param kernel_fun: The name of the kernel function to use. Options include ``'gaussian'``, ``'tensornorm'``, etc.
        :type kernel_fun: :class:`str`, optional
        :param map: The name of the mapping function to apply. Options include ``'linear'``, ``'affine'``, etc.
        :type map: :class:`str`, optional
        :param polynomial_order: The polynomial order for the kernel function. Defaults to ``2``.
        :type polynomial_order: :class:`float`, optional
        :param regularization: Regularization parameter for the kernel. Defaults to ``1e-8``.
        :type regularization: :class:`numpy.ndarray`, optional
        :param rescale: Whether to rescale the data.
        :type rescale: :class:`bool`, optional
        :param rescale_params: Parameters for data rescaling. Defaults to ``{'max': 1000, 'seed': 42}``.
        :type rescale_params: :class:`dict`, optional
        :param kwargs: Arbitrary keyword arguments.
        :type kwargs: dict

        Returns:
            np.array: A distance matrix representing the distances between each pair of points in x and y as induced by the pd kernel.
        """
        x,y = column_selector(x,**kwargs),column_selector(y,**kwargs)
        x,y = pad_axis(x,y)
        params = {'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization)}
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x,y,x, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)
        if distance is not None:
            return cd.op.Dnm(get_matrix(x),get_matrix(y), {'distance' : "norm22"})
        return cd.op.Dnm(get_matrix(x),get_matrix(y))
    
    def discrepancy_error(x: np.array = None, z : np.array = None, disc_type="raw", 
                    kernel_fun = "tensornorm", map = "unitcube", polynomial_order=2, reg: float = 1e-8, rescale_params: dict = {'max': 2000, 'seed':42}, 
                    rescale = False, verbose = False, **kwargs):
        params = {'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization)}
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x, x, x, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)
        return cd.tools.discrepancy_error(x,z, disc_type)

    def norm_projection(x: np.array = None, z: np.array = None, fx: np.array = None, kernel_fun: str = "tensornorm", map: str = "unitcube", 
                    polynomial_order=2, regularization: float = 1e-8, reg: np.ndarray = [], 
                    rescale: bool = False, rescale_params: dict = {'max': 2000, 'seed':42}, verbose = False, **kwargs):
        params = {'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization)}
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x,x,x, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)
        return cd.tools.norm_projection(x,x,z,fx)


def distance_labelling(x, y, label = None, distance = None, kernel_fun = "tensornorm", map = "unitcube", 
                   polynomial_order=2, regularization: float = 1e-8, rescale = False,
                    maxmin: str = 'max', axis: int = 1, **kwargs) -> np.ndarray:
    """
    Computes and labels distances using a kernel-induced distance matrix.

    This function calculates the distance matrix between two sets of data points (x and y) using 
    a specified kernel function. It then labels these distances based on either the softmax or softmin 
    indices, depending on the 'max' parameter in kwargs.

    Args:
        x (np.array): The first set of data points.
        y (np.array): The second set of data points.
        axis (int, optional): The axis along which to compute the distances. Default is 1.
        max (bool, optional): Determines the type of labelling:
            - If True, uses softmax labelling.
            - If False (default), uses softmin labelling.
            :param kernel_fun: The name of the kernel function to use. Options include ``'gaussian'``, ``'tensornorm'``, etc.
    :type kernel_fun: :class:`str`, optional
    :param map: The name of the mapping function to apply. Options include ``'linear'``, ``'affine'``, etc.
    :type map: :class:`str`, optional
    :param polynomial_order: The polynomial order for the kernel function. Defaults to ``2``.
    :type polynomial_order: :class:`float`, optional
    :param regularization: Regularization parameter for the kernel. Defaults to ``1e-8``.
    :type regularization: :class:`numpy.ndarray`, optional
    :param rescale: Whether to rescale the data.
    :type rescale: :class:`bool`, optional
    :param rescale_params: Parameters for data rescaling. Defaults to ``{'max': 1000, 'seed': 42}``.
    :type rescale_params: :class:`dict`, optional
    :param kwargs: Arbitrary keyword arguments.
    :type kwargs: dict

    Returns:
        np.array: An array of labelled distances between the data points in x and y.
    """
    # print('######','distance_labelling','######')

    x, y = column_selector(x,**kwargs),column_selector(y,**kwargs)
    D = op.Dnm(x, y, distance, kernel_fun = kernel_fun, map = map, 
                   polynomial_order=polynomial_order, regularization = regularization, rescale = False, **kwargs)
    if  maxmin == 'max': 
        return softmaxindice(D, axis=axis)
    return softminindice(D, axis=axis)


def discrepancy(x: np.array = None, y: np.array = None, z : np.array = None, disc_type="raw", 
                kernel_fun = "tensornorm", map = "unitcube", 
                   polynomial_order=2, reg: float = 1e-8, rescale = False, **kwargs): 

    if 'discrepancy:xmax' in kwargs: x= random_select_interface(xmaxlabel = 'discrepancy:xmax', seedlabel = 'discrepancy:seed',**{**kwargs,**{'x':x}})
    if 'discrepancy:ymax' in kwargs: y= random_select_interface(xmaxlabel = 'discrepancy:ymax', seedlabel = 'discrepancy:seed',**{**kwargs,**{'x':y}})
    if 'discrepancy:zmax' in kwargs: z= random_select_interface(xmaxlabel = 'discrepancy:zmax', seedlabel = 'discrepancy:seed',**{**kwargs,**{'x':z}})
    if 'discrepancy:nmax' in kwargs:
        nmax = int(kwargs.get('discrepancy:nmax'))
        if len(x) + 2 * len(y) + len(z) > nmax: return np.NaN
    params = {'rescalekernel':{'max': 1000, 'seed':42},
        'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=reg),
        'rescale': rescale,
        }
    kernel.init(**params)
    debug = 0.
    if (len(y)):
        debug += cd.tools.discrepancy_error(x,y,disc_type)
        if (len(z)):
            debug += cd.tools.discrepancy_error(y,z,disc_type)
        return np.sqrt(debug)
    else: return np.sqrt(cd.tools.discrepancy_error(x,z,disc_type))

class discrepancy_functional:
    """
    A kernel-induced discrepancy between two distributions.

    Discrepancy is a non-parametric method to test the equality of two distributions. It's computed in a
    Reproducing Kernel Hilbert Space (RKHS) using a specified kernel function.

    Attributes:
    Nx (int): The number of samples in the first distribution 'x'.
    x (array-like): The first distribution for which MMD is to be computed.
    Kxx (float): The kernel-induced distance computed within 'x'.

    Args:
        x (array-like): The first input distribution.
        y (array-like, optional): The second input distribution. If not provided, it defaults to an empty list.
        **kwargs: Additional keyword arguments for the kernel function.

    Methods:
        eval(ys, **kwargs): Computes the MMD between the distribution 'ys' and the initial distribution 'x'.

    Example:
        Define two distributions

        >>> x = np.array([...])
        >>> y = np.array([...])

        Initialize discrepancy functional for 'x'
        
        >>> discrepancy = discrepancy_functional(x)

        Compute MMD between 'x' and 'y'
        
        >>> discrepancy_value = discrepancy.eval(y)
    """
    def __init__(self,x, fx, kernel_fun = "tensornorm", map = "unitcube", 
                   polynomial_order=2, regularization: float = 1e-8, reg: np.ndarray = [], 
                   rescale = False, **kwargs):     
        """
        Args:

        :param kernel_fun: The name of the kernel function to use. Options include ``'gaussian'``, ``'tensornorm'``, etc.
        :type kernel_fun: :class:`str`, optional
        :param map: The name of the mapping function to apply. Options include ``'linear'``, ``'affine'``, etc.
        :type map: :class:`str`, optional
        :param polynomial_order: The polynomial order for the kernel function. Defaults to ``2``.
        :type polynomial_order: :class:`float`, optional
        :param regularization: Regularization parameter for the kernel. Defaults to ``1e-8``.
        :type regularization: :class:`numpy.ndarray`, optional
        :param rescale: Whether to rescale the data.
        :type rescale: :class:`bool`, optional
        :param rescale_params: Parameters for data rescaling. Defaults to ``{'max': 1000, 'seed': 42}``.
        :type rescale_params: :class:`dict`, optional
        :param kwargs: Arbitrary keyword arguments.
        :type kwargs: dict
        """  
        self.Nx = len(x)
        self.x = x.copy()

        self.Kxx = op.Knm(x, x, fx, Kinv = None, kernel_fun = kernel_fun, map = map, 
                   polynomial_order=polynomial_order, regularization = regularization, 
                   reg = reg, rescale = rescale, **kwargs)
        self.Kxx = np.sum(self.Kxx ) / (self.Nx*self.Nx)
        pass
    def eval(self, ys, fx, Kinv = None, kernel_fun = "tensornorm", map = "unitcube", 
                   polynomial_order=2, regularization = 1e-8, reg = [], rescale = False, **kwargs):
        """
        Args:

        :param kernel_fun: The name of the kernel function to use. Options include ``'gaussian'``, ``'tensornorm'``, etc.
        :type kernel_fun: :class:`str`, optional
        :param map: The name of the mapping function to apply. Options include ``'linear'``, ``'affine'``, etc.
        :type map: :class:`str`, optional
        :param polynomial_order: The polynomial order for the kernel function. Defaults to ``2``.
        :type polynomial_order: :class:`float`, optional
        :param regularization: Regularization parameter for the kernel. Defaults to ``1e-8``.
        :type regularization: :class:`numpy.ndarray`, optional
        :param rescale: Whether to rescale the data.
        :type rescale: :class:`bool`, optional
        :param rescale_params: Parameters for data rescaling. Defaults to ``{'max': 1000, 'seed': 42}``.
        :type rescale_params: :class:`dict`, optional
        :param kwargs: Arbitrary keyword arguments.
        :type kwargs: dict
        """
        N = len(ys)
        Kxy = op.Knm(x = ys, y = self.x, fx = fx, Kinv = None, kernel_fun = kernel_fun, map = map, 
                   polynomial_order=polynomial_order, regularization = regularization, 
                   reg = reg, rescale = rescale, **kwargs)/(self.Nx)
        out = np.zeros([N])
        for n in range(N):
            out[n] = 1.+self.Kxx-2.*np.sum(Kxy[n])
        return out
    
class diffops:
    def nabla_Knm(x, y, kernel_fun = "tensornorm", map = "unitcube", 
                   polynomial_order=2, reg: float = 1e-8, rescale = False, 
                   rescale_params: dict = {'max': 1000, 'seed':42}, verbose = False, **kwargs):
        """
        Args:

        :param kernel_fun: The name of the kernel function to use. Options include ``'gaussian'``, ``'tensornorm'``, etc.
        :type kernel_fun: :class:`str`, optional
        :param map: The name of the mapping function to apply. Options include ``'linear'``, ``'affine'``, etc.
        :type map: :class:`str`, optional
        :param polynomial_order: The polynomial order for the kernel function. Defaults to ``2``.
        :type polynomial_order: :class:`float`, optional
        :param regularization: Regularization parameter for the kernel. Defaults to ``1e-8``.
        :type regularization: :class:`numpy.ndarray`, optional
        :param rescale: Whether to rescale the data.
        :type rescale: :class:`bool`, optional
        :param rescale_params: Parameters for data rescaling. Defaults to ``{'max': 1000, 'seed': 42}``.
        :type rescale_params: :class:`dict`, optional
        :param kwargs: Arbitrary keyword arguments.
        :type kwargs: dict
        """
        
        params = {'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization)}
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x,y,x, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)
        return cd.op.nabla_Knm(get_matrix(x),get_matrix(y))

    def nabla(x, y, z, fx, kernel_fun = "tensornorm", map = "unitcube", 
                   polynomial_order=2, regularization: float = 1e-8, reg: np.ndarray = [], rescale = False, 
                   rescale_params: dict = {'max': 1000, 'seed':42}, verbose = False, **kwargs):
        """
        Compute the kernel-induced gradient of a function.

        Args:
            x (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Input data points for the gradient computation.
            y (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Secondary data points used in the kernel computation.
            z (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Tertiary data points used in the kernel computation.
            fx (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Function values or responses at the data points in `x`.
        :param kernel_fun: The name of the kernel function to use. Options include ``'gaussian'``, ``'tensornorm'``, etc.
        :type kernel_fun: :class:`str`, optional
        :param map: The name of the mapping function to apply. Options include ``'linear'``, ``'affine'``, etc.
        :type map: :class:`str`, optional
        :param polynomial_order: The polynomial order for the kernel function. Defaults to ``2``.
        :type polynomial_order: :class:`float`, optional
        :param regularization: Regularization parameter for the kernel. Defaults to ``1e-8``.
        :type regularization: :class:`numpy.ndarray`, optional
        :param rescale: Whether to rescale the data.
        :type rescale: :class:`bool`, optional
        :param rescale_params: Parameters for data rescaling. Defaults to ``{'max': 1000, 'seed': 42}``.
        :type rescale_params: :class:`dict`, optional
        :param kwargs: Arbitrary keyword arguments.
        :type kwargs: dict

        Returns:
            :class:`numpy.ndarray`: The computed gradient of the function.

        Example:

            >>> x = np.random.randn(100, 1)
            >>> z = np.random.randn(100, 1)
            >>> fx = x * 2
            >>> fz = z * 2
            >>> gradient = diffops.nabla(x,x,z,fx,kernel_fun="linear", map=None)
        """

        params = {'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization)}
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x,y,z, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)

        return cd.op.nabla(get_matrix(x),get_matrix(y),get_matrix(z), fx,get_matrix(reg))
    
    def nabla_inv(x, y, z, fz, kernel_fun:str = "tensornorm", map: str = "unitcube", 
                   polynomial_order:int = 2, regularization: float = 1e-8, reg: np.ndarray = [],
                    rescale: bool = False, rescale_params: dict = {'max': 1000, 'seed':42}, 
                    verbose = False, **kwargs):
        """
        Compute the inverse of the kernel-induced gradient operation.

        Args:
            x (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Input data points.
            y (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Secondary data points.
            z (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Tertiary data points.
            fz (:class:`numpy.ndarray`, optional): The vector field for the inverse gradient computation.
        :param kernel_fun: The name of the kernel function to use. Options include ``'gaussian'``, ``'tensornorm'``, etc.
        :type kernel_fun: :class:`str`, optional
        :param map: The name of the mapping function to apply. Options include ``'linear'``, ``'affine'``, etc.
        :type map: :class:`str`, optional
        :param polynomial_order: The polynomial order for the kernel function. Defaults to ``2``.
        :type polynomial_order: :class:`float`, optional
        :param regularization: Regularization parameter for the kernel. Defaults to ``1e-8``.
        :type regularization: :class:`numpy.ndarray`, optional
        :param rescale: Whether to rescale the data.
        :type rescale: :class:`bool`, optional
        :param rescale_params: Parameters for data rescaling. Defaults to ``{'max': 1000, 'seed': 42}``.
        :type rescale_params: :class:`dict`, optional
        :param kwargs: Arbitrary keyword arguments.
        :type kwargs: dict

        Returns:
            :class:`numpy.ndarray`: The computed inverse gradient of the vector field.

        Example:
        
            >>> x_data = np.array([...])
            >>> y_data = np.array([...])
            >>> z_data = np.array([...])
            >>> vector_field = np.array([...])
            >>> inv_gradient = nabla_inv(x_data, y_data, z_data, fz=vector_field)
        """
        
        params = {'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization)}
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x,y,z, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)
        return cd.op.nabla_inv(get_matrix(x),get_matrix(y),get_matrix(z), fz)
    
    def nablaT(x, y, z, fz, kernel_fun = "tensornorm", map = "unitcube", 
                   polynomial_order=2, regularization: float = 1e-8, reg: np.ndarray = [], 
                   rescale = False, rescale_params: dict = {'max': 1000, 'seed':42},  
                   verbose = False, **kwargs):
        """
        Compute the divergence of a vector field using a kernel-induced method.

        This function calculates the divergence (nabla transpose) of a vector field in the context of kernel methods. 

        Args:
            x class:`numpy.ndarray`: The input data points where the divergence is calculated.
            y class:`numpy.ndarray`: Secondary data points used in the kernel computation.
            z class:`numpy.ndarray`: Tertiary data points used in the kernel computation.
            fz class:`numpy.ndarray`: The vector field for which the divergence is computed. Defaults to an empty list.
        :param kernel_fun: The name of the kernel function to use. Options include ``'gaussian'``, ``'tensornorm'``, etc.
        :type kernel_fun: :class:`str`, optional
        :param map: The name of the mapping function to apply. Options include ``'linear'``, ``'affine'``, etc.
        :type map: :class:`str`, optional
        :param polynomial_order: The polynomial order for the kernel function. Defaults to ``2``.
        :type polynomial_order: :class:`float`, optional
        :param regularization: Regularization parameter for the kernel. Defaults to ``1e-8``.
        :type regularization: :class:`numpy.ndarray`, optional
        :param rescale: Whether to rescale the data.
        :type rescale: :class:`bool`, optional
        :param rescale_params: Parameters for data rescaling. Defaults to ``{'max': 1000, 'seed': 42}``.
        :type rescale_params: :class:`dict`, optional
        :param kwargs: Arbitrary keyword arguments.
        :type kwargs: dict

        Returns:
            class:`numpy.ndarray`: The computed divergence of the vector field at each point in `x`.

        Example:
            Example usage with NumPy arrays

            >>> x_data = np.array([...])
            >>> y_data = np.array([...])
            >>> z_data = np.array([...])
            >>> vector_field = np.array([...])
            >>> divergence = nablaT(x_data, y_data, z_data, fz=vector_field)
        """

        params = {'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization)}
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x,y,z, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)

        return cd.op.nablaT(get_matrix(x),get_matrix(y),get_matrix(z),fz)
    
    def nablaT_inv(x, y, z, fx, kernel_fun = "tensornorm", map = "unitcube", 
                   polynomial_order=2, reg: float = 1e-8, rescale = False, 
                   rescale_params: dict = {'max': 1000, 'seed':42}, verbose = False, **kwargs):
        """
        Compute the inverse of the transposed gradient operation.

        Args:
            x (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Input data points.
            y (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Secondary data points.
            z (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Tertiary data points.
            fx (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Function values or responses at the data points.
        :param kernel_fun: The name of the kernel function to use. Options include ``'gaussian'``, ``'tensornorm'``, etc.
        :type kernel_fun: :class:`str`, optional
        :param map: The name of the mapping function to apply. Options include ``'linear'``, ``'affine'``, etc.
        :type map: :class:`str`, optional
        :param polynomial_order: The polynomial order for the kernel function. Defaults to ``2``.
        :type polynomial_order: :class:`float`, optional
        :param regularization: Regularization parameter for the kernel. Defaults to ``1e-8``.
        :type regularization: :class:`numpy.ndarray`, optional
        :param rescale: Whether to rescale the data.
        :type rescale: :class:`bool`, optional
        :param rescale_params: Parameters for data rescaling. Defaults to ``{'max': 1000, 'seed': 42}``.
        :type rescale_params: :class:`dict`, optional
        :param kwargs: Arbitrary keyword arguments.
        :type kwargs: dict

        Returns:
            :class:`numpy.ndarray`: The computed inverse of the transposed gradient.

        Example:

            >>> x_data = np.array([...])
            >>> y_data = np.array([...])
            >>> z_data = np.array([...])
            >>> fx_data = np.array([...])
            >>> inv_transpose_gradient = nablaT_inv(x_data, y_data, z_data, fx_data)
        """
        
        params = {'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization)}
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x,y,z, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)

        return cd.op.nablaT_inv(get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx))
    
    def nablaT_nabla(x, y, fx, kernel_fun = "tensornorm", map = "unitcube", 
                   polynomial_order=2, regularization: float = 1e-8, rescale = False, 
                   rescale_params: dict = {'max': 1000, 'seed':42}, verbose = False, **kwargs):
        """
        Compute the kernel-induced discrete Laplace operator.

        This function calculates the discrete Laplace operator using a kernel method. It computes this 
        operator as the dot product of the transposed gradient vector and the gradient vector, which is 
        consistent with a differential operator that resembles the Laplace operator in its behavior.

        Args:
            x (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Input data points for the Laplace operator computation.
            y (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Secondary data points used in the kernel computation.
            fx (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Function values or responses at the data points.
        :param kernel_fun: The name of the kernel function to use. Options include ``'gaussian'``, ``'tensornorm'``, etc.
        :type kernel_fun: :class:`str`, optional
        :param map: The name of the mapping function to apply. Options include ``'linear'``, ``'affine'``, etc.
        :type map: :class:`str`, optional
        :param polynomial_order: The polynomial order for the kernel function. Defaults to ``2``.
        :type polynomial_order: :class:`float`, optional
        :param regularization: Regularization parameter for the kernel. Defaults to ``1e-8``.
        :type regularization: :class:`numpy.ndarray`, optional
        :param rescale: Whether to rescale the data.
        :type rescale: :class:`bool`, optional
        :param rescale_params: Parameters for data rescaling. Defaults to ``{'max': 1000, 'seed': 42}``.
        :type rescale_params: :class:`dict`, optional
        :param kwargs: Arbitrary keyword arguments.
        :type kwargs: dict
        Returns:
            class:`numpy.ndarray`: The computed discrete Laplace operator values for each point in `x`.

        Note:
            The discrete Laplace operator computed here is not consistent with the "true" Laplace operator, 
            but instead aligns with another differential operator, described as \(-\nabla \cdot (\nabla f\mu)\).

        Example:
            Example usage with NumPy arrays
            
            >>> x_data = np.array([...])
            >>> y_data = np.array([...])
            >>> fx_data = np.array([...])
            >>> laplace_operator = nablaT_nabla(x_data, y_data, fx_data)
        """

        params = {'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization)}
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x,y,x, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)
        
        return cd.op.nablaT_nabla(x=get_matrix(x),y=get_matrix(y),fx=get_matrix(fx))
    
    def nablaT_nabla_inv(x, y, fx, kernel_fun = "tensornorm", map = "unitcube", 
                   polynomial_order=2, regularization: float = 1e-8, rescale = False, rescale_params: dict = {'max': 1000, 'seed':42},
                   verbose = False, **kwargs):
        """
        Args:

        :param kernel_fun: The name of the kernel function to use. Options include ``'gaussian'``, ``'tensornorm'``, etc.
        :type kernel_fun: :class:`str`, optional
        :param map: The name of the mapping function to apply. Options include ``'linear'``, ``'affine'``, etc.
        :type map: :class:`str`, optional
        :param polynomial_order: The polynomial order for the kernel function. Defaults to ``2``.
        :type polynomial_order: :class:`float`, optional
        :param regularization: Regularization parameter for the kernel. Defaults to ``1e-8``.
        :type regularization: :class:`numpy.ndarray`, optional
        :param rescale: Whether to rescale the data.
        :type rescale: :class:`bool`, optional
        :param rescale_params: Parameters for data rescaling. Defaults to ``{'max': 1000, 'seed': 42}``.
        :type rescale_params: :class:`dict`, optional
        :param kwargs: Arbitrary keyword arguments.
        :type kwargs: dict
        """
        
        params = {'rescalekernel':{'max': 1000, 'seed':42},
        'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization),
        'rescale': rescale,
        }
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x,y,x, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)

        return cd.op.nablaT_nabla_inv(get_matrix(x),get_matrix(y),get_matrix(fx))
    
    def Leray_T(x, y, fx, kernel_fun:str = "tensornorm", map:str = "unitcube", 
                   polynomial_order:int=2, regularization: float = 1e-8, 
                   rescale:bool = False, rescale_params: dict = {'max': 1000, 'seed':42}, 
                   verbose = False, **kwargs):
        params = {'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization)}
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x,y,x, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)
        return cd.op.Leray_T(get_matrix(x),get_matrix(y),fx)
    
    def Leray(x, y, fx, kernel_fun:str = "tensornorm", map:str = "unitcube", 
                   polynomial_order:int=2, regularization: float = 1e-8, 
                   rescale:bool = False, rescale_params: dict = {'max': 1000, 'seed':42},  
                   verbose = False, **kwargs):
        """
        Compute the Leray operator for a given set of input matrices.

        Args:
            x (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Input data points.
            y (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Secondary data points.
            fx (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Function values or responses at the data points.
            **kwargs (dict): Arbitrary keyword arguments.

        Returns:
            :class:`numpy.ndarray` or :class:`pandas.DataFrame`: The result of the Leray operator computation.

        Example:
            >>> x_data = np.array([...])
            >>> y_data = np.array([...])
            >>> fx_data = np.array([...])
            >>> leray_result = Leray(x_data, y_data, fx_data)
        """

        params = {'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization)}
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x,y,x, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)

        return cd.op.Leray(get_matrix(x),get_matrix(y),fx)
    
    def hessian(x, z, fx, kernel_fun:str = "tensornorm", map:str = "unitcube", 
                   polynomial_order:int = 2, regularization: float = 1e-8, rescale:bool = False, **kwargs) -> np.ndarray:
        """
        Compute the kernel-induced Hessian matrix of a function.

        Args:
            x (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Input data points where the Hessian matrix is calculated.
            z (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Tertiary data points used in the kernel computation.
            fx (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Function values or responses at the data points in `x`.
        :param kernel_fun: The name of the kernel function to use. Options include ``'gaussian'``, ``'tensornorm'``, etc.
        :type kernel_fun: :class:`str`, optional
        :param map: The name of the mapping function to apply. Options include ``'linear'``, ``'affine'``, etc.
        :type map: :class:`str`, optional
        :param polynomial_order: The polynomial order for the kernel function. Defaults to ``2``.
        :type polynomial_order: :class:`float`, optional
        :param regularization: Regularization parameter for the kernel. Defaults to ``1e-8``.
        :type regularization: :class:`numpy.ndarray`, optional
        :param rescale: Whether to rescale the data.
        :type rescale: :class:`bool`, optional
        :param rescale_params: Parameters for data rescaling. Defaults to ``{'max': 1000, 'seed': 42}``.
        :type rescale_params: :class:`dict`, optional
        :param kwargs: Arbitrary keyword arguments.
        :type kwargs: dict

        Returns:
            :class:`numpy.ndarray`: The computed Hessian matrix of the function.

        Note:
            The function computes the Hessian matrix for each input data point. If the 'fx' parameter is provided,
            the function computes a modified Hessian matrix using this additional information.

        Example:

            >>> x_data = np.array([...])
            >>> z_data = np.array([...])
            >>> fx_data = np.array([...])
            >>> hessian_matrix = hessian(x_data, z_data, fx_data)
        """
        indices = distance_labelling(x = z, y = x)
        grad = diffops.nabla(x = x, y = x, z = z, fx = [], kernel_fun = kernel_fun, map = map, 
                   polynomial_order=polynomial_order, regularization = regularization, rescale = rescale)
        N_X = x.shape[0]
        #N_Z = z.shape[0]
        D = x.shape[1]
        gradT = np.zeros([N_X,D,N_X])
        def helper(d): gradT[:,d,:] = grad[:,d,:].T.copy()
        [helper(d) for d in range(D)]
        if fx is not None: 
            fx = get_matrix(fx)
            # gradf = np.squeeze(op.nabla(x=x, y=y, z=x, fx=np.squeeze(fx),**kwargs))
            # spot_baskets = z[:,1]
            # thetas = gradf[:,0]
            # deltas = gradf[:,1]
            # multi_plot([[z,thetas],[z,deltas]],plotD,projection='3d',loc = 'upper left',prop={'size': 3},mp_ncols=2,**kwargs)
            out = np.zeros( [N_X, D,D, fx.shape[1]])
            for d in itertools.product(range(D), range(D)):
                debug = grad[:,d[1],:] @ get_matrix(fx)
                # multi_plot([[z,debug]],plotD,projection='3d',loc = 'upper left',prop={'size': 3},mp_ncols=2,**kwargs)
                mat = gradT[:,d[0],:]
                out[:,d[0], d[1],:]  = -mat @ debug
                # multi_plot([[z,out[:,d[0], d[1],:]]],plotD,projection='3d',loc = 'upper left',prop={'size': 3},mp_ncols=2,**kwargs)
            out = out[indices,:,:,:]
            # codpy_hessians = np.squeeze(out)
            # thetas = codpy_hessians[:,0,0]
            # gammas = codpy_hessians[:,1,1]
            # crossed = codpy_hessians[:,1,0]
            # multi_plot([[z,fx],[z,thetas],[z,gammas],[z,crossed]],plotD,projection='3d',loc = 'upper left',prop={'size': 3},mp_ncols=2,**kwargs)

            return out
        else: 
            hess = np.zeros( [N_X, D,D, N_X])
            for d in itertools.product(range(D), range(D)):
                hess[:,d[0], d[1],:]  = -gradT[:,d[0],:] @ grad[:,d[1],:]
                # test = hess[:,d[0]*D + d[1],:]
            return hess[indices,:,:,:]


class _factories:
    def get_kernel_factory_keys():
        return cd.factories.kernel_factory_keys()
    def get_map_factory_keys():
        return cd.factories.maps_factory_keys()

class kernel:
    def rescale(x=[],y=[],z=[],max_=None,seed=42):
        if max_ is not None:
            x,y,z = random_select(x=x,xmax = max_,seed=seed),random_select(x=y,xmax = max_,seed=seed),random_select(x=z,xmax = max_,seed=seed)
        x,y,z = get_matrix(x),get_matrix(y),get_matrix(z)
        cd.kernel.rescale(x,y,z)
    def get_kernel_ptr():
        return cd.get_kernel_ptr()
    def set_kernel_ptr(kernel_ptr):
        cd.set_kernel_ptr(kernel_ptr)
    def pipekernel_ptr(kernel_ptr):
        cd.kernel.pipekernel_ptr(kernel_ptr)
    def pipekernel_fun(kernel_fun, regularization = 1e-8):
        kern1 = kernel.get_kernel_ptr()
        kernel_fun()
        kern2 = kernel.get_kernel_ptr()
        kernel.set_kernel_ptr(kern1)
        kernel.pipekernel_ptr(kern2)
        cd.kernel.set_regularization(regularization)
    def init(x = [], y = [], z = [], **kwargs):
        set_codpykernel = kwargs.get('set_codpykernel',None)
        if set_codpykernel is not None: set_codpykernel()
        rescale = kwargs.get('rescale',False)
        if (rescale): kernel.rescale(x, y, z, **kwargs)
    def map(x):
        return cd.kernel.map(get_matrix(x))
    def get_map_ptr():
        return cd.kernel.get_map_ptr()
    def set_map_ptr(map_ptr):
        cd.kernel.set_map_ptr(map_ptr)


class map_setters:
    class set:    
        def __init__(self,strings):
            self.strings = strings
        def __call__(self,**kwargs): set_map(self.strings,kwargs)
    def set_linear_map(**kwargs): 
        """
        Set a linear map for the kernel.

        Args:
        - **kwargs: Arbitrary keyword arguments for the linear map configuration.
        """
        cd.kernel.set_map("linear_map",kwargs)
    def set_affine_map(**kwargs): 
        """
        Set an affine map for the kernel.

        Args:
        - **kwargs: Arbitrary keyword arguments for the affine map configuration.
        """
        cd.kernel.set_map("affine_map",kwargs)
    def set_log_map(**kwargs): 
        """
        Set a logarithmic map for the kernel.

        Args:
        - **kwargs: Arbitrary keyword arguments for the log map configuration.
        """
        cd.kernel.set_map("log",kwargs)
    def set_exp_map(**kwargs): 
        """
        Set an exponential map for the kernel.

        Args:
        - **kwargs: Arbitrary keyword arguments for the exponential map configuration.
        """
        cd.kernel.set_map("exp",kwargs)
    def set_scale_std_map(**kwargs): 
        """
        Set a standard scaling map for the kernel.

        Args:
        - **kwargs: Arbitrary keyword arguments for the scale standard map configuration.
        """
        cd.kernel.set_map("scale_std",kwargs)
    def set_erf_map(**kwargs): 
        """
        Set an error function (ERF) map for the kernel.

        Args:
        - **kwargs: Arbitrary keyword arguments for the ERF map configuration.
        """
        cd.kernel.set_map("scale_to_erf",kwargs)
    def set_erfinv_map(**kwargs):
        """
        Set an inverse error function (ERF) map for the kernel.

        Args:
        - **kwargs: Arbitrary keyword arguments for the inverse ERF map configuration.
        """
        cd.kernel.set_map("scale_to_erfinv",kwargs)
    def set_scale_factor_map(**kwargs): 
        """
        Set a scaling factor map for the kernel.

        Args:
        - **kwargs: Arbitrary keyword arguments for the scale factor map configuration.
        """
        cd.kernel.set_map("scale_factor",kwargs)
    def set_scale_factor_helper(**kwargs): 
        """
        Helper function to set a scaling factor map with specified bandwidth.

        Args:
        - **kwargs: Arbitrary keyword arguments, including 'bandwidth' for the scale factor map configuration.
        """
        return lambda : partial(map_setters.set_scale_factor_map, **{'h':str(kwargs["bandwidth"])})()
    def set_unitcube_map(**kwargs): 
        """
        Set a unit cube map for the kernel.

        Args:
        - **kwargs: Arbitrary keyword arguments for the unit cube map configuration.
        """
        cd.kernel.set_map("scale_to_unitcube",kwargs)
    def set_grid_map(**kwargs): 
        """
        Set a grid map for the kernel.

        Args:
        - **kwargs: Arbitrary keyword arguments for the grid map configuration.
        """
        cd.kernel.set_map("map_to_grid",kwargs)
    def set_mean_distance_map(**kwargs): 
        """
        Set a mean distance map for the kernel.

        Args:
        - **kwargs: Arbitrary keyword arguments for the mean distance map configuration.
        """
        cd.kernel.set_map("scale_to_mean_distance",kwargs)
    def set_min_distance_map(**kwargs): 
        """
        Set a minimum distance map for the kernel.

        Args:
        - **kwargs: Arbitrary keyword arguments for the minimum distance map configuration.
        """
        cd.kernel.set_map("scale_to_min_distance",kwargs)
    def set_standard_mean_map(**kwargs):
        """
        Set a standard mean map pipeline for the kernel.

        This function sets a mean distance map and then pipes it through the erf-inverse and unit cube maps.

        Args:
        - **kwargs: Arbitrary keyword arguments for the standard mean map configuration.
        """
        map_setters.set_mean_distance_map(**kwargs)
        _pipe__map_setters.pipe_erfinv_map()
        _pipe__map_setters.pipe_unitcube_map()
    def set_standard_min_map(**kwargs):
        """
        Set a standard minimum map pipeline for the kernel.

        This function sets a minimum distance map and then pipes it through the erf-inverse and unit cube maps.

        Args:
        - **kwargs: Arbitrary keyword arguments for the standard minimum map configuration.
        """
        map_setters.set_min_distance_map(**kwargs)
        _pipe__map_setters.pipe_erfinv_map()
        _pipe__map_setters.pipe_unitcube_map()
    def set_unitcube_min_map(**kwargs):
        """
        Set a unit cube minimum map pipeline for the kernel.

        This function sets a minimum distance map and then pipes it through the unit cube map.

        Args:
        - **kwargs: Arbitrary keyword arguments for the unit cube minimum map configuration.
        """
        map_setters.set_min_distance_map(**kwargs)
        _pipe__map_setters.pipe_unitcube_map()
    def set_unitcube_erfinv_map(**kwargs):
        """
        Set a unit cube erf-inverse map pipeline for the kernel.

        This function sets an erf-inverse map and then pipes it through the unit cube map.

        Args:
        - **kwargs: Arbitrary keyword arguments for the unit cube erf-inverse map configuration.
        """
        map_setters.set_erfinv_map()
        _pipe__map_setters.pipe_unitcube_map()
    def set_unitcube_mean_map(**kwargs):
        """
        Set a unit cube mean map pipeline for the kernel.

        This function sets a mean distance map and then pipes it through the unit cube map.

        Args:
        - **kwargs: Arbitrary keyword arguments for the unit cube mean map configuration.
        """
        map_setters.set_mean_distance_map(**kwargs)
        _pipe__map_setters.pipe_unitcube_map()
    def map_helper(map_setter, **kwargs): 
        """
        Helper function to create a custom map setting function.

        This function creates a partial function for a specified map setter with provided arguments.

        Args:
        - map_setter (function): The map setter function to be used.
        - **kwargs: Arbitrary keyword arguments for the map setter function.
        """
        return partial(map_setter, kwargs)

def check_map_strings(strings): 
    if isinstance(strings,list): [check_map_strings(s) for s in strings]
    else: 
        ok = strings in _factories.get_map_factory_keys()
        if not ok: 
            raise NameError("unknown map:"+strings)
        
def set_map(strings,check_=True,kwargs={}):
    if check_:
        if kernel.get_kernel_ptr() == None: 
            raise AssertionError("set a kernel first, see set_kernel")
        check_map_strings(strings)
    if isinstance(strings,list):
        ss = strings.copy()
        cd.kernel.set_map(ss.pop(0),kwargs)
        [_pipe__map_setters.pipe(s) for s in ss]
    else: cd.kernel.set_map(strings,kwargs)

def checkkernel_strings(strings): 
    ok = strings in _factories.get_kernel_factory_keys()
    if not ok: 
        raise NameError("unknown kernel:"+strings)
def set_kernel(strings,reg=1e-8,check_=True):
    if check_: checkkernel_strings(strings)
    cd.set_kernel(strings)
    cd.kernel.set_regularization(reg)

class kernel_setters:
    def kernel_helper(setter, polynomial_order:int = 0,regularization:float = 1e-8,set_map = None):
        return partial(setter, polynomial_order,regularization,set_map)
    def set_kernel(kernel_string,polynomial_order:int = 2,regularization:float = 1e-8,set_map = None):
        """
        Set the kernel function with specified parameters.

        This method configures the kernel function used in the calculations. It allows setting
        the type of kernel, its polynomial order, regularization factor, and an optional mapping function.

        Args:
        - kernel_string (str): The name of the kernel function to use.
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function to apply.

        The method configures the kernel and its associated parameters, preparing it for use 
        in subsequent calculations.
        """
        cd.kernel.set_polynomial_order(polynomial_order)
        cd.set_kernel(kernel_string)
        if (set_map) : set_map()
        if (polynomial_order > 0):
            linearkernel = kernel_setters.kernel_helper(setter = kernel_setters.set_linear_regressorkernel,polynomial_order = polynomial_order,regularization = regularization,set_map = None)
            kernel.pipekernel_fun(linearkernel,regularization)
        cd.kernel.set_regularization(regularization)

    def set_linear_regressorkernel(polynomial_order:int = 2,regularization:float = 1e-8,set_map = None):
        """
        Set the linear regression kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function to apply.
        """
        cd.kernel.set_polynomial_order(polynomial_order)
        cd.set_kernel("linear_regressor")
        cd.kernel.set_regularization(regularization)
        if (set_map) : set_map()

    def set_absnormkernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = None): 
        """
        Set the absolute norm kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function to apply.
        """
        kernel_setters.set_kernel("absnorm",polynomial_order,regularization,set_map)
    def set_tensornorm_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = None): 
        """
        Set the tensor norm kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function to apply.
        """
        kernel_setters.set_kernel("tensornorm",polynomial_order,regularization,set_map)
    def set_gaussian_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = None): 
        """
        Set the Gaussian kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function to apply.
        """
        kernel_setters.set_kernel("gaussian",polynomial_order,regularization,set_map)
    def set_matern_tensor_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = None): 
        """
        Set the Matérn tensor kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function to apply.
        """
        kernel_setters.set_kernel("materntensor",polynomial_order,regularization,set_map)
    default_multiquadricnormkernel_map = partial(map_setters.set_standard_mean_map, distance ='norm2')
    def set_multiquadricnorm_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = default_multiquadricnormkernel_map): 
        """
        Set the multi-quadric norm kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function defined as `default_multiquadricnormkernel_map`.
        """
        kernel_setters.set_kernel("multiquadricnorm",polynomial_order,regularization,set_map)
    default_multiquadrictensorkernel_map = partial(map_setters.set_standard_min_map, distance ='normifty')
    def set_multiquadrictensor_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = default_multiquadrictensorkernel_map): 
        """
        Set the multi-quadric tensor kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function defined as `default_multiquadrictensorkernel_map`.
        """
        kernel_setters.set_kernel("multiquadrictensor",polynomial_order,regularization,set_map)
    default_sincardtensorkernel_map = partial(map_setters.set_min_distance_map, distance ='normifty')
    def set_sincardtensor_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = default_sincardtensorkernel_map):
        """
        Set the sinc cardinal tensor kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function defined as `default_sincardtensorkernel_map`.
        """
        kernel_setters.set_kernel("sincardtensor",polynomial_order,regularization,set_map)        
    default_sincardsquaretensorkernel_map = partial(map_setters.set_min_distance_map, distance ='normifty')
    def set_sincardsquaretensor_kernel(polynomial_order:int = 0,regularization:float = 0,set_map = default_sincardsquaretensorkernel_map): 
        """
        Set the sinc cardinal square tensor kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function defined as `default_sincardsquaretensorkernel_map`.
        """
        kernel_setters.set_kernel("sincardsquaretensor",polynomial_order,regularization,set_map)        
    def set_dotproduct_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = None):
        """
        Set the dot product kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function to apply.
        """
        kernel_setters.set_kernel("DotProduct",polynomial_order,regularization,set_map)        
    def set_gaussianper_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = None):
        """
        Set the Gaussian periodic kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function to apply.
        """
        kernel_setters.set_kernel("gaussianper",polynomial_order,regularization,set_map)     
    def set_matern_norm_kernel(polynomial_order:int = 2,regularization:float = 1e-8,set_map = map_setters.set_mean_distance_map): 
        """
        Set the Matérn norm kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function defined as `map_setters.set_mean_distance_map`.
        """
        kernel_setters.set_kernel("maternnorm",polynomial_order,regularization,set_map) 
    def set_scalar_product_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = None): 
        """
        Set the scalar product kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function to apply.
        """
        kernel_setters.set_kernel("scalar_product",polynomial_order,regularization,set_map)

class _pipe__map_setters:
    def pipe(s,**kwargs):
        cd.kernel.pipe_map(s,kwargs)
    def pipe_log_map(**kwargs):_pipe__map_setters.pipe("log",**kwargs)
    def pipe_exp_map(**kwargs):_pipe__map_setters.pipe("exp",**kwargs)
    def pipe_linear_map(**kwargs):_pipe__map_setters.pipe("linear_map",**kwargs)
    def pipe_affine_map(**kwargs):_pipe__map_setters.pipe_affine_map("affine_map",**kwargs)
    def pipe_scale_std_map(**kwargs):_pipe__map_setters.pipe("scale_std",**kwargs)
    def pipe_erf_map(**kwargs):_pipe__map_setters.pipe("scale_to_erf",**kwargs)
    def pipe_erfinv_map(**kwargs):_pipe__map_setters.pipe("scale_to_erfinv",**kwargs)
    def pipe_unitcube_map(**kwargs):_pipe__map_setters.pipe("scale_to_unitcube",**kwargs)
    def pipe_mean_distance_map(**kwargs):_pipe__map_setters.pipe("scale_to_mean_distance",**kwargs)
    def pipe_min_distance_map(**kwargs):_pipe__map_setters.pipe("scale_to_min_distance",**kwargs)




kernel_settings = {
    "linear": lambda: lambda polynomial_order, regularization, map_func: kernel_setters.kernel_helper(
        kernel_setters.set_linear_regressorkernel,  polynomial_order, regularization, map_func
    ),
    "gaussian": lambda: lambda polynomial_order, regularization, map_func: kernel_setters.kernel_helper(
        kernel_setters.set_gaussiankernel,  polynomial_order, regularization, map_func
    ),
    "tensornorm": lambda: lambda polynomial_order, regularization, map_func: kernel_setters.kernel_helper(
        kernel_setters.set_tensornormkernel, polynomial_order, regularization, map_func
    ),
    "absnorm": lambda: lambda polynomial_order, regularization, map_func: kernel_setters.kernel_helper(
        kernel_setters.set_absnormkernel, polynomial_order, regularization, map_func
    ),
    "matern": lambda: lambda polynomial_order, regularization, map_func: kernel_setters.kernel_helper(
        kernel_setters.set_matern_tensorkernel, polynomial_order, regularization, map_func
    ),
    "multiquadricnorm": lambda: lambda polynomial_order, regularization, map_func: kernel_setters.kernel_helper(
        kernel_setters.set_multiquadricnormkernel, polynomial_order, regularization, map_func
    ),
    "multiquadrictensor": lambda: lambda polynomial_order, regularization, map_func: kernel_setters.kernel_helper(
        kernel_setters.set_multiquadrictensorkernel, polynomial_order, regularization, map_func
    ),
    "sincardtensor": lambda: lambda polynomial_order, regularization, map_func: kernel_setters.kernel_helper(
        kernel_setters.set_sincardtensorkernel, polynomial_order, regularization, map_func
    ),
    "sincardsquaretensor": lambda: lambda polynomial_order, regularization, map_func: kernel_setters.kernel_helper(
        kernel_setters.set_sincardsquaretensorkernel, polynomial_order, regularization, map_func
    ),
    "dotproduct": lambda: lambda polynomial_order, regularization, map_func: kernel_setters.kernel_helper(
        kernel_setters.set_dotproductkernel, polynomial_order, regularization, map_func
    ),
    "gaussianper": lambda: lambda polynomial_order, regularization, map_func: kernel_setters.kernel_helper(
        kernel_setters.set_gaussianperkernel, polynomial_order, regularization, map_func
    ),
    "maternnorm": lambda: lambda polynomial_order, regularization, map_func: kernel_setters.kernel_helper(
        kernel_setters.set_matern_normkernel, polynomial_order, regularization, map_func
    ),
    "scalarproduct": lambda: lambda polynomial_order, regularization, map_func: kernel_setters.kernel_helper(
        kernel_setters.set_scalar_productkernel, polynomial_order, regularization, map_func
    )
}

_map_settings = {
    "linear": map_setters.set_linear_map,
    "affine": map_setters.set_affine_map,
    "log": map_setters.set_log_map,
    "exp": map_setters.set_exp_map,
    "scalestd": map_setters.set_scale_std_map,
    "erf": map_setters.set_erf_map,
    "erfinv": map_setters.set_erfinv_map,
    "scalefactor": map_setters.set_scale_factor_map,
    "bandwidth": map_setters.set_scale_factor_helper,
    "grid": map_setters.set_grid_map,
    "unitcube": map_setters.set_unitcube_map,
    "meandistance": map_setters.set_mean_distance_map,
    "mindistance": map_setters.set_min_distance_map,
    "standardmin": map_setters.set_standard_mean_map,
    "standardmean": map_setters.set_standard_mean_map
}
def _requires_bandwidth(map_name: str) -> bool:
    bandwidth_required_maps = {"scale_factor"}
    return map_name in bandwidth_required_maps

def _requires_rescale(map_name: str) -> bool:
    maps_needing_rescale = {
        "unitcube",
        "meandistance",
        "mindistance",
        "standardmin",
        "standardmean"
    }
    return map_name in maps_needing_rescale

def kernel_helper2(kernel, map, polynomial_order=0, regularization=1e-8, bandwidth = 1.0):
    """
    Set the kernel function with specified parameters using string identifiers.

    Args:
        kernel (str): The name of the kernel function to use.
        map (str): The name of the mapping function to use.
        polynomial_order (int): The polynomial order for the kernel function.
        regularization (float): The regularization parameter for the kernel.

    Returns:
        The configured kernel function.
    """
    kernel_func_creator = kernel_settings.get(kernel)
    map_func = _map_settings.get(map)

    if not kernel_func_creator:
        raise ValueError(f"Kernel '{kernel}' not recognized.")
    if map is not None:
        map_func = _map_settings.get(map)
        if not map_func:
            raise ValueError(f"Map '{map}' not recognized.")
        if _requires_bandwidth(map):
            map_func = map_func(bandwidth=bandwidth)
    else:
        map_func = None

    kernel_func = kernel_func_creator()

    return kernel_func(polynomial_order, regularization, map_func)

   
class _Cache:
    """
    A class for caching and computing kernel matrices using a provided positive definite (pd) kernel.
    This class allows for the computation of a kernel matrix based on training features, responses, and internal parameters. 

    Attributes:
        params (dict): A dictionary of parameters used for kernel matrix computations.
        x (np.array): Training feature data.
        fx (np.array): Training response data.
        y (np.array): Internal parameter, set to x by default.
        knm_inv (np.array): Inverse kernel matrix.
        order (int): Polynomial order of the kernel.
        reg (float): Regularization parameter of the kernel.
        kernel (function): Pointer to the kernel function.

    Methods:
        __call__(z, fx): Computes the kernel matrix based on test input features z and internal parameter y.
    """
    params = {}
    def __init__(self,**kwargs):
        self.params = kwargs.copy()
        self.x = kwargs['x'].copy()
        self.fx = kwargs.get('fx',None)
        self.y = kwargs.get('y',self.x)
        if self.fx is not None: self.fx= self.fx.copy()
        self.params['x'],self.params['y'],self.params['fx'] = self.x,self.y,self.fx
        self.knm_inv =  op.Knm_inv(**self.params)
        self.order = cd.kernel.get_polynomial_order()
        self.reg = cd.kernel.get_regularization()
        self.kernel =  kernel.get_kernel_ptr()
    def __call__(self, **kwargs):
        y,z = get_matrix(self.params['y']),get_matrix(kwargs['z'])
        # return op.projection(**{**self.params,**{'x':y,'y':y,'z':z,'fx':kwargs['fx']}}) #for debug
        kernel.set_kernel_ptr(self.kernel)
        cd.kernel.set_polynomial_order(self.order)
        cd.kernel.set_regularization(self.reg)
        Knm= op.Knm(**{**self.params,**{'x':z,'y':y,'fy':self.knm_inv,'set_codpykernel':None,'rescale':False}})
        # test = self.fx - op.projection(**{**self.params,**{'z':z,'fx':self.fx,'set_codpykernel':None,'rescale':False}})  
        return Knm

if __name__ == "__main__":
    set_kernel("tensornorm",1e-2)
    test = cd.kernel.get_regularization()
    set_map("scale_to_unitcube")
    x = np.random.randn(10, 2)
    fx = np.random.randn(10, 3)
    kernel.rescale(x)
    Knm = op.Knm(x=x,y=x)
    # Knm_inv = lalg.cholesky(x=Knm,eps=1e-2)
    Knm_inv = cd.lalg.cholesky(Knm,1e-8)
    Kinv = op.Knm_inv(x=x,y=x,fx=fx)
    Kinv1 = np.linalg.solve(x.T @ x, fx.T).T
    pass
