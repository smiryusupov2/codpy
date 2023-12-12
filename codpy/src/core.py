import pandas as pd
import numpy as np
from functools import partial
import codpydll
import codpypyd as cd
from codpy.utils.selection import column_selector
from codpy.utils.random import random_select, random_select_interface
from codpy.utils.dictionary import *
from codpy.utils.data_conversion import get_matrix
from codpy.utils.utils import pad_axis, softmaxindice, softminindice


def _get_codpy_param():
    return  {'rescale_kernel':{'max': 1000, 'seed':42},
    'sharp_discrepancy':{'max': 1000, 'seed':42},
    'discrepancy':{'max': 1000, 'seed':42},
    'validator_compute': ['accuracy_score','discrepancy_error','norm'],
    'set_codpy_kernel' : kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 2,1e-8 ,map_setters.set_unitcube_map),
    'rescale': True,
    }

class _codpy_param_getter:
    def get_params(**kwargs) : return kwargs.get('codpy',{})
    def get_kernel_fun(**kwargs): return _codpy_param_getter.get_params(**kwargs)['set_kernel']


class op:
    def _get_xyzfxreg(kwargs):
        x,z,fx,reg = kwargs.get('x',[]),kwargs.get('z',[]),kwargs.get('fx',[]),kwargs.get('reg',[])
        y = kwargs.get('y',x)
        return x,y,z,fx,reg
    # def sum_axis(x,axis=-1):
    #     if axis != -1: 
    #         out = np.sum(x,axis)
    #     else: out=x
    #     return out

    def projection(**kwargs):
        """
        Performs projection in kernel regression for efficient computation, targeting a lower sampling space.

        This function optimizes kernel regression computations by projecting the training data (x, fx) onto 
        a representative subset (y), thus reducing the computational burden associated with large datasets.
        It facilitates both interpolation within the training data domain and extrapolation outside this domain.

        :param kwargs: Arbitrary keyword arguments, including:
        :type kwargs: dict
        :param x: Training feature data.
        :type x: :class:`numpy.ndarray` or :class:`pandas.DataFrame`
        :param y: Representative subset of 'x', used for projection.
        :type y: :class:`numpy.ndarray` or :class:`pandas.DataFrame`
        :param z: Test feature data for prediction.
        :type z: :class:`numpy.ndarray`, :class:`pandas.DataFrame`, or list
        :param fx: Training response data corresponding to 'x'.
        :type fx: :class:`numpy.ndarray` or :class:`pandas.DataFrame`
        :param reg: Regularization parameters, if needed.
        :type reg: :class:`numpy.ndarray`, optional

        :returns: The projected responses for the test data 'z'.
        :rtype: :class:`numpy.ndarray` or :class:`pandas.DataFrame`
        """
        projection_format_switchDict = { pd.DataFrame: lambda **kwargs :  projection_dataframe(**kwargs) }
        z = kwargs['z']
        if isinstance(z,list): 
            def fun(zi):
                kwargs['z'] = zi
                return op.projection(**kwargs)
            out = [fun(zi) for zi in z]
            return out
        def projection_dataframe(**kwargs):
            x,y,z,fx,reg = op._get_xyzfxreg(kwargs)
            x,y,z = column_selector([x,y,z],**kwargs)
            x_,y_,z_,fx_,reg_ = get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx),get_matrix(reg)
            f_z = cd.op.projection(x_,y_,z_,fx_,reg_)
            # f_z = cd.op.projection(get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx),get_matrix(reg))
            if isinstance(fx,pd.DataFrame): 
                f_z = pd.DataFrame(f_z,columns = list(fx.columns))
                if isinstance(z,pd.DataFrame): f_z.index = z.index

            return f_z

        kernel.init(**kwargs)
        type_debug = type(kwargs.get('fx',[]))

        def debug_fun(**kwargs):
            x,y,z,fx,reg = kwargs.get('x',[]),kwargs.get('y',[]),kwargs.get('z',[]),kwargs.get('fx',[]),kwargs.get('reg',[])
            f_z = cd.op.projection(get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx),get_matrix(reg))
            return f_z

        method = projection_format_switchDict.get(type_debug,debug_fun)
        f_z = method(**kwargs)

        # if kwargs.get('save_cd_data',False):
        #     class Object:pass
        #     temp = Object
        #     temp.x ,temp.y,temp.z,temp.fx,temp.f_z,temp.fz = x ,y,z,fx,f_z,kwargs.get("fz",[])
        #     data_generator.save_cd_data(temp,**kwargs)
        return f_z

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

        # if kwargs.get('save_cd_data',False):
        #     class Object:pass
        #     temp = Object
        #     temp.x ,temp.y,temp.z,temp.fx,temp.f_z,temp.fz = x ,y,z,fx,f_z,kwargs.get("fz",[])
        #     data_generator.save_cd_data(temp,**kwargs)
        return f_z

    def extrapolation(**kwargs):
        """
        Performs extrapolation in the context of kernel regression.

        This method leverages the kernel regression framework to extrapolate values between data points. 

        Args:
            **kwargs: Data points 'x', 'z', and any other parameters required for 
                    the projection operation.
        Returns:
            Extrapolated responses.
        """
        return op.projection(**{**kwargs,**{'y':kwargs['x']}})

    def interpolation(**kwargs):
        """
        Performs interpolation in the context of kernel regression.

        This method leverages the kernel regression framework to interpolate values between data points. 

        Args:
            **kwargs: Data points 'x', 'z', and any other parameters required for 
                    the projection operation.

        Returns:
            Interpolated responses.
        """
        return op.projection(**{**kwargs,**{'y':kwargs['z']}})

    def norm(**kwargs):
        """
        Calculate the kernel-induced norm based on the provided matrices.

        This function computes a norm projection using the kernel initialization parameters. 
        It supports flexible argument input through keyword arguments.

        Args:
            x (list, optional): The first matrix. Defaults to an empty list.
            y (list, optional): The second matrix. Defaults to a list containing `x`.
            z (list, optional): The third matrix. Defaults to an empty list.
            fx (list, optional): The fourth matrix. Defaults to an empty list.
            reg (list, optional): Regularization parameters. Defaults to an empty list.

        The function uses `get_matrix` to convert the input lists into appropriate matrix forms 
        before performing the norm projection.

        Returns:
            The result of the norm projection, calculated using the input matrices.
        """
        kernel.init(**kwargs)
        x,y,z,fx,reg = kwargs.get('x',[]),kwargs.get('y',[x]),kwargs.get('z',[]),kwargs.get('fx',[]),kwargs.get('reg',[])
        return cd.tools.norm_projection(get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx))

    def coefficients(**kwargs) -> np.ndarray:
        """
        Computes the regressors or coefficients for kernelized regression, using a specified PD kernel.

        This method initializes the kernel function with the given parameters and then 
        computes the regression coefficients based on the input datasets. 

        Args:
            x (np.array): The input data points for which the regression coefficients are computed.
            y (np.array): Internal parameter, can be y = x.
            fx (np.array, optional): Responses associated with 'x'.

        Returns:
            np.array: The computed regression coefficients or regressors that model the 
            relationship between the input data 'x' and the target 'y'.
        """
        kernel.init(**kwargs)
        x, y, _, fx, _ = op._get_xyzfxreg(kwargs)
        return cd.op.coefficients(get_matrix(x),get_matrix(y),get_matrix(fx))
    def Knm(**kwargs) -> np.ndarray:
        """
        Computes the kernel matrix induced by a positive definite (pd) kernel.

        This method calculates the kernel matrix using the specified pd kernel function. 

        Args:
            x (np.array): The first set of data points.
            y (np.array): The second set of data points. If not provided, 'x' is used.
            fy (np.array): Precomputed inverse kernel matrix K(y,y).
        Returns:
            np.array: The computed kernel matrix, representing the kernel-induced distances or similarities between the data points in 'x' and 'y'.
        """
        kernel.init(**kwargs)
        x,y, _, _, _ = op._get_xyzfxreg(kwargs)
        fy = kwargs.get('fy',[])
        x,y,fy = get_matrix(x),get_matrix(y),get_matrix(fy)
        debug = cd.op.Knm(x,y,fy)
        return debug
    
    def Dnm(**kwargs) -> np.ndarray:
        """
        Computes a distance matrix induced by a positive definite (pd) kernel.

        This function calculates the distance matrix between sets of data points
        x and y based on a specified pd kernel.

        Args:
            x (np.array): The first set of data points.
            y (np.array): The second set of data points. If not provided, defaults to x.
            distance (function, optional): a name of distance function.

        Returns:
            np.array: A distance matrix representing the distances between each pair of points in x and y as induced by the pd kernel.
        """
        x, y, _, _, _ = op._get_xyzfxreg(kwargs)
        x,y = column_selector(x,**kwargs),column_selector(y,**kwargs)
        x,y = pad_axis(x,y)
        kernel.init(**kwargs)
        distance = kwargs.get('distance',None)
        if distance is not None:
            return cd.op.Dnm(get_matrix(x),get_matrix(y),{'distance':distance})
        return cd.op.Dnm(get_matrix(x),get_matrix(y))

def distance_labelling(**kwargs) -> np.ndarray:
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

    Returns:
        np.array: An array of labelled distances between the data points in x and y.
    """
    x,y,axis = kwargs['x'],kwargs['y'],kwargs.get('axis',1)
    # print('######','distance_labelling','######')
    x, y = column_selector(x,**kwargs),column_selector(y,**kwargs)
    D = op.Dnm(**{**kwargs,**{'x':x,'y':y}})
    if kwargs.get('max',False) : return softmaxindice(D,**{**kwargs,**{'x':x,'y':y,'axis':axis}})
    return softminindice(D,**{**kwargs,**{'x':x,'y':y,'axis':axis}})

def discrepancy(disc_type="raw", **kwargs):
    x, y, z, _, _ = kwargs.get('x',[]),kwargs.get('y',[]),kwargs.get('z',[]),kwargs.get('fx',[]),kwargs.get('reg',[])

    if 'discrepancy:xmax' in kwargs: x= random_select_interface(xmaxlabel = 'discrepancy:xmax', seedlabel = 'discrepancy:seed',**{**kwargs,**{'x':x}})
    if 'discrepancy:ymax' in kwargs: y= random_select_interface(xmaxlabel = 'discrepancy:ymax', seedlabel = 'discrepancy:seed',**{**kwargs,**{'x':y}})
    if 'discrepancy:zmax' in kwargs: z= random_select_interface(xmaxlabel = 'discrepancy:zmax', seedlabel = 'discrepancy:seed',**{**kwargs,**{'x':z}})
    if 'discrepancy:nmax' in kwargs:
        nmax = int(kwargs.get('discrepancy:nmax'))
        if len(x) + 2 * len(y) + len(z) > nmax: return np.NaN
    kernel.init(**kwargs)
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
    def __init__(self,x, y=[],**kwargs):
        self.Nx = len(x)
        self.x = x.copy()
        kwargs['x'],kwargs['y']=x,y
        kernel.init(**kwargs)
        kwargs['y'] = x
        self.Kxx = op.Knm(**kwargs)
        self.Kxx = np.sum(self.Kxx ) / (self.Nx*self.Nx)
        pass
    def eval(self,ys,**kwargs):
        N = len(ys)
        Kxy = op.Knm(x = ys, y = self.x)/(self.Nx)
        out = np.zeros([N])
        for n in range(N):
            out[n] = 1.+self.Kxx-2.*np.sum(Kxy[n])
        return out

class Cache:
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
        Knm= op.Knm(**{**self.params,**{'x':z,'y':y,'fy':self.knm_inv,'set_codpy_kernel':None,'rescale':False}})
        # test = self.fx - op.projection(**{**self.params,**{'z':z,'fx':self.fx,'set_codpy_kernel':None,'rescale':False}})  
        return Knm
    
class diffops:
    """
    A class for computing various kernel-induced differential operators.

    This class includes methods for gradient operations, inversions, and Hessian calculations.
    It's designed to work with data in the form of NumPy arrays or similar data structures.
    The methods of this class are useful in tasks involving mathematical analysis, machine learning,
    and data processing where differential operations are required.

    Methods:
        nabla_Knm(**kwargs): Compute the gradient of the kernel matrix between two sets of points.
        Knm_inv(**kwargs): Compute the inverse of the kernel matrix.
        nabla(**kwargs): Compute the gradient operation.
        nabla_inv(**kwargs): Compute the inverse of the gradient operation.
        nablaT(**kwargs): Compute the transpose of the gradient operation.
        nablaT_inv(**kwargs): Compute the inverse of the transpose of the gradient operation.
        nablaT_nabla(**kwargs): Compute the product of the transpose of the gradient and the gradient.
        nablaT_nabla_inv(**kwargs): Compute the inverse of the product of the transpose of the gradient and the gradient.
        Leray_T(**kwargs): Compute the Leray projection transpose operation.
        Leray(**kwargs): Compute the Leray projection operation.
        hessian(**kwargs): Compute the Hessian matrix for a given function and its gradient.

    Each method requires certain keyword arguments, typically including:
    x, y, z: Input data points or distributions.
    fx: Function values at the data points.
    reg: Regularization parameters.

    Example:
        Example usage for computing the gradient of a kernel matrix
        
        >>> x_data = np.array([...])
        >>> y_data = np.array([...])
        >>> gradient_kernel_matrix = diffops().nabla_Knm(x=x_data, y=y_data)
    """
    def nabla_Knm(**kwargs):
        kernel.init(**kwargs)
        x, y, _, _, _ = op._get_xyzfxreg(kwargs)
        return cd.op.nabla_Knm(get_matrix(x),get_matrix(y))
    def Knm_inv(**kwargs):
        kernel.init(**kwargs)
        x,y, _,fx,reg = op._get_xyzfxreg(kwargs)
        return cd.op.Knm_inv(get_matrix(x),get_matrix(y),get_matrix(fx),get_matrix(reg))
    def nabla(**kwargs):
        """
        Compute the kernel-induced gradient of a function.

        This function calculates the gradient of a function defined by a positive definite (PD) kernel. 
        The gradient is computed with respect to the input variables using kernel methods.

        Args:
            kwargs: A dictionary of keyword arguments. 

        Returns:
            The gradient of function as a matrix.
        """
        kernel.init(**kwargs)
        x,y,z,fx,reg = op._get_xyzfxreg(kwargs)
        return cd.op.nabla(get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx),get_matrix(reg))
    def nabla_inv(**kwargs):
        kernel.init(**kwargs)
        fz = kwargs.get('fz',[])
        x,y,z, _, _ = op._get_xyzfxreg(kwargs)
        return cd.op.nabla_inv(get_matrix(x),get_matrix(y),get_matrix(z),fz)
    def nablaT(**kwargs):
        kernel.init(**kwargs)
        x,y,z, _, _ = op._get_xyzfxreg(kwargs)
        fz = kwargs.get('fz',[])
        return cd.op.nablaT(get_matrix(x),get_matrix(y),get_matrix(z),fz)
    def nablaT_inv(**kwargs):
        kernel.init(**kwargs)
        x, y, z, fx, _ = op._get_xyzfxreg(kwargs)
        return cd.op.nablaT_inv(get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx))
    def nablaT_nabla(**kwargs):
        kernel.init(**kwargs)
        x, y, _ ,fx, _ = op._get_xyzfxreg(kwargs)
        return cd.op.nablaT_nabla(x=get_matrix(x),y=get_matrix(y),fx=get_matrix(fx))
    def nablaT_nabla_inv(**kwargs):
        kernel.init(**kwargs)
        x, y, _, fx, _ = op._get_xyzfxreg(kwargs)
        return cd.op.nablaT_nabla_inv(get_matrix(x),get_matrix(y),get_matrix(fx))
    def Leray_T(**kwargs):
        kernel.init(**kwargs)
        x,y, _,fx, _ = op._get_xyzfxreg(kwargs)
        return cd.op.Leray_T(get_matrix(x),get_matrix(y),fx)
    def Leray(**kwargs):
        """
        Compute the Leray operator for a given set of input matrices.

        The Leray operator is a mathematical construct often used in various fields such as fluid dynamics 
        and differential equations. This function computes the Leray operator using the provided input 
        matrices, which are processed through a kernel method.

        Args:
            kwargs: A dictionary of keyword arguments.

        Returns:
            The result of the Leray operator computation as a matrix or an array-like structure, 
            depending on the input matrices.
        """
        kernel.init(**kwargs)
        x,y, _,fx, _ = op._get_xyzfxreg(kwargs)
        return cd.op.Leray(get_matrix(x),get_matrix(y),fx)
    def hessian(**kwargs):
        """
        Compute the kernel-induced Hessian matrix of a function.

        This function calculates the Hessian matrix, which represents the second-order partial 
        derivatives of a function defined by a positive definite (PD) kernel. 
        Args:
            kwargs: A dictionary of keyword arguments.

        The function computes the Hessian matrix for each input data point. If the 'fx' parameter
        is provided, the function computes a modified Hessian matrix using this additional information.

        Returns:
            The Hessian matrix of the function, which can vary in dimensions based on 
            the input matrices and the presence of the 'fx' parameter.
        """
        import itertools
        x, _, z, fx, _ = op._get_xyzfxreg(kwargs)
        indices = distance_labelling(**{**kwargs,**{'x':z,'y':x}})
        grad = op.nabla(**{**kwargs,**{'fx':[]}})
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
    def rescale(x=[],y=[],z=[],**kwargs):
        x,y,z = column_selector([x,y,z],**kwargs)
        def get_param(**kwargs): return kwargs.get("rescale_kernel",None)
        param = get_param(**kwargs)
        if param is not None:
            max_ = param.get("max",None)
            if max_ is not None:
                seed_ = param.get("seed",None)
                x,y,z = random_select(x=x,xmax = max_,seed=seed_),random_select(x=y,xmax = max_,seed=seed_),random_select(x=z,xmax = max_,seed=seed_)
        x,y,z = get_matrix(x),get_matrix(y),get_matrix(z)
        cd.kernel.rescale(x,y,z)
    def get_kernel_ptr():
        return cd.get_kernel_ptr()
    def set_kernel_ptr(kernel_ptr):
        cd.set_kernel_ptr(kernel_ptr)
    def pipe_kernel_ptr(kernel_ptr):
        cd.kernel.pipe_kernel_ptr(kernel_ptr)
    def pipe_kernel_fun(kernel_fun, regularization = 1e-8):
        kern1 = kernel.get_kernel_ptr()
        kernel_fun()
        kern2 = kernel.get_kernel_ptr()
        kernel.set_kernel_ptr(kern1)
        kernel.pipe_kernel_ptr(kern2)
        cd.kernel.set_regularization(regularization)
    def init(**kwargs):
        set_codpy_kernel = kwargs.get('set_codpy_kernel',None)
        if set_codpy_kernel is not None: set_codpy_kernel()
        rescale = kwargs.get('rescale',False)
        if (rescale): kernel.rescale(**kwargs)


class map_setters:
    class set:    
        def __init__(self,strings):
            self.strings = strings
        def __call__(self,**kwargs):    
            if isinstance(self.strings,list):
                ss = self.strings.copy()
                cd.kernel.set_map(ss.pop(0),kwargs)
                [_pipe_map_setters.pipe(s) for s in ss]
            else: cd.kernel.set_map(self.strings,kwargs)
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
        _pipe_map_setters.pipe_erfinv_map()
        _pipe_map_setters.pipe_unitcube_map()
    def set_standard_min_map(**kwargs):
        """
        Set a standard minimum map pipeline for the kernel.

        This function sets a minimum distance map and then pipes it through the erf-inverse and unit cube maps.

        Args:
        - **kwargs: Arbitrary keyword arguments for the standard minimum map configuration.
        """
        map_setters.set_min_distance_map(**kwargs)
        _pipe_map_setters.pipe_erfinv_map()
        _pipe_map_setters.pipe_unitcube_map()
    def set_unitcube_min_map(**kwargs):
        """
        Set a unit cube minimum map pipeline for the kernel.

        This function sets a minimum distance map and then pipes it through the unit cube map.

        Args:
        - **kwargs: Arbitrary keyword arguments for the unit cube minimum map configuration.
        """
        map_setters.set_min_distance_map(**kwargs)
        _pipe_map_setters.pipe_unitcube_map()
    def set_unitcube_erfinv_map(**kwargs):
        """
        Set a unit cube erf-inverse map pipeline for the kernel.

        This function sets an erf-inverse map and then pipes it through the unit cube map.

        Args:
        - **kwargs: Arbitrary keyword arguments for the unit cube erf-inverse map configuration.
        """
        map_setters.set_erfinv_map()
        _pipe_map_setters.pipe_unitcube_map()
    def set_unitcube_mean_map(**kwargs):
        """
        Set a unit cube mean map pipeline for the kernel.

        This function sets a mean distance map and then pipes it through the unit cube map.

        Args:
        - **kwargs: Arbitrary keyword arguments for the unit cube mean map configuration.
        """
        map_setters.set_mean_distance_map(**kwargs)
        _pipe_map_setters.pipe_unitcube_map()
    def map_helper(map_setter, **kwargs): 
        """
        Helper function to create a custom map setting function.

        This function creates a partial function for a specified map setter with provided arguments.

        Args:
        - map_setter (function): The map setter function to be used.
        - **kwargs: Arbitrary keyword arguments for the map setter function.
        """
        return partial(map_setter, kwargs)


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
            linear_kernel = kernel_setters.kernel_helper(setter = kernel_setters.set_linear_regressor_kernel,polynomial_order = polynomial_order,regularization = regularization,set_map = None)
            kernel.pipe_kernel_fun(linear_kernel,regularization)
        cd.kernel.set_regularization(regularization)

    def set_linear_regressor_kernel(polynomial_order:int = 2,regularization:float = 1e-8,set_map = None):
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

    def set_absnorm_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = None): 
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
    default_multiquadricnorm_kernel_map = partial(map_setters.set_standard_mean_map, distance ='norm2')
    def set_multiquadricnorm_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = default_multiquadricnorm_kernel_map): 
        """
        Set the multi-quadric norm kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function defined as `default_multiquadricnorm_kernel_map`.
        """
        kernel_setters.set_kernel("multiquadricnorm",polynomial_order,regularization,set_map)
    default_multiquadrictensor_kernel_map = partial(map_setters.set_standard_min_map, distance ='normifty')
    def set_multiquadrictensor_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = default_multiquadrictensor_kernel_map): 
        """
        Set the multi-quadric tensor kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function defined as `default_multiquadrictensor_kernel_map`.
        """
        kernel_setters.set_kernel("multiquadrictensor",polynomial_order,regularization,set_map)
    default_sincardtensor_kernel_map = partial(map_setters.set_min_distance_map, distance ='normifty')
    def set_sincardtensor_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = default_sincardtensor_kernel_map):
        """
        Set the sinc cardinal tensor kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function defined as `default_sincardtensor_kernel_map`.
        """
        kernel_setters.set_kernel("sincardtensor",polynomial_order,regularization,set_map)        
    default_sincardsquaretensor_kernel_map = partial(map_setters.set_min_distance_map, distance ='normifty')
    def set_sincardsquaretensor_kernel(polynomial_order:int = 0,regularization:float = 0,set_map = default_sincardsquaretensor_kernel_map): 
        """
        Set the sinc cardinal square tensor kernel with specified parameters.

        Args:
        - polynomial_order (int): The polynomial order for the kernel function.
        - regularization (float): The regularization parameter for the kernel.
        - set_map (callable, optional): An optional mapping function defined as `default_sincardsquaretensor_kernel_map`.
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

class _pipe_map_setters:
    def pipe(s,**kwargs):
        cd.kernel.pipe_map(s,kwargs)
    def pipe_log_map(**kwargs):_pipe_map_setters.pipe("log",**kwargs)
    def pipe_exp_map(**kwargs):_pipe_map_setters.pipe("exp",**kwargs)
    def pipe_linear_map(**kwargs):_pipe_map_setters.pipe("linear_map",**kwargs)
    def pipe_affine_map(**kwargs):_pipe_map_setters.pipe_affine_map("affine_map",**kwargs)
    def pipe_scale_std_map(**kwargs):_pipe_map_setters.pipe("scale_std",**kwargs)
    def pipe_erf_map(**kwargs):_pipe_map_setters.pipe("scale_to_erf",**kwargs)
    def pipe_erfinv_map(**kwargs):_pipe_map_setters.pipe("scale_to_erfinv",**kwargs)
    def pipe_unitcube_map(**kwargs):_pipe_map_setters.pipe("scale_to_unitcube",**kwargs)
    def pipe_mean_distance_map(**kwargs):_pipe_map_setters.pipe("scale_to_mean_distance",**kwargs)
    def pipe_min_distance_map(**kwargs):_pipe_map_setters.pipe("scale_to_min_distance",**kwargs)

