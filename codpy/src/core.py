import pandas as pd
import numpy as np
from functools import partial
import itertools
import codpydll
import codpypyd as cd
from codpy.utils.selection import *
from codpy.utils.random import *
from codpy.utils.dictionary import *


def get_codpy_param():
    return  {'rescale_kernel':{'max': 1000, 'seed':42},
    'sharp_discrepancy':{'max': 1000, 'seed':42},
    'discrepancy':{'max': 1000, 'seed':42},
    'validator_compute': ['accuracy_score','discrepancy_error','norm'],
    'set_codpy_kernel' : kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 2,1e-8 ,map_setters.set_unitcube_map),
    'rescale': True,
    }

class codpy_param_getter:
    def get_params(**kwargs) : return kwargs.get('codpy',{})
    def get_kernel_fun(**kwargs): return codpy_param_getter.get_params(**kwargs)['set_kernel']

class factories:
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
                [pipe_map_setters.pipe(s) for s in ss]
            else: cd.kernel.set_map(self.strings,kwargs)
    def set_linear_map(**kwargs): cd.kernel.set_map("linear_map",kwargs)
    def set_affine_map(**kwargs): cd.kernel.set_map("affine_map",kwargs)
    def set_log_map(**kwargs): cd.kernel.set_map("log",kwargs)
    def set_exp_map(**kwargs): cd.kernel.set_map("exp",kwargs)
    def set_scale_std_map(**kwargs): cd.kernel.set_map("scale_std",kwargs)
    def set_erf_map(**kwargs): cd.kernel.set_map("scale_to_erf",kwargs)
    def set_erfinv_map(**kwargs): cd.kernel.set_map("scale_to_erfinv",kwargs)
    def set_unitcube_map(**kwargs): cd.kernel.set_map("scale_to_unitcube",kwargs)
    def set_grid_map(**kwargs): cd.kernel.set_map("map_to_grid",kwargs)
    def set_mean_distance_map(**kwargs): cd.kernel.set_map("scale_to_mean_distance",kwargs)
    def set_min_distance_map(**kwargs): cd.kernel.set_map("scale_to_min_distance",kwargs)
    def set_standard_mean_map(**kwargs):
        map_setters.set_mean_distance_map(**kwargs)
        pipe_map_setters.pipe_erfinv_map()
        pipe_map_setters.pipe_unitcube_map()
    def set_standard_min_map(**kwargs):
        map_setters.set_min_distance_map(**kwargs)
        pipe_map_setters.pipe_erfinv_map()
        pipe_map_setters.pipe_unitcube_map()
    def set_unitcube_min_map(**kwargs):
        map_setters.set_min_distance_map(**kwargs)
        pipe_map_setters.pipe_unitcube_map()
    def set_unitcube_erfinv_map(**kwargs):
        map_setters.set_erfinv_map()
        pipe_map_setters.pipe_unitcube_map()
    def set_unitcube_mean_map(**kwargs):
        map_setters.set_mean_distance_map(**kwargs)
        pipe_map_setters.pipe_unitcube_map()
    def map_helper(map_setter, **kwargs): return partial(map_setter, kwargs)


class kernel_setters:
    def kernel_helper(setter, polynomial_order:int = 0,regularization:float = 1e-8,set_map = None):
        return partial(setter, polynomial_order,regularization,set_map)
    def set_kernel(kernel_string,polynomial_order:int = 2,regularization:float = 1e-8,set_map = None):
        cd.kernel.set_polynomial_order(polynomial_order)
        cd.set_kernel(kernel_string)
        if (set_map) : set_map()
        if (polynomial_order > 0):
            linear_kernel = kernel_setters.kernel_helper(setter = kernel_setters.set_linear_regressor_kernel,polynomial_order = polynomial_order,regularization = regularization,set_map = None)
            kernel.pipe_kernel_fun(linear_kernel,regularization)
        cd.kernel.set_regularization(regularization)

    def set_linear_regressor_kernel(polynomial_order:int = 2,regularization:float = 1e-8,set_map = None):
        cd.kernel.set_polynomial_order(polynomial_order)
        cd.set_kernel("linear_regressor")
        cd.kernel.set_regularization(regularization)
        if (set_map) : set_map()

    def set_absnorm_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = None): kernel_setters.set_kernel("absnorm",polynomial_order,regularization,set_map)
    def set_tensornorm_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = None): kernel_setters.set_kernel("tensornorm",polynomial_order,regularization,set_map)
    def set_gaussian_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = None): kernel_setters.set_kernel("gaussian",polynomial_order,regularization,set_map)
    def set_matern_tensor_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = None): kernel_setters.set_kernel("materntensor",polynomial_order,regularization,set_map)
    default_multiquadricnorm_kernel_map = partial(map_setters.set_standard_mean_map, distance ='norm2')
    def set_multiquadricnorm_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = default_multiquadricnorm_kernel_map): kernel_setters.set_kernel("multiquadricnorm",polynomial_order,regularization,set_map)
    default_multiquadrictensor_kernel_map = partial(map_setters.set_standard_min_map, distance ='normifty')
    def set_multiquadrictensor_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = default_multiquadrictensor_kernel_map): kernel_setters.set_kernel("multiquadrictensor",polynomial_order,regularization,set_map)
    default_sincardtensor_kernel_map = partial(map_setters.set_min_distance_map, distance ='normifty')
    def set_sincardtensor_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = default_sincardtensor_kernel_map):
        kernel_setters.set_kernel("sincardtensor",polynomial_order,regularization,set_map)        
    default_sincardsquaretensor_kernel_map = partial(map_setters.set_min_distance_map, distance ='normifty')
    def set_sincardsquaretensor_kernel(polynomial_order:int = 0,regularization:float = 0,set_map = default_sincardsquaretensor_kernel_map): kernel_setters.set_kernel("sincardsquaretensor",polynomial_order,regularization,set_map)        
    def set_dotproduct_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = None):kernel_setters.set_kernel("DotProduct",polynomial_order,regularization,set_map)        
    def set_gaussianper_kernel(polynomial_order:int = 0,regularization:float = 1e-8,set_map = None):kernel_setters.set_kernel("gaussianper",polynomial_order,regularization,set_map)        
    def set_matern_norm_kernel(polynomial_order:int = 2,regularization:float = 1e-8,set_map = map_setters.set_mean_distance_map): kernel_setters.set_kernel("maternnorm",polynomial_order,regularization,set_map) 


class pipe_map_setters:
    def pipe(s,**kwargs):
        cd.kernel.pipe_map(s,kwargs)
    def pipe_log_map(**kwargs):pipe_map_setters.pipe("log",**kwargs)
    def pipe_exp_map(**kwargs):pipe_map_setters.pipe("exp",**kwargs)
    def pipe_linear_map(**kwargs):pipe_map_setters.pipe("linear_map",**kwargs)
    def pipe_affine_map(**kwargs):pipe_map_setters.pipe_affine_map("affine_map",**kwargs)
    def pipe_scale_std_map(**kwargs):pipe_map_setters.pipe("scale_std",**kwargs)
    def pipe_erf_map(**kwargs):pipe_map_setters.pipe("scale_to_erf",**kwargs)
    def pipe_erfinv_map(**kwargs):pipe_map_setters.pipe("scale_to_erfinv",**kwargs)
    def pipe_unitcube_map(**kwargs):pipe_map_setters.pipe("scale_to_unitcube",**kwargs)
    def pipe_mean_distance_map(**kwargs):pipe_map_setters.pipe("scale_to_mean_distance",**kwargs)
    def pipe_min_distance_map(**kwargs):pipe_map_setters.pipe("scale_to_min_distance",**kwargs)

class op:
    #set_codpy_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 2,1e-8,map_setters.set_min_distance_map)
    set_codpy_kernel = kernel_setters.kernel_helper(kernel_setters.set_matern_tensor_kernel, 2,1e-8,map_setters.set_standard_mean_map)
    def sum_axis(x,axis=-1):
        if axis != -1: 
            out = np.sum(x,axis)
        else: out=x
        return out

    def discrepancy(x,z,y=[],disc_type="raw", **kwargs):
        if 'discrepancy:xmax' in kwargs: x= random_select_interface(x,xmaxlabel = 'discrepancy:xmax', seedlabel = 'discrepancy:seed',**kwargs)
        if 'discrepancy:ymax' in kwargs: y= random_select_interface(y,xmaxlabel = 'discrepancy:ymax', seedlabel = 'discrepancy:seed',**kwargs)
        if 'discrepancy:zmax' in kwargs: z= random_select_interface(z,xmaxlabel = 'discrepancy:zmax', seedlabel = 'discrepancy:seed',**kwargs)
        if 'discrepancy:nmax' in kwargs: 
            nmax = int(kwargs.get('discrepancy:nmax'))
            if len(x) + 2 * len(y) + len(z) > nmax: return np.NaN
        kernel.init(x=x,z=z,y=y,**kwargs)
        if (len(y)): 
            debug = cd.tools.discrepancy_error(x,y,disc_type)
            debug += cd.tools.discrepancy_error(y,z,disc_type)
            return np.sqrt(debug)
        else: return np.sqrt(cd.tools.discrepancy_error(x,z,disc_type))

    class discrepancy_functional:
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

    def projection(**kwargs):
        projection_format_switchDict = { pd.DataFrame: lambda **kwargs :  projection_dataframe(**kwargs) }
        z = kwargs['z']
        if isinstance(z,list): 
            def fun(zi):
                kwargs['z'] = zi
                return op.projection(**kwargs)
            out = [fun(zi) for zi in z]
            return out
        def projection_dataframe(**kwargs):
            x,y,z,fx,reg = kwargs.get('x',[]),kwargs.get('y',[]),kwargs.get('z',[]),kwargs.get('fx',[]),kwargs.get('reg',[])
            x,y,z = column_selector([x,y,z],**kwargs)
            f_z = cd.op.projection(get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx),get_matrix(reg))
            if isinstance(fx,pd.DataFrame): f_z = pd.DataFrame(f_z,columns = list(fx.columns), index = z.index)
            return f_z

        kernel.init(**kwargs)
        type_debug = type(kwargs.get('x',[]))

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
    def weighted_projection(**kwargs):
        projection_format_switchDict = { pd.DataFrame: lambda **kwargs :  projection_dataframe(**kwargs) }
        z = kwargs['z']
        if isinstance(z,list): 
            def fun(zi):
                kwargs['z'] = zi
                return op.weighted_projection(**kwargs)
            out = [fun(zi) for zi in z]
            return out
        def projection_dataframe(**kwargs):
            x,y,z,fx,weights = kwargs.get('x',[]),kwargs.get('y',[]),kwargs.get('z',[]),kwargs.get('fx',[]),np.array(kwargs.get('weights',[]))
            x,y,z = column_selector([x,y,z],**kwargs)
            f_z = cd.op.weighted_projection(get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx),weights)
            if isinstance(fx,pd.DataFrame): f_z = pd.DataFrame(f_z,columns = list(fx.columns), index = z.index)
            return f_z

        kernel.init(**kwargs)
        type_debug = type(kwargs.get('x',[]))

        def debug_fun(**kwargs):
            x,y,z,fx,weights = kwargs.get('x',[]),kwargs.get('y',[]),kwargs.get('z',[]),kwargs.get('fx',[]),np.array(kwargs.get('weights',[]))
            f_z = cd.op.weighted_projection(get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx),weights)
            return f_z

        method = projection_format_switchDict.get(type_debug,debug_fun)
        f_z = method(**kwargs)

        # if kwargs.get('save_cd_data',False):
        #     class Object:pass
        #     temp = Object
        #     temp.x ,temp.y,temp.z,temp.fx,temp.f_z,temp.fz = x ,y,z,fx,f_z,kwargs.get("fz",[])
        #     data_generator.save_cd_data(temp,**kwargs)
        # return f_z

    def extrapolation(x,fx,z,reg=[], **kwargs):
        return op.projection(x=x,y=x,z=z,fx=fx,reg=reg,**kwargs)
    def interpolation(x,z,fx,reg=[], **kwargs):
        return op.projection(x=x,y=z,z=z,fx=fx,reg=reg,**kwargs)

    def norm(x,y,z,fx, set_codpy_kernel = None, rescale = False,**kwargs):
        if (set_codpy_kernel): set_codpy_kernel()
        if (rescale): kernel.rescale(x,y,z,**kwargs)
        return cd.tools.norm_projection(get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx))

    def coefficients(x,y,fx, **kwargs):
        kernel.init(x=x,y=y,**kwargs)
        return cd.op.coefficients(get_matrix(x),get_matrix(y),get_matrix(fx))
    def Knm(x,y,**kwargs):
        kernel.init(x=x,y=y,**kwargs)
        return cd.op.Knm(get_matrix(x),get_matrix(y))
    def Dnm(x,y, **kwargs):
        x,y = column_selector(x,**kwargs),column_selector(y,**kwargs)
        x,y = pad_axis(x,y)
        kernel.init(x=x,y=y,**kwargs)
        distance = kwargs.get('distance',None)
        if distance is not None:
            return cd.op.Dnm(get_matrix(x),get_matrix(y),{'distance':distance})
        return cd.op.Dnm(get_matrix(x),get_matrix(y))

    def nabla_Knm(x,y,**kwargs):
        kernel.init(x=x,y=y,**kwargs)
        return cd.op.nabla_Knm(get_matrix(x),get_matrix(y))
    def Knm_inv(x,y,reg=[],**kwargs):
        kernel.init(x=x,y=y,**kwargs)
        return cd.op.Knm_inv(get_matrix(x),get_matrix(y),get_matrix(reg))
    def nabla(x,y,z,fx = [],reg = [],**kwargs):
        kernel.init(x=x,y=y,z=z,**kwargs)
        return cd.op.nabla(get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx),get_matrix(reg))
    def nabla_inv(x,y,z,fz = [],**kwargs):
        kernel.init(x=x,y=y,z=z,**kwargs)
        return cd.op.nabla_inv(get_matrix(x),get_matrix(y),get_matrix(z),fz)
    def nablaT(x,y,z,fz = [], **kwargs):
        kernel.init(x=x,y=y,z=z,**kwargs)
        return cd.op.nablaT(get_matrix(x),get_matrix(y),get_matrix(z),fz)
    def nablaT_inv(x,y,z,fx = [], **kwargs):
        kernel.init(x=x,y=y,z=z,**kwargs)
        return cd.op.nablaT_inv(get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx))
    def nablaT_nabla(x,y,fx=[], **kwargs):
        kernel.init(x=x,y=y,**kwargs)
        return cd.op.nablaT_nabla(get_matrix(x),get_matrix(y),get_matrix(fx))
    def nablaT_nabla_inv(x,y, fx= [], **kwargs):
        kernel.init(x=x,y=y,**kwargs)
        return cd.op.nablaT_nabla_inv(get_matrix(x),get_matrix(y),get_matrix(fx))
    def Leray_T(x,y,fx=[], **kwargs):
        kernel.init(x=x,y=y,**kwargs)
        return cd.op.Leray_T(get_matrix(x),get_matrix(y),fx)
    def Leray(x,y,fx, **kwargs):
        kernel.init(x=x,y=y,**kwargs)
        return cd.op.Leray(get_matrix(x),get_matrix(y),fx)
    def hessian(x,y,z,fx = None,**kwargs):
        # z = x
        indices = alg.distance_labelling(x=z,y=x,**kwargs)
        grad = op.nabla(x=x, y=y, z=x, **kwargs)
        N_X = x.shape[0]
        N_Z = z.shape[0]
        D = x.shape[1]
        gradT = np.zeros([N_X,D,N_X])
        def helper(d) :gradT[:,d,:] = grad[:,d,:].T.copy()
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
