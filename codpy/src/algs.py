import numpy as np
import codpydll
import codpypyd as cd
from codpy.src.core import op, kernel, kernel_setters, map_setters
from codpy.utils.data_conversion import get_matrix
from codpy.utils.data_processing import lexicographical_permutation

class Denoiser:
    """
    A class for performing least squares regression penalized by the norm of the gradient, 
    induced by a positive definite (PD) kernel.

    This class initializes with various parameters and sets up a regression framework 
    that includes regularization based on the gradient's magnitude. It is designed to 
    work with gradient norms induced by a PD kernel.

    Args:
        **kwargs: Arbitrary keyword arguments.
            x (array_like): The input data points.
            epsilon (float, optional): Regularization parameter for controlling the strength 
                of the gradient penalty. Default is 0.001.
            Other parameters relevant to the regression and kernel methods can be passed as well.

    Attributes:
        params (dict): Dictionary storing all the parameters passed during initialization and their values.
        epsilon (float): Regularization parameter.

    Methods:
        __call__(**kw): Performs the denoising operation on input data.
            Parameters:
                **kw: Arbitrary keyword arguments, with optional 'z' to specify new data points for denoising.
            Returns:
                The denoised output for the input data.

    Example:
        # Initialize the denoiser with input data 'x' and optional parameters
        denoiser = Denoiser(x=input_data, epsilon=0.01)
        
        # Perform denoising on the input data or new data points 'z'
        denoised_output = denoiser(z=new_data)
    """
    def __init__(self,**kwargs):
        self.params= kwargs.copy()
        self.params['x']=kwargs['x']
        self.params['y']=kwargs['x']
        self.epsilon = kwargs.get('epsilon',0.001)
        self.params['reg'] = self.epsilon*op.nablaT_nabla(**{**kwargs,**{'fx':[],'y':self.params['x'],'x':self.params['x']}})
        # self.kernel =  kernel.get_kernel_ptr()
        # self.fx = self.params['fx'].copy()
        # self.params['fx'] = lalg.prod(op.Knm_inv(**self.params),self.params['fx'])
        self.params['set_codpy_kernel'],self.params['rescale'] = None,False
        # plt.scatter(self.params['x'].squeeze(),self.fx.squeeze())
        # plt.plot(self.__call__(z=kwargs['x']))
        # self.params['fx'] = self.params['fx']
        pass
    def __call__(self, **kw):
        z = kw.get('z',self.params['x'])
        # self.reg = self.epsilon*op.nablaT_nabla(**{**kw,**{'fx':[],'y':self.y,'x':self.x}})
        self.kernel = kernel_setters.kernel_helper(kernel_setters.set_matern_norm_kernel, 0,1e-8 ,map_setters.set_standard_mean_map)
        out = op.extrapolation(**{**self.params,**{'z':z}})
        return out
    pass

class alg:
    def iso_probas_projection(x, fx, probas, fun_permutation = lexicographical_permutation, **kwargs):
        # print('######','iso_probas_projection','######')
        kernel.init(**kwargs)
        Nx,Dx = np.shape(x)
        Nx,Df = np.shape(fx)
        Ny = len(probas)
        fy,y,permutation = fun_permutation(fx,x)
        out = np.concatenate((y,fy), axis = 1)
        quantile = np.array(np.arange(start = .5/Nx,stop = 1.,step = 1./Nx)).reshape(Nx,1)
        out = op.projection(x = quantile,y = probas,z = probas, fx = out, set_codpy_kernel = set_codpy_kernel, rescale = rescale)
        return out[:,0:Dx],out[:,Dx:],permutation

    def Pi(x, z, fz=[], nmax=10,**kwargs):
        # print('######','Pi','######')
        kernel.init(**kwargs)
        if (rescale): kernel.rescale(x,z)
        out = cd.alg.Pi(x = x,y = x,z = z, fz = fz,nmax = nmax)
        return out