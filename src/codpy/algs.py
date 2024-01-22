import numpy as np
import codpydll
import codpypyd as cd
from codpy.core import op, _kernel, _kernel_helper2, _requires_rescale
from utils.data_processing import lexicographical_permutation
import warnings


class alg:
    def iso_probas_projection(x, fx, probas, fun_permutation = lexicographical_permutation, kernel_fun = None, map = None,
                              polynomial_order = 2, regularization = 1e-8, rescale = False, rescale_params: dict = {'max': 1000, 'seed':42}, 
                              verbose = False, **kwargs):
        # print('######','iso_probas_projection','######')
        params = {'set_codpy_kernel' : _kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization)}
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescale_kernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            _kernel.init(x,x,x, **params)
        else:
            params['rescale'] = rescale
            _kernel.init(**params)
        Nx,Dx = np.shape(x)
        Nx,Df = np.shape(fx)
        Ny = len(probas)
        fy,y,permutation = fun_permutation(fx,x)
        out = np.concatenate((y,fy), axis = 1)
        quantile = np.array(np.arange(start = .5/Nx,stop = 1.,step = 1./Nx)).reshape(Nx,1)
        out = op.projection(x = quantile,y = probas,z = probas, fx = out, kernel_fun=kernel_fun, rescale = rescale)
        return out[:,0:Dx],out[:,Dx:],permutation

    def Pi(x, z, fz=[], nmax=10, rescale = False, **kwargs):
        # print('######','Pi','######')
        _kernel.init(**kwargs)
        if (rescale): _kernel.rescale(x,z)
        out = cd.alg.Pi(x = x,y = x,z = z, fz = fz,nmax = nmax)
        return out