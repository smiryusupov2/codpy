from preamble import *
from codpy_predictors import *

##################################### Kernels
# set_gaussian_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 0,1e-8,map_setters.set_mean_distance_map)
# set_tensornorm_kernel = kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 0,0,map_setters.set_unitcube_map)
# set_per_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussianper_kernel,2,1e-8,None)

codpy_params = {'rescale:xmax': 1000,
'rescale:seed':42,
'sharp_discrepancy:xmax':1000,
'sharp_discrepancy:seed':30,
'sharp_discrepancy:itermax':5,
'discrepancy:xmax':500,
'discrepancy:ymax':500,
'discrepancy:zmax':500,
'discrepancy:nmax':2000}



if __name__ == "__main__":
   pass

