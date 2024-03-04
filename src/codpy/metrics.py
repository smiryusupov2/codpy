import numpy as np
from data_conversion import get_data

def get_classification_error(test_values,extrapolated_values):
    if test_values.size == 0: return 0.
    out = [test_values[n] == extrapolated_values[n] for n in range(len(test_values))]
    out = np.sum(out)
    out /= test_values.size
    #print("\n ************  classification error ********** : %s" %   (out))
    return out
    
def get_relative_mean_squared_error(a,b):
    a = get_data(a).flatten()
    b = get_data(b).flatten()
    out = np.linalg.norm(a-b)
    l = np.linalg.norm(a)
    r = np.linalg.norm(b)
    debug = l + r + 0.00001
    out /= debug
    return out 
def get_mean_squared_error(a,b):
    a = get_data(a).flatten()
    b = get_data(b).flatten()
    return np.linalg.norm(a-b)


def get_relative_error(x,z, ord = None):
    x,z = get_data(x),get_data(z)
    if (x.ndim == 1): x,z = x.reshape(len(x),1),z.reshape(len(x),1)
    debug = x-z
    debug = np.linalg.norm(debug.flatten(),ord)
    n = debug 
    debug = np.linalg.norm(x.flatten(),ord) + la.norm(z.flatten(),ord) + 1e-8
    n /= debug 
    # n *= n
    # print("\n ************  L2 error ********** : %s" %   (n))
    return n

def get_L2_error(x,z):
    return get_relative_error(x,z)