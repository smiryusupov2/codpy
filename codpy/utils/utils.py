from typing import List,Set,Dict,get_type_hints
import numpy as np
import functools
import operator
from codpy.utils.data_conversion import get_matrix, get_data, my_len


def pad_axis(x,y, axis = 1):
    if x.shape[axis] == y.shape[axis] : return x,y
    if x.shape[axis] < y.shape[axis] : 
        y,x = pad_axis(y,x)
        return x,y
    ref_shape = np.array(y.shape)
    ref_shape[axis] = x.shape[axis]
    out = np.zeros(ref_shape)
    slices = [slice(0, y.shape[dim]) for dim in range(y.ndim)]    
    out[slices] = y
    return x,out

def format_32(x,**kwargs):
    if x.ndim==3:
        shapes = x.shape
        x = np.swapaxes(x,1,2)
        return x.reshape((x.shape[0]*x.shape[1],x.shape[2]))
def format_23(x,shapes,**kwargs):
    if x.ndim==2:
        out = x.reshape((shapes[0],shapes[2],shapes[1]))
        out = np.swapaxes(out,1,2)
        return out

def lexicographical_permutation(x,fx=[],**kwargs):
    # x = get_data(x)
    if x.ndim==1: index_array = np.argsort(x)
    else: 
        indexfx = kwargs.get("indexfx",0)
        index_array = np.argsort(a = x[:,indexfx])
    x_sorted = x[index_array]
    if (my_len(fx) != my_len(x_sorted)): return (x_sorted,index_array)
    if type(fx) == type([]): out = [s[index_array] for s in fx]
    # else: out = np.take_along_axis(arr = x,indices = index_array,axis = indexx)
    else: out = fx[index_array]
    return (x_sorted,out,index_array)


def null_rows(data_frame):
    return data_frame[data_frame.isnull().any(axis = 1)]

def unity_partition(fx, unique_values = []):
    if len(unique_values) == 0: unique_values= np.unique(get_data(fx).flatten())
    P = len(unique_values)
    N = len(fx)
    out = np.zeros((N,P))
    d=0
    for v in unique_values :
        indices = np.where(fx == v)
        out[indices[0],d]=1.
        d=d+1
    #print(out)
    return out

def softmaxindice(mat, **kwargs):
    #print(mat[0:5])
    axis=kwargs.get('axis',1)
    label = kwargs.get('label',None)
    if kwargs.get('diagonal',False):
        def helper(n):mat[n,n] = -1e+150
        [helper(n) for n in range(mat.shape[0])]
    out = np.argmax(get_matrix(mat), axis)
    if isinstance(label,np.ndarray):
        test = list()
        def helper(i):test.append(label[out[i]])
        [helper(i) for i in range(0,len(out))]
        out = test
    return out
def softminindice(mat, **kwargs):
    return softmaxindice(-mat, **kwargs)

def hot_label(mat, label=None):
    #print(mat[0:5])
    return np.argmax(get_matrix(mat), axis)

def flatten(list):
    return functools.reduce(operator.iconcat, list, [])


if __name__ == "__main__":
    # for k in cast_dict:print(k,cast_dict[k])
    # test = [1,2]
    # test=map_invertion(test)
    # print(test)
    pass