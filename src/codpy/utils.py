from typing import List, Set,Dict, get_type_hints
import numpy as np
import functools
import operator
from data_conversion import get_matrix


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


def null_rows(data_frame):
    return data_frame[data_frame.isnull().any(axis = 1)]

def flatten(list):
    return functools.reduce(operator.iconcat, list, [])


def softmaxindice(mat : np.ndarray, label: np.ndarray = [], axis: int = 1, diagonal: bool = False, **kwargs):
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
        >>> max_indices = softmaxindice(mat)
        
        max_indices will be [2, 2, 2] for axis=1
        
        >>> labels = np.array(['a', 'b', 'c'])
        >>> max_labels = softmaxindice(mat, label=labels)
        
        max_labels will be ['c', 'c', 'c'] for axis=1

    Note:
    This function is useful in scenarios where you need to find the dominant indices or labels from a set of softmax
    probabilities.
    """
    #print(mat[0:5])
    if diagonal:
        def helper(n):mat[n,n] = -1e+150
        [helper(n) for n in range(mat.shape[0])]
    out = np.argmax(get_matrix(mat), axis)
    if isinstance(label,np.ndarray):
        test = list()
        def helper(i):test.append(label[out[i]])
        [helper(i) for i in range(0,len(out))]
        out = test
    return out

def softminindice(mat : np.ndarray, label: np.ndarray = [], axis: int = 1, diagonal: bool = False, **kwargs):
    return softmaxindice(-mat, label, axis, diagonal, **kwargs)

if __name__ == "__main__":
    # for k in cast_dict:print(k,cast_dict[k])
    # test = [1,2]
    # test=map_invertion(test)
    # print(test)
    pass