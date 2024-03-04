import numpy as np
import pandas as pd
from data_conversion import my_len

def set_seed(seedlabel = 'seed',**kwargs):
    # print('######','sharp_discrepancy','######')
    if seedlabel in kwargs:
        seed = int(kwargs.get(seedlabel))
        np.random.seed(seed)


def random_select_ndarray(x,xmax,seed=0):
    if (len(x) > xmax):
        if seed: np.random.seed(seed)
        arr = np.arange(xmax)
        np.random.shuffle(arr)
        x = x[arr]
    return x

def random_select_dataframe(x,xmax,seed=0):
    if (len(x) > xmax):
        if seed: np.random.seed(seed)
        arr = np.arange(xmax)
        np.random.shuffle(arr)
        x = x.iloc[arr]
    return x


def random_select(x,xmax,seed=0):
    # print('######','sharp_discrepancy','######')
    if isinstance(x,list):return[random_select(y,xmax,seed) for y in x]
    if not my_len(x): return x
    test = type(x)
    switchDict = {np.ndarray: random_select_ndarray, 
                 pd.DataFrame: random_select_dataframe}
    if test in switchDict.keys(): return switchDict[type(x)](x,xmax,seed)
    else:
        raise TypeError("unknown type "+ str(test) + " in random_select")

def random_select_interface(x, xmaxlabel = None, seedlabel = 42):
    # print('######','sharp_discrepancy','######')
    if xmaxlabel is None or x is None: return x
    return random_select(x=x,xmax = xmaxlabel,seed=seedlabel)

