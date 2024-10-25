from typing import List,Set,Dict,get_type_hints
import numpy as np
import pandas as pd
#from utils import *




###
#################### casting tools ########################################
###

# cast_Dict = {list:{
#     dict:list_to_dic,
#     Dict[int, Set[int]]:list_to_dic_int_set_int},
# int:{set:int_to_set},
# dict:{list:dict_to_list}
# }
cast_dict = {}

def declare_cast(source,target,fun,dic = cast_dict):
    # print('###########')
    if source in dic.keys():
        dic[source].update({target:fun})
    else:
        dic.update({source:{target:fun}})
    # for k in cast_dict:print(k,cast_dict[k])    
    pass

def cast(data,type_out, type_in = None):
    if type_in is None: type_in = type(data)
    if type_in == type_out : return data
    if type_in not in cast_dict.keys(): return type_out(data)
    test = cast_dict[type_in]
    return test[type_out](data)

def ndarray_to_dic_int_set_int(data : np.ndarray) -> Dict[int, Set[int]]:
    out = {}
    def helper(k,v):out.update({int(k):cast(int(v),Set[int])})
    [helper(k,v) for k,v in zip(range(0,len(data)),data)]
    return out
declare_cast(np.ndarray,Dict[int, Set[int]],ndarray_to_dic_int_set_int)



def list_int_to_dic_int_set_int(data : List[int]) -> Dict[int, Set[int]]:
    out = {}
    def helper(k,v):out.update({k:cast(v,Set[int])})
    [helper(k,v) for k,v in zip(range(0,len(data)),data)]
    return out
declare_cast(List[int],Dict[int, Set[int]],list_int_to_dic_int_set_int)

def int_to_set(data : int) -> set:return set([data])
declare_cast(int,set,int_to_set)
def int_to_Set_int(data : int) -> Set:return set([data])
declare_cast(int,Set[int],int_to_Set_int)

def dic_int_set_int_to_list_int(data : dict) -> List[int]:
    keys = list(data.keys())
    keys.sort()
    out = []
    for i in keys : out += list(data[i])
    return out
declare_cast(Dict[int, Set[int]],List[int],dic_int_set_int_to_list_int)

def dic_int_set_int_to_ndarray(data : dict) -> np.ndarray:
    keys = list(data.keys())
    keys.sort()
    out = []
    for i in keys : out += list(data[i])
    return np.asarray(out)
declare_cast(Dict[int, Set[int]],np.ndarray,dic_int_set_int_to_ndarray)

def get_closest_key(dic,mykey):
    return min(dic.keys(), key = lambda x: abs(x-mykey))
def get_last_key(dic):
    return list(dic.keys())[-1]
def get_closest_value(dic,mykey):
    return dic[get_closest_key(dic,mykey)]

def lazy_dic_evaluation(dic,key,fun):
    out = dic.get(key,None)
    if out is None: return fun()
    return out

def get_dictionnary(x, y, fun = max):
    dic = {}
    switchDict = {np.ndarray: lambda x: x, pd.DataFrame: lambda x : x.data, pd.Series: lambda x : x.values}
    x_data = get_data(x)
    y_data = get_data(y)
    size_ = len(x_data)
    if size_ != len(y_data): raise AssertionError("map_array_inversion error, different length. len(x)="+len(x_data)," len(y)="+len(y_data))
    for i in np.unique(x_data):
        # find index of points in cluster
        index = np.where(x_data == i)
        value = list(y_data[index])
        if fun is not None : dic[i] =fun(value, key = value.count)
        else: dic[i] = value

    return dic

def split_by_values(values, x):
    dic = get_dictionnary(values, x, fun = None)
    vals = [np.repeat(keys,len(dic[keys])) for keys in dic.keys()]
    if  isinstance(values,pd.DataFrame): vals = [pd.DataFrame(vals,columns = values.columns)]
    if  isinstance(x,pd.DataFrame): split = [pd.DataFrame(dic[keys],columns = x.columns) for keys in dic.keys()]
    else: split = dic.values()
    return vals, split


def get_surjective_dictionnary(x, y):
    # return {X: Y for X, Y in zip(x, y)}
    dic = get_dictionnary(x, y)
    out = {}
    def helper(key,value): out[key] = value
    [helper(key,value) for key,value in dic.items()]
    return out

def remap(x, dic, missing_key_value = 0):
    switch_get_item = {np.ndarray: lambda item,dic: dic.get(item,missing_key_value)}
    get_item = switch_get_item[type(x)]
    return np.asarray([get_item(item,dic) for item in x])