from typing import List,Set,Dict,get_type_hints
import numpy as np
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


if __name__ == "__main__":
    # for k in cast_dict:print(k,cast_dict[k])
    # test = [1,2]
    # test=map_invertion(test)
    # print(test)
    pass

