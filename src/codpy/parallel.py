import multiprocessing as mp
import concurrent.futures
from tqdm import tqdm
import time
# from utils.scenarios import execute_function_list


## multiprocessing utilities# 
def parallel_task(param_list,fun,**kwargs):
    param_list = [{**p,**kwargs} for p in param_list]
    debug = kwargs.get('debug',True)
    if debug: return [fun(p) for p,i in zip(param_list,tqdm (range (len(param_list)), desc="parallel…", ascii=False, ncols=75))]
    if len(param_list) < 2: return [fun(p) for p in param_list]
    cores = min(mp.cpu_count(),len(param_list))

    out = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        results = executor.map(fun, param_list, chunksize= max(int( len(param_list)/cores),1) )
        for result,i in zip(results,tqdm (range (len(param_list)), desc="parallel…", ascii=False, ncols=75)):
            out.append(result)

    # pool = mp.Pool(cores)
    # out = pool.map(fun, param_list)
    # pool.close()
    # pool.join()
    return out
          

def elapsed_time(fun,msg="elapsed_time in seconds:",**kwargs):
    start = time.time()
    out = fun(**kwargs)
    print(msg,time.time()-start)
    return out

# def execution_time(**kwargs):
#     start_time = time.time()
#     out = execute_function_list(**kwargs)
#     msg = 'time in seconds:'
#     if 'msg' in kwargs:
#         msg = kwargs['msg']
#     print("*********execution time***********",msg,time.time() - start_time)
#     return out