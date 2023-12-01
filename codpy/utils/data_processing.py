import functools
import operator
import pandas as pd
import numpy as np
import codpydll
import codpypyd as cd
from .metrics import *
from codpy.utils.utils import lexicographical_permutation
from codpy.utils.selection import column_selector
from codpy.src.core import op

def select_list_of_words(list,list_of_words):
    return [col for col in list if any(word in col for word in list_of_words) ]
def get_matching_cols(a,match):
    if  isinstance(match,list): 
        out= []
        def fun(out,m):out+=m
        [fun(out,get_matching_cols(a,m)) for m in match]
        return out
    return [col for col in a.columns if match in col]
def get_starting_cols(a,match):
    # for col in a: 
    #     if not isinstance(col,str): return[]   
    return [col for col in a if isinstance(col,str) and any(col.startswith(m) for m in match)]

def flatten(list):
    return functools.reduce(operator.iconcat, list, [])

def select_constant_columns(df):
    out = list()
    def fun(out,df,col):
        unique_values = df[col].unique()
        if len(unique_values) == 1:out.append(col)
    [fun(out,df,col) for col in df.columns]
    return out

def variable_selector(**params):
    kwargs = params.copy()
    x,y,z,fx,fz = kwargs['x'],kwargs['y'],kwargs['z'],kwargs['fx'],kwargs['fz']
    error_fun,predictor = kwargs.get('error_fun',get_mean_squared_error),kwargs.get('predictor',op.projection)
    variable_selector_csv = kwargs.get('variables_selector_csv',[])
    matching_cols = kwargs.get('selector_cols',[])

    cols_drop = params.get('cols_drop',[])
    cols_drop = set(cols_drop) | set(select_constant_columns(x))
    params['cols_drop'] = list(cols_drop)

    x,y,z = column_selector([x,y,z],**params)
    
    xyzcolumns = list(x.columns)
    keep_columns = kwargs.get("keep_columns",[])
    keep_columns = get_matching_cols(x,keep_columns)
    matching_cols = get_matching_cols(x,matching_cols)

    def helper(x,y,z,fx,fz,cols):
        kwargs['cols_keep'] = cols
        try:
            f_z = predictor(**kwargs)
            erreur = error_fun(fz, f_z)
        except Exception as e:
            print(e)
            erreur = float('Inf')
        print("columns:",cols, " -- erreur:",erreur)
        return erreur

    cols = list(set(x.columns).difference(set(keep_columns+matching_cols)))
    def fun(col):
        # values = set(x[col])
        erreur = helper(x,y,z,fx,fz,col)
        # if len(values) > 1:
        #     return helper(x,y,z,fx,fz,col)
        # else:
        #     erreur = float('Inf')
        #     print("columns:",col, " -- erreur:",erreur)
        return erreur
    erreurs = np.asarray([fun(keep_columns + [col]) for col in cols])
    erreurs, order = lexicographical_permutation(erreurs)
    best_erreur,best_erreurs = erreurs[0],[erreurs[0]]*(len(keep_columns)+1)
    best_col, best_cols = cols[order[0]],keep_columns + [cols[order[0]]]
    order = list(order)+list(range(len(cols),len(cols)+len(matching_cols)))
    cols += matching_cols
    for n in range(0,len(cols)):
        col = cols[order[n]]
        if col not in best_cols :
            best_cols.append(col)
            erreur = helper(x,y,z,fx,fz,best_cols) 
            if erreur >= best_erreur: best_cols.remove(col)
            else : 
                best_erreur = erreur
                best_erreurs.append(erreur)
    output = { 'keep_columns' : best_cols, 'errors' : best_erreurs}
    
    if len(variable_selector_csv): 
        # csv_file = output.copy()
        # csv_file['keep_columns'].insert(0,"NoNe") 
        # csv_file['errors'].insert(0,erreurs) 
        pd.DataFrame(output).to_csv(variable_selector_csv,sep = ',')
    return output['keep_columns']


def hot_encoder(data_frame,cat_cols_include = []):
    # data_frame.to_csv (r'data_frame.csv', header=True)
    num_dataframe = data_frame.select_dtypes(include='number')
    num_cols = set(num_dataframe.columns)
    if len(cat_cols_include):num_cols.difference_update(cat_cols_include)
    cat_cols = set(data_frame.columns)
    cat_cols = cat_cols.difference(num_cols)
    cat_dataframe = data_frame[cat_cols]
    num_dataframe = data_frame[num_cols]
    index = cat_dataframe.index
    values =cat_dataframe.to_numpy(dtype = str)
    cols = np.array(cat_dataframe.columns,dtype=str)
    (cat_dataframe,cat_columns) = cd.tools.hot_encoder(values,cols)
    cat_dataframe = pd.DataFrame(cat_dataframe, columns = cat_columns, index = index)
    if len(num_cols) :
        if not cat_dataframe.empty :
            cat_dataframe = pd.concat([num_dataframe,cat_dataframe], axis=1,join="inner")
        else :
            cat_dataframe = num_dataframe
    # cat_dataframe.to_csv (r'hot_encoder.csv', header=True)
    return cat_dataframe