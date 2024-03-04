import pandas as pd


def column_selector(data,**kwargs):
    """
    Select or drop columns from a pandas DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame to process.
        cols_drop (list, optional): List of columns to drop.
        cols_keep (list, optional): List of columns to keep.
        split (bool, optional): If True, returns both the modified and dropped columns as separate DataFrames.

    Returns:
        pd.DataFrame or (pd.DataFrame, pd.DataFrame): Modified DataFrame or a tuple of modified and dropped DataFrames.
    """
    if isinstance(data,list):return [column_selector(y,**kwargs) for y in data]
    if not isinstance(data, pd.DataFrame):
         raise ValueError("Input must be a pandas DataFrame")
    cols_drop = kwargs.get('cols_drop',[])
    cols_keep = kwargs.get('cols_keep',[])
    if len(cols_drop)+len(cols_keep) == 0: return data
    if len(cols_drop):test = cols_drop[0]
    else:test = cols_keep[0]
    if not isinstance(test,str) :
        test = set(data.columns) - set(cols_drop)
        if len(cols_keep): test = [c for c in test if c in cols_keep]
        return data[test]
    cols_drop = get_starting_cols(list(data.columns),cols_drop)
    x0 = data
    if len(cols_drop):
        x0 = x0.drop(cols_drop,axis=1)
    cols_keep = get_starting_cols(list(x0.columns),cols_keep)
    if len(cols_keep):
        x0 = x0[cols_keep]
    if kwargs.get("split",False):
        trash_cols = set(data.columns)-set(x0.columns)
        return x0,data[trash_cols]
    return x0

# def column_selector(column_selector_x,**kwargs):
#     if isinstance(column_selector_x,list):return [column_selector(y,**kwargs) for y in column_selector_x]
#     if not isinstance(column_selector_x,pd.DataFrame):return column_selector_x
#     cols_drop = kwargs.get('cols_drop',[])
#     cols_keep = kwargs.get('cols_keep',[])
#     if len(cols_drop)+len(cols_keep) == 0: return column_selector_x
#     if len(cols_drop):test = cols_drop[0]
#     else:test = cols_keep[0]
#     if not isinstance(test,str) :
#         test = set(column_selector_x.columns) - set(cols_drop)
#         if len(cols_keep): test = [c for c in test if c in cols_keep]
#         return column_selector_x[test]
#     cols_drop = get_starting_cols(list(column_selector_x.columns),cols_drop)
#     cols_keep = get_starting_cols(list(column_selector_x.columns),cols_keep)
#     x0 = column_selector_x
#     if len(cols_drop):
#         x0 = x0.drop(cols_drop,axis=1)
#     if len(cols_keep):
#         x0 = x0[cols_keep]
#     if kwargs.get("split",False):
#         trash_cols = set(column_selector_x.columns)-set(x0.columns)
#         return x0,column_selector_x[trash_cols]
#     return x0

# def raw_data_column_selector(xs,**kwargs):
#     if isinstance(xs,list): return [get_data_column_selector(x,**kwargs) for x in xs]
#     params = data_generator.get_params(**kwargs)
#     variables_cols_keep = params.get('variables_cols_keep',[])
#     variables_cols_drop = params.get('variables_cols_drop',[])
#     return column_selector(xs,cols_drop = variables_cols_drop, cols_keep = variables_cols_keep)

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
    return [col for col in a if isinstance(col,str) and any(col.startswith(m) for m in match)]

def select_constant_columns(df):
    out = list()
    def fun(out,df,col):
        unique_values = df[col].unique()
        if len(unique_values) == 1:out.append(col)
    [fun(out,df,col) for col in df.columns]
    return out
