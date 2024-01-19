import pandas as pd
import numpy as np
import codpydll
import codpypyd as cd
from codpy.core import op
from codpy.utils.metrics import get_L2_error, get_classification_error
from codpy.utils.utils import softmaxindice
from codpy.utils.data_processing import hot_encoder
import errno
import os


def select_constant_columns(df):
    out = list()
    def fun(out,df,col):
        unique_values = df[col].unique()
        if len(unique_values) == 1:out.append(col)
    [fun(out,df,col) for col in df.columns]
    return out

def df_type(df:pd.DataFrame,**categ):
    df_columns  = df.columns
    set_categ   = set(categ.keys())
    set_column  = set(list(df_columns))
    test = list(set_categ - set_column)
    if (test):
        raise NameError(errno.ENOSTR,os.strerror(errno.ENOSTR),test)
    out = df.astype(categ)
    return out

def get_dataframe_error(test_values:pd.DataFrame,extrapolated_values:pd.DataFrame):
    num_cols = test_values._get_numeric_data().columns   
    print('get_dataframe_error.num_cols:',num_cols)
    cat_cols=list(set(test_values.columns) - set(num_cols))
    print('get_dataframe_error.cat_cols:',cat_cols)
    num_error = get_L2_error(test_values[num_cols].to_numpy(),extrapolated_values[num_cols].to_numpy())
    num_error += get_classification_error(test_values[cat_cols].to_numpy(),extrapolated_values[cat_cols].to_numpy())
    return num_error

def dataframe_discrepancy(df_x,df_z, kernel_fun = None, rescale=True, max = 2000):
    x,z = format_df_xz_to_np(df_x,df_z)
    return op.discrepancy_error(x = x,z = z, kernel_fun=kernel_fun, rescale=rescale, rescale_params = {'max':max})

def dataframe_norm_projection(df_x:pd.DataFrame,df_z:pd.DataFrame,df_fx:pd.DataFrame, kernel_fun = None, rescale=True, max = 2000):
    (x,z,fx,fx_columns) = format_df_to_np(df_x,df_z,df_fx)
    return op.norm_projection(x,z,fx, kernel_fun=kernel_fun, rescale=rescale, rescale_params = {'max':max})



def dataframe_extrapolation(df_x:pd.DataFrame,df_z:pd.DataFrame,df_fx:pd.DataFrame, kernel_fun = "gaussian", rescale=True, max = 2000):
    (x,z,fx,fx_columns) = format_df_to_np(df_x,df_z,df_fx)
    if len(x) < max:
        fz = op.extrapolation(x=x,fx=fx,z=z,kernel_fun=kernel_fun,rescale=rescale)
    else:
        index  = np.random.choice(x.shape[0], size=max,replace = False)
        y = x[index]
        fz = op.projection(x=x,fx=fx,z=z,kernel_fun=kernel_fun,rescale=rescale)


    df_fz_format = pd.DataFrame(data = fz, columns=fx_columns)
    num_cols = df_fx._get_numeric_data().columns                          
    cat_cols=list(set(df_fx.columns) - set(num_cols))
    out = pd.DataFrame(columns=df_fx.columns)
    out[num_cols] = df_fz_format[num_cols]

    df_fz_format_cols = df_fz_format.columns

    for cat in cat_cols:
        list_col = [s for s in df_fz_format_cols if cat in s]
        list_cat = [str.split(s,':cat:')[-1] for s in list_col] 
        mat = df_fz_format[list_col].to_numpy()
        test = softmaxindice(mat)
        values = [list_cat[s] for s in test]
        out[cat] = values
    return out

# def fun_helper_dataframe_extrapolation(**kwargs):
#     set_codpy_kernel= fun_helper_base(**kwargs)
#     df_x,df_fx, df_z, df_fz = kwargs['df_x'],kwargs['df_fx'],kwargs['df_z'],kwargs['df_fz']
#     out = dataframe_extrapolation(df_x,df_z,df_fx, set_codpy_kernel=set_codpy_kernel)
#     return get_dataframe_error(out,df_fz)

def df_intersect_concat(df_x:pd.DataFrame,df_z:pd.DataFrame):
    cols = list( set(df_x.columns) & set(df_z.columns))
    out = pd.concat([df_x[cols],df_z[cols]], ignore_index=True)
    return out

def format_df_xz_to_np(df_x:pd.DataFrame,df_z:pd.DataFrame):
    df_xz = df_intersect_concat(df_x,df_z)
    df_xz_format = hot_encoder(df_xz)
    df_xz_format = df_xz_format.dropna()
    np_xz = df_xz_format.to_numpy()
    x = np_xz[:len(df_x)]
    z = np_xz[len(df_x):]
    return x,z


def format_df_to_np(df_x:pd.DataFrame,df_z:pd.DataFrame,df_fx:pd.DataFrame):
    x,z = format_df_xz_to_np(df_x,df_z)
    df_fx_format = hot_encoder(df_fx)
    fx = df_fx_format.to_numpy()
    columns = df_fx_format.columns
    return x,z,fx,columns


###################### wrappers for dataframes


def fun_extrapolation(x,fx,y,fy,z,fz, kernel_fun = None, rescale = False):
    debug = op.projection(x = x,y = x,z = z, fx = fx,kernel_fun=kernel_fun,rescale = rescale)
    return get_classification_error(softmaxindice(fz),softmaxindice(debug))
def fun_discrepancy(x,fx,y,fy,z,fz, kernel_fun=None, rescale = False):
    return 1.-op.discrepancy(x=x,z=z,kernel_fun=kernel_fun,rescale = rescale)
def fun_norm(x,fx,y,fy,z,fz, kernel_fun = None, rescale = True):
    return op.norm(x=x,y=x,z=x,fx=fx,kernel_fun = kernel_fun,rescale = rescale)
def fun_projection(x,fx, y,fy,z,fz, kernel_fun = None, rescale = True):
    debug = op.projection(x = x,y = y,z = z, fx = fx,kernel_fun=kernel_fun,rescale = rescale)
    return get_classification_error(softmaxindice(fz),softmaxindice(debug))

# def fun_helper_base(**kwargs):
#     set_codpy_kernel = kernel_setters.set_gaussian_kernel
#     if 'set_codpy_kernel' in kwargs:
#             set_codpy_kernel = kwargs['set_codpy_kernel']
#     return set_codpy_kernel


# def fun_helper_projection(**kwargs):
#     set_codpy_kernel_ = fun_helper_base(**kwargs)
#     x_,fx_, y_, fy_, z_, fz_ = kwargs['x'],kwargs['fx'],kwargs['y'],kwargs['fy'],kwargs['z'],kwargs['fz']
#     out = fun_projection(x_,fx_,y_,fy_,z_,fz_, set_codpy_kernel = set_codpy_kernel)
#     return out

# def fun_helper_extrapolation(**kwargs):
#     set_codpy_kernel = fun_helper_base(**kwargs)
#     x_,fx_, z_,fz_ = kwargs['x'],kwargs['fx'],kwargs['z'],kwargs['fz']
#     out = fun_extrapolation(x_,fx_,x_,fx_,z_,fz_,set_codpy_kernel = set_codpy_kernel)
#     return out