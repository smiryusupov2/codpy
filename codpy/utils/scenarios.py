import abc
import numpy as np
import pandas as pd
import itertools
import time
import xarray
import copy
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from codpy.utils.data_conversion import get_matrix
from codpy.utils.data_processing import lexicographical_permutation
from codpy.utils.data_processing import column_selector
from codpy.utils.file import save_to_file
from codpy.src.core import kernel_setters
from codpy.utils.metrics import get_relative_mean_squared_error
from codpy.utils.clustering_utils import graphical_cluster_utilities, add_confusion_matrix
from codpy.utils.graphical import compare_plot_lists, multi_plot



def data_gen(N, D = None, x=None, fun = None, nabla_fun = None, type = "sto", mins=[],maxs=[],**kwargs):
    if D==None: D = x.shape[1]
    sizez = (int(N**(1/D)) + 1) ** D

    if len(mins) != D: 
        if x is not None: mins = [np.min(get_matrix(x)[:,d]) for d in range(D)]
        else: mins = np.repeat(-1.,D)
    if len(maxs) != D: 
        if x is not None: maxs = [np.max(get_matrix(x)[:,d]) for d in range(D)]
        else: maxs = np.repeat(1.,D)

    #print('data_genMD.sizez:',sizez)
    #print('data_genMD.sizex:',sizex)

    if type == "sto":
        z = np.zeros([sizez,D])
        for d in range(0,D): z[:,d] = np.random.uniform(mins[d], maxs[d],sizez) 
    else:
        Dz = int(sizez**(1/D))
        z = np.linspace(mins[0],maxs[0], Dz)
        y = z
        for i in range(0, D-1):
            y = np.linspace(mins[i+1],maxs[i+1], Dz)
            z = tuple(itertools.product(z, y))
        z = np.reshape(z,(Dz*Dz,2))  

    fx, nabla_fx = None, None
    z,permutation = lexicographical_permutation(x=z)
    if isinstance(x, pd.DataFrame): z = pd.DataFrame(z,columns = x.columns)
    if fun is not None: fz = fun(z,**kwargs)
    else: return z
    if nabla_fun is not None: 
        nabla_fz = nabla_fun(z,**kwargs)
        return z,fz,nabla_fz
    else: return z,fz

    
def execute(x_,fx_,y_,fy_,z_,fz_, Xs, Ys,fun):
    sizeXs = len(Xs)
    sizeYs = len(Ys)
    df_scores = xarray.DataArray(np.zeros( (sizeXs, sizeYs) ), dims=('Nx', 'Ny'), coords={'Nx': Xs, 'Ny' : Ys})
    df_times = xarray.DataArray(np.zeros( (sizeXs, sizeYs) ), dims=('Nx', 'Ny'), coords={'Nx': Xs, 'Ny' : Ys})
    for x in Xs:
        for y in Ys:
            start_time = time.time()
            # test = fun(x_[0:int(x)],fx_[0:int(x)],y_[0:int(y)],fy_[0:int(y)],z_,fz_)
            try:
                test = fun(x_[0:int(x)],fx_[0:int(x)],y_[0:int(y)],fy_[0:int(y)],z_,fz_)
            except :
                test = np.NaN
            # print("error:" + str(test) + " x:" + str(x) + " y:" + str(y) + "time:" + str(time.time() - start_time))
            df_scores.loc[x,y] = test
            df_times.loc[x,y] = time.time() - start_time
    return (df_scores,df_times)

def scenarios(x_, Xmax=0, X_steps=0):
    if (Xmax> 0) : Xs = np.arange(Xmax/X_steps,(Xmax/X_steps) *X_steps +1,int(Xmax/X_steps))
    else: Xs = np.arange(len(x_)/X_steps,(len(x_)/X_steps) *X_steps +1,int(len(x_)/X_steps)) 
    return Xs

def execute_function_list(**kwargs):
    fun_list_reverse = kwargs['fun_list'].copy()
    fun_list_reverse = fun_list_reverse[::-1]
    for fun_ in fun_list_reverse:
        kwargs = fun_(**kwargs)
    return kwargs



class data_generator(abc.ABC):
    index = []
    def get_index(self):
        return self.index
    index_z = []
    def get_index_z(self):
        if not len(self.index_z):
            self.index_z = [str(i) for i in range(0,len(self.fz))]
        return self.index_z

    @abc.abstractmethod
    def get_data(self,**kwargs):
        pass

    def id(self,name = "data_generator"):
        return name


    def set_data(self,**kwargs):
        self.D,self.Nx,self.Ny,self.Nz,self.Df = int(kwargs.get("D",0)),int(kwargs.get("Nx",0)),int(kwargs.get("Ny",0)),int(kwargs.get("Nz",0)),int(kwargs.get("Df",0))
        self.x,self.y,self.z,self.fx,self.fy,self.fz,self.dfx,self.dfz = [],[],[],[],[],[],[],[]
        self.map_ids = []
        if self.Ny >0 & self.Nx >0 : self.Ny = min(self.Ny,self.Nx)
        def crop(x,Nx):
            if isinstance(x,list):return [crop(y,Nx) for y in x]
            if (Nx>0 and Nx < x.shape[0]):return x[:Nx]
            return x

        if abs(self.Nx)*abs(self.Ny)*abs(self.Nz) >0:
            self.x, self.fx, self.y, self.fy, self.z, self.fz = self.get_data(**kwargs)
            self.column_selector(**kwargs)
            if bool(kwargs.get('data_generator_crop',True)):
                self.x,self.fx,self.y,self.fy,self.z,self.fz = crop(self.x,self.Nx),crop(self.fx,self.Nx),crop(self.y,self.Ny),crop(self.fy,self.Ny),crop(self.z,self.Nz),crop(self.fz,self.Nz)
            self.Ny = get_matrix(self.y).shape[0]
            if  not isinstance(self.z,list): self.Nz = get_matrix(self.z).shape[0]
            self.Nx = get_matrix(self.x).shape[0]
            self.D = get_matrix(self.x).shape[1]
            if (len(get_matrix(self.fx))):
                if self.fx.ndim == 1: self.Df = 1
                else:self.Df = self.fx.shape[1]
    def get_nb_features(self):
        return self.fx.shape[1]
 
    def copy_data(self,out):
        out.x,out.y,out.z,out.fx,out.fy,out.fz, out.dfx,out.dfz = self.x.copy(),self.y.copy(),self.z.copy(),self.fx.copy(),self.fy.copy(),self.fz.copy(),self.dfx.copy(),self.dfz.copy()
        out.D,out.Nx,out.Ny,out.Nz = self.D,self.Nx,self.Ny,self.Nz
        return out

    def get_input_data(self):
        return self.D,self.Nx,self.Ny,self.Nz,self.Df
    def get_output_data(self):
        # print(self.x)
        # print(self.fx)
        return self.x,self.y,self.z,self.fx,self.fy,self.fz,self.dfx,self.dfz

    def get_params(**kwargs) :
        return kwargs.get('data_generator',None)

    def column_selector(self,**kwargs):
        params = data_generator.get_params(**kwargs)
        if params is None : return
        params = params.get('variables_selector',None)
        if params is None : return
        variables_cols_drop = params.get('variables_cols_drop',[])
        variables_cols_keep = params.get('variables_cols_keep',[])
        values_cols_drop = params.get('values_cols_drop',[])
        values_cols_keep = params.get('values_cols_keep',[])

        if len(variables_cols_drop) or len(variables_cols_keep):
            self.x = column_selector(self.x,cols_drop = variables_cols_drop, cols_keep = variables_cols_keep)
            self.y = column_selector(self.y,cols_drop = variables_cols_drop, cols_keep = variables_cols_keep)
            self.z = column_selector(self.z,cols_drop = variables_cols_drop, cols_keep = variables_cols_keep)
        if len(values_cols_drop) or len(values_cols_keep):
            self.fx = column_selector(self.fx,cols_drop = values_cols_drop, cols_keep = values_cols_keep)
            self.fy = column_selector(self.fy,cols_drop = values_cols_drop, cols_keep = values_cols_keep)
            self.fz = column_selector(self.fz,cols_drop = values_cols_drop, cols_keep = values_cols_keep)

    def save_cd_data(object,**params):
        if params is not None:
            save_cv_data = params.get('save_cv_data',None)
            if save_cv_data is not None:
                index = save_cv_data.get('index',False)
                header = save_cv_data.get('header',False)
                x_csv,y_csv,z_csv,fx_csv,fz_csv,f_z_csv = save_cv_data.get('x_csv',None),save_cv_data.get('y_csv',None),save_cv_data.get('z_csv',None),save_cv_data.get('fx_csv',None),save_cv_data.get('fz_csv',None),save_cv_data.get('f_z_csv',None)
                if x_csv is not None :  save_to_file(object.x, file_name=x_csv,sep = ';', index=index, header=header)
                if y_csv is not None :  save_to_file(object.y, file_name=y_csv,sep = ';', index=index, header=header)
                if z_csv is not None :  save_to_file(object.z, file_name=z_csv,sep = ';', index=index, header=header)
                if fx_csv is not None :  save_to_file(object.fx, file_name=fx_csv,sep = ';', index=index, header=header)
                if fz_csv is not None :  save_to_file(object.fz, file_name=fz_csv,sep = ';', index=index, header=header)
                if f_z_csv is not None :  save_to_file(object.f_z, file_name=f_z_csv,sep = ';', index=index, header=header)

    def __init__(self,**kwargs):
        self.set_data(**kwargs)

class data_predictor(abc.ABC):
    score,elapsed_predict_time,norm_function,discrepancy_error = np.NaN,np.NaN,np.NaN,np.NaN
    set_kernel,generator = None,None

    def get_params(**kwargs) :
        return kwargs.get('data_predictor',None)
    
    def __init__(self): pass
    def __init__(self,**kwargs):
        self.set_kernel = kwargs.get('set_kernel',kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 2,1e-8 ,map_setters.set_unitcube_map))
        self.set_kernel()
        self.accuracy_score_function = kwargs.get('accuracy_score_function',get_relative_mean_squared_error)
        self.name = kwargs.get('name','data_predictor')
    def get_index(self):
        if (self.generator):return self.generator.get_index()
        else:return []
    def get_index_z(self):
        if (self.generator):return self.generator.get_index_z()
        else:return []
    def get_input_data(self):
        return self.x,self.y,self.z,self.fx,self.fy,self.fz, self.dfx, self.dfz
    def copy_data(self,out):
        out.generator, out.set_kernel = self.generator,self.set_kernel
        out.x,out.y,out.z,out.fx,out.fy, out.fz, out.dfx, out.dfz = self.x.copy(),self.y.copy(),self.z.copy(),self.fx.copy(),self.fy.copy(),self.fz.copy(),self.dfx.copy(),self.dfz.copy()
        out.f_z = self.f_z.copy()
        # out.df_z= self.df_z.copy()
        out.D,out.Nx,out.Ny,out.Nz,out.Df = self.D,self.Nx,self.Ny,self.Nz,self.Df
        out.elapsed_predict_time,out.norm_function,out.discrepancy_error,out.accuracy_score= self.elapsed_predict_time,self.norm_function,self.discrepancy_error,self.accuracy_score
        return out
    def set_data(self,generator,**kwargs):
        self.generator = generator
        self.D,self.Nx,self.Ny,self.Nz,self.Df = generator.get_input_data()
        self.x,self.y,self.z,self.fx, self.fy, self.fz, self.dfx, self.dfz = generator.get_output_data()
        self.f_z,self.df_z = [],[]
        self.elapsed_predict_time,self.norm_function,self.discrepancy_error,self.accuracy_score = np.NaN,np.NaN,np.NaN,np.NaN
        if (self.D*self.Nx*self.Ny ):
            self.preamble(**kwargs)
            start = time.time()
            self.predictor(**kwargs)
            self.elapsed_predict_time = time.time()-start
            self.validator(**kwargs)

    def column_selector(self,**kwargs):
        variables_cols_drop = kwargs.get('variables_cols_drop',[])
        variables_cols_keep = kwargs.get('variables_cols_keep',[])
        values_cols_drop = kwargs.get('values_cols_drop',[])
        values_cols_keep = kwargs.get('values_cols_keep',[])

        if len(variables_cols_drop) or len(variables_cols_keep):
            self.x = column_selector(self.x,cols_drop = variables_cols_drop, cols_keep = variables_cols_keep)
            self.y = column_selector(self.y,cols_drop = variables_cols_drop, cols_keep = variables_cols_keep)
            self.z = column_selector(self.z,cols_drop = variables_cols_drop, cols_keep = variables_cols_keep)
        if len(values_cols_drop) or len(values_cols_keep):
            self.fx = column_selector(self.fx,cols_drop = values_cols_drop, cols_keep = values_cols_keep)
            self.fy = column_selector(self.fy,cols_drop = values_cols_drop, cols_keep = values_cols_keep)
            # self.fz = column_selector(self.fz,cols_drop = values_cols_drop, cols_keep = values_cols_keep)


    def get_map_cluster_indices(self,cluster_indices=[],element_indices=[],**kwargs):
        if not len(element_indices): element_indices = self.f_z
        if not len(cluster_indices): 
            test = type(self.z)
            switchDict = {np.ndarray: self.get_index_z, pd.DataFrame: lambda : list(self.z.index)}
            if test in switchDict.keys(): cluster_indices = switchDict[test]()
            else:
                raise TypeError("unknown type "+ str(test) + " in standard_scikit_cluster_predictor.get_map_cluster_indices")

        if not len(cluster_indices): return {}
        if len(cluster_indices) == len(element_indices):
            return pd.DataFrame({'key': element_indices,'values':cluster_indices}).groupby("key")["values"].apply(list)
        else: return {}

    def preamble(self,**kwargs):
      return
    @abc.abstractmethod
    def predictor(self,**kwargs):
      pass
    
    def is_validator_compute(self,field,**kwargs):
        if 'validator_compute' in kwargs:
            debug = field in kwargs.get('validator_compute') 
            return debug
        return False

    def validator(self,**kwargs):
        kwargs['set_codpy_kernel'] = kwargs.get("set_codpy_kernel",self.set_kernel)
        kwargs['rescale'] = kwargs.get("rescale",True)
        if len(self.fx) and self.set_kernel: 
            if self.is_validator_compute(field ='norm_function',**kwargs): self.norm_function = op.norm(x= self.x,y= self.y,z= self.z,fx = self.fx,**kwargs)
        if len(self.fz)*len(self.f_z):
            if self.is_validator_compute(field = 'accuracy_score',**kwargs): self.accuracy_score = self.accuracy_score_function(self.fz, self.f_z)
        if len(self.x)*len(self.z) and self.set_kernel: 
            if ( self.is_validator_compute(field ='discrepancy_error',**kwargs)): self.discrepancy_error = op.discrepancy(x=self.x, y = self.y, z= self.z, **kwargs)


    def get_numbers(self):
        return self.D,self.Nx,self.Ny,self.Nz,self.Df

    def get_output_data(self):
        return self.f_z,self.df_z
    
    def get_params(**kwargs) :
        return kwargs.get('data_predictor',None)

    def get_new_params(self,**kwargs) :
        return kwargs

    def save_cd_data(self,**kwargs):
        params = data_predictor.get_params(**kwargs)
        if params is not None:
            x_csv,y_csv,z_csv,fx_csv,fz_csv,f_z_csv = params.get('x_csv',None),params.get('y_csv',None),params.get('z_csv',None),params.get('fx_csv',None),params.get('fz_csv',None),params.get('f_z_csv',None)
            if x_csv is not None :  self.x.to_csv(x_csv,sep = ';', index=False)
            if y_csv is not None :  self.y.to_csv(y_csv,sep = ';', index=False)
            if z_csv is not None :  self.z.to_csv(z_csv,sep = ';', index=False)
            if fx_csv is not None : self.fx.to_csv(fx_csv,sep = ';', index=False)
            if fz_csv is not None : self.fz.to_csv(fz_csv,sep = ';', index=False)
            if f_z_csv is not None: self.f_z.to_csv(f_z_csv,sep = ';', index=False)

    
    def id(self,name = ""):
        return self.name


class data_accumulator:
    def __init__(self,**kwargs):
        self.set_data(generators = [],predictors= [],**kwargs)

    def set_data(self,generators =[], predictors = [],**kwargs):
        self.generators,  self.predictors = generators,predictors

    def accumulate_reshape_helper(self,x):
        if len(x) == 0: return []
        newshape = np.insert(x.shape,0,1)
        return x.reshape(newshape)

    def accumulate(self,predictor,generator,**kwargs):
        self.generators.append(copy.copy(generator))
        self.predictors.append(copy.copy(predictor))

     
    def plot_learning_and_train_sets(self,xs=[],zs=[],title="training (red) vs test (green) variables and values",labelx='variables ',labely=' values'):
        D = len(self.generators)
        fig = plt.figure()
        d=0
        for d in range(0,D):
            g = self.generators[d]
            if (len(xs)): x = xs[d]
            else: x = self.generators[d].x[:,0]
            ax=fig.add_subplot(1,D,d+1)
            plotx,plotfx,permutation = lexicographical_permutation(x.flatten(),g.fx.flatten())
            ax.plot(plotx,plotfx,color = 'red')
            if (len(zs)): z = zs[d]
            else: z = self.generators[d].z[:,0]
            plotz,plotfz,permutation = lexicographical_permutation(z.flatten(),g.fz.flatten())
            ax.plot(plotz,plotfz,color = 'green')
            plt.xlabel(labelx)
            plt.ylabel(labely+self.predictors[d].id())
            d = d+1
        plt.title(title)

    def plot_predicted_values(self,zs=[],title="predicted (red) vs test (green) variables and values",labelx='z',labely='predicted values'):
        d = 0
        D = len(self.predictors)
        fig = plt.figure()
        for d in range(0,D):
            p = self.predictors[d]
            if (len(zs)): z = zs[d]
            else: z = self.generators[d].z[:,0]
            ax=fig.add_subplot(1,D,d+1)
            plotx,plotfx,permutation = lexicographical_permutation(z.flatten(),p.f_z.flatten())
            ax.plot(plotx,plotfx,color = 'red')
            plotx,plotfx,permutation = lexicographical_permutation(z.flatten(),p.fz.flatten())
            ax.plot(plotx,plotfx,color = 'green')
            plt.xlabel(labelx)
            plt.ylabel(labely+p.id())
            d = d+1
        plt.title(title)

    def plot_errors(self,fzs=[],title="error on predicted set ",labelx='f(z)',labely='error:'):
        d = 0
        D = len(self.predictors)
        fig = plt.figure()
        for p in self.predictors:
            ax=fig.add_subplot(1,D,d+1)
            if (len(fzs)): fz = fzs[d]
            else: fz = p.fz
            plotx,plotfx,permutation = lexicographical_permutation(get_matrix(fz).flatten(),get_matrix(p.f_z).flatten()-get_matrix(p.fz).flatten())
            ax.plot(plotx,plotfx, color= "red")
            ax.plot(plotx,get_matrix(p.fz).flatten()-get_matrix(p.fz).flatten(), color= "green")
            plt.xlabel(labelx)
            plt.ylabel(labely+p.id())
            d = d+1

        plt.title(title)

    def format_helper(self,x):
        return x.reshape(len(x),1)
    def get_elapsed_predict_times(self):
        return  np.asarray([np.round(s.elapsed_predict_time,2) for s in self.predictors])
    def get_discrepancy_errors(self):
        return  np.asarray([np.round(s.discrepancy_error,4) for s in self.predictors])
    def get_norm_functions(self):
        return  np.asarray([np.round(s.norm_function,2) for s in self.predictors])
    def get_accuracy_score(self):
        return  np.asarray([np.round(s.accuracy_score,4) for s in self.predictors])
    def get_numbers(self):
        return  np.asarray([s.get_numbers() for s in self.predictors])
    def get_Nxs(self):
        return  np.asarray([s.Nx for s in self.predictors])
    def get_Nys(self):
        return  np.asarray([s.Ny for s in self.predictors])
    def get_Nzs(self):
        return  np.asarray([s.Nz for s in self.predictors])
    def get_predictor_ids(self):
        return  np.asarray([s.id() for s in self.predictors])
    def get_xs(self):
        return  [s.x for s in self.predictors]
    def get_ys(self):
        return  [s.y for s in self.predictors]
    def get_zs(self):
        return  [s.z for s in self.predictors]
    def get_fxs(self):
        return  [s.fx for s in self.predictors]
    def get_fys(self):
        return  [s.fy for s in self.predictors]
    def get_fzs(self):
        return  [s.fz for s in self.predictors]
    def get_f_zs(self):
        return  [s.f_z for s in self.predictors]

    def confusion_matrices(self):
        return  [s.confusion_matrix() for s in self.predictors]

    def plot_clusters(self, **kwargs):
        multi_plot(self.predictors ,graphical_cluster_utilities.plot_clusters, **kwargs)

    def plot_confusion_matrices(self, **kwargs):
        multi_plot(self.predictors ,add_confusion_matrix.plot_confusion_matrix, **kwargs)

    def get_maps_cluster_indices(self,cluster_indices=[],element_indices=[],**kwargs):
        out = []
        for predictor in self.predictors:
            out.append(predictor.get_map_cluster_indices(cluster_indices=cluster_indices,element_indices=element_indices,**kwargs))
        return out


    def get_output_datas(self):
        execution_time = self.format_helper(self.get_elapsed_predict_times())
        discrepancy_errors = self.format_helper(self.get_discrepancy_errors())
        norm_function = self.format_helper(self.get_norm_functions())
        scores = self.format_helper(self.get_accuracy_score())
        numbers = self.get_numbers()
        indices = self.format_helper(self.get_predictor_ids())
        indices = pd.DataFrame(data=indices,columns=["predictor_id"])
        numbers = np.concatenate((numbers,execution_time,scores,norm_function,discrepancy_errors), axis=1)
        numbers = pd.DataFrame(data=numbers,columns=["D", "Nx","Ny","Nz","Df","execution_time","scores","norm_function","discrepancy_errors"])
        numbers = pd.concat((indices,numbers),axis=1)
        return  numbers


class standard_cluster_predictor(data_predictor,graphical_cluster_utilities):
    score_silhouette, score_calinski_harabasz, homogeneity_test, inertia,  discrepancy = np.NaN,np.NaN,np.NaN,np.NaN,np.NaN
    estimator = None

    def copy_data(self,out):
        super().copy_data(out)
        out.score_silhouette,out.score_calinski_harabasz,out.homogeneity_test,out.inertia= self.score_silhouette,self.score_calinski_harabasz,self.homogeneity_test,self.inertia
        return out

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if 'accuracy_score_function' not in kwargs: 
            self.accuracy_score_function = metrics.accuracy_score

    def validator(self,**kwargs):
        super().validator(**kwargs)
        if len(self.z)*len(self.f_z):
            try: 
                if self.is_validator_compute(field ='score_silhouette',**kwargs): self.score_silhouette = metrics.silhouette_score(self.z, self.f_z)
                if self.is_validator_compute(field ='score_calinski_harabasz',**kwargs): self.score_calinski_harabasz = metrics.calinski_harabasz_score(self.z, self.f_z)
            except:
                pass
        if len(self.fz)*len(self.f_z): 
            if self.is_validator_compute(field ='homogeneity_test',**kwargs): self.homogeneity_test = metrics.homogeneity_score(self.fz, self.f_z) 
        if (self.estimator):
            if self.is_validator_compute(field ='inertia',**kwargs): self.inertia = self.estimator.inertia_
        else:
            if self.is_validator_compute(field ='inertia',**kwargs):
                self.inertia = KMeans(n_clusters=self.Ny).fit(self.x).inertia_
        pass

class scenario_generator:
    gpa=[]
    results =[]
    data_generator,predictor,accumulator = [],[],[]
    def set_data(self,data_generator,predictor,accumulator,**kwargs):
        self.gpa.append((data_generator,predictor,accumulator))
    def __init__(self,data_generator = None,predictor= None,accumulator= None,**kwargs):
        if data_generator:self.set_data(data_generator,predictor,accumulator,**kwargs)
    def run_scenarios_cube(self,Ds, Nxs,Nys,Nzs,**kwargs):
        for d in Ds:
            for nx in Nxs:
                for ny in Nys:
                    for nz in Nzs:
                        self.data_generator.set_data(int(d),int(nx),int(ny),int(nz),**kwargs)
                        self.predictor.set_data(self.data_generator,**kwargs)
                        print("  predictor:",self.predictor.id()," d:", d," nx:",nx," ny:",ny," nz:",nz)
                        self.accumulator.accumulate(self.predictor,self.data_generator,**kwargs)
    def run_scenarios(self,list_scenarios,data_generator,predictor,accumulator,**kwargs):
        for scenario in list_scenarios:
            self.data_generator,self.predictor,self.accumulator = data_generator,predictor,accumulator
            data_generator.set_data(**scenario,**kwargs)
            predictor.set_data(data_generator,**scenario,**kwargs)
            # print("predictor:",self.predictor.id()," d:", d," nx:",nx," ny:",ny," nz:",nz)
            accumulator.accumulate(predictor,data_generator,**kwargs)
        if not len(self.results): self.results = accumulator.get_output_datas()
        else: self.results = pd.concat((self.results,accumulator.get_output_datas()))
        # print(self.results)
    def run_all(self,list_scenarios,**kwargs):
        self.results = []
        for scenario in list_scenarios:
            d,nx,ny,nz = scenario
            d,nx,ny,nz = int(d),int(nx),int(ny),int(nz)
            # print(" d:", d," nx:",nx," ny:",ny," nz:",nz)
            for data_generator,predictor,accumulator in self.gpa:
                run_scenarios(self,list_scenarios,data_generator,predictor,accumulator,**kwargs)

    def compare_plot(self,axis_label, field_label, **kwargs):
        xs=[]
        fxs=[]
        # self.results.to_excel("results.xlsx")
        groups = self.results.groupby("predictor_id")
        predictor_ids = list(groups.groups.keys())
        groups = groups[(axis_label,field_label)]
        for name, group in groups:
            xs.append(group[axis_label].values.astype(float))
            fxs.append(group[field_label].values.astype(float))
            pass
        compare_plot_lists(listxs = xs, listfxs = fxs, listlabels=predictor_ids, xscale ="linear",yscale ="linear", Show = True,**kwargs)

    def compare_plot_ax(self,axis_field_label, ax,**kwargs):
        xs=[]
        fxs=[]
        axis_label,field_label = axis_field_label[0],axis_field_label[1]
        # self.results.to_excel("results.xlsx")
        groups = self.results.groupby("predictor_id")
        predictor_ids = list(groups.groups.keys())
        groups = groups[list((axis_label,field_label))]
        for name, group in groups:
            xs.append(group[axis_label].values.astype(float))
            fxs.append(group[field_label].values.astype(float))
        pass
        compare_plot_lists({'listxs' : xs, 'listfxs' : fxs, 'ax':ax,'listlabels':predictor_ids, 'labelx':axis_label,'labely':field_label,**kwargs})

    def compare_plots(self,axis_field_labels, **kwargs):
        multi_plot(axis_field_labels,self.compare_plot_ax, **kwargs)
