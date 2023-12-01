from preamble import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


##################################### Kernels
# set_gaussian_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 0,1e-8,map_setters.set_mean_distance_map)
# set_tensornorm_kernel = kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 0,0,map_setters.set_unitcube_map)
# set_per_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussianper_kernel,2,1e-8,None)

codpy_params = {'rescale:xmax': 1000,
'rescale:seed':42,
'sharp_discrepancy:xmax':1000,
'sharp_discrepancy:seed':30,
'sharp_discrepancy:itermax':5,
'discrepancy:xmax':500,
'discrepancy:ymax':500,
'discrepancy:zmax':500,
'discrepancy:nmax':2000}

####A standard run

def standard_supervised_run(scenario_generator,scenarios_list,generator,predictor,accumulator,**kwargs):
    scenario_generator.run_scenarios(scenarios_list,generator,predictor,accumulator,**kwargs)
    if bool(kwargs.get("Show_results",True)):
        results = accumulator.get_output_datas().dropna(axis=1)
        print(results)
    if bool(kwargs.get("Show_confusion",True)):accumulator.plot_confusion_matrices(**kwargs,mp_title = "confusion matrices for "+predictor.id())
    if bool(kwargs.get("Show_maps",True)):print(accumulator.get_maps_cluster_indices())

######################### regressors ######################################""
# class codpyprRegressor(data_predictor):
    
#     def predictor(self,**kwargs):
#         if (self.D*self.Nx*self.Ny*self.Nz ):
#             self.f_z = op.projection(x = self.x,y = self.y,z = self.z, fx = self.fx,set_codpy_kernel=self.set_kernel,rescale = True,**kwargs)
#             pass
#     def id(self,name = ""):
#         return "codpy pred"

# class codpyexRegressor(data_predictor):
    
#     def predictor(self,**kwargs):
#         kwargs['set_codpy_kernel'] = kwargs.get('set_codpy_kernel',self.set_kernel)
#         kwargs['rescale'] = kwargs.get('rescale',False)
#         self.column_selector(**kwargs)
#         if (self.D*self.Nx*self.Ny*self.Nz ):
#             self.f_z = op.projection(x = self.x,y = self.x,z = self.z, fx = self.fx,**kwargs)
#     def id(self,name = ""):
#         return "codpy extra"


def codpy_Classifier(**kwargs): #label_codpy_predictor
    f_z = op.projection(**kwargs)
    out= np.zeros(f_z.shape)
    softmaxindice_ = softmaxindice(f_z)
    def helper(n): out[n,softmaxindice_[n]] = 1.
    [helper(n) for n in range(f_z.shape[0])]
    if isinstance(f_z,pd.DataFrame): out= pd.DataFrame(out,columns=f_z.columns)
    return out

def classifier_score_fun(**kwargs) :
    from sklearn.metrics import confusion_matrix
    fz,f_z = softmaxindice(mat = kwargs['fz']),softmaxindice(mat = kwargs['f_z'])
    out = confusion_matrix(fz,f_z)
    print("confusion matrix:",out)
    score = np.trace(out)/np.sum(out) 
    print("overall score :", (score * 100),"%")
    return 1.-score

def softmax_predictor(f_z): return softmaxindice(f_z)
    # out= np.zeros(f_z.shape)
    # softmaxindice_ = softmaxindice(f_z)
    # def helper(n): out[n,softmaxindice_[n]] = 1.
    # [helper(n) for n in range(f_z.shape[0])]
    # if isinstance(f_z,pd.DataFrame): out= pd.DataFrame(out,columns=f_z.columns)
    # return out

def proba_predictor(**kwargs): 
    out = op.projection(**kwargs)
    return out

def proba_classifier(**kwargs):return softmax_predictor(proba_predictor(**kwargs))

def weighted_predictor(**kwargs): 
    fx = kwargs['fx']
    weights = np.zeros(fx.shape[0])
    for col in fx.columns:
        index = list(fx[fx[col]==1.].index)
        nb = len(index)
        if nb: weights[index] = nb
    return(op.weighted_projection(weights=weights,**kwargs))

def weighted_classifier(**kwargs): return softmax_predictor(weighted_predictor(**kwargs))


def get_occurence_count(x):
    cols_ordinal = {}
    for col in x.columns:
        my_value_count = x[col].value_counts()
        my_value_count = my_value_count.loc[my_value_count.index.isin([1.0])]
        if my_value_count.empty : cols_ordinal[col] = 0
        else : cols_ordinal[col] = int(my_value_count)
    cols_ordinal = dict(sorted(cols_ordinal.items(),key=lambda x:x[1]))
    return cols_ordinal

def codpy_rl_classifier(kwargs):
    import random

    random_state=kwargs.get("random_state",42)
    max_number = kwargs.get('max_number',5000)
    # predictor = kwargs.get('predictor',proba_predictor)
    proba_predictor_ = kwargs.get('proba_predictor',proba_predictor)
    batch_size = kwargs.get('batch_size',100)
    if 'y' in kwargs:
        if kwargs['y'].shape[0] >= max_number: return kwargs

    fz= kwargs['fz']
    repartition_rem_index,repartition_rem_count = {},{}
    columns = list(kwargs.get("columns",kwargs['fz'].columns))
    for col in columns : repartition_rem_index[col] = fz[fz[col]==1.].index
    for col in columns : repartition_rem_count[col] = len(repartition_rem_index[col])
    keep_indices = set()
    probas_erreurs = {}

    def error_fun(kwargs): 
        out= pd.DataFrame.abs(kwargs['fz']-kwargs['f_z']).sort_values(ascending = False)
        return out

    def add_indices_fun(**kwargs) :
        erreurs_values = kwargs.get('error_fun',error_fun)(kwargs)
        erreurs = list(erreurs_values.index)[:batch_size]
        test = erreurs_values.loc[erreurs]
        cols_ordinal = get_occurence_count(kwargs["fy"])
        print("repartition training set:",cols_ordinal)
        print("repartition remaining set :",repartition_rem_count)
        add_indices = list()
        added = {}
        for col in cols_ordinal:
            erreurs_list = [item for item in erreurs if item in repartition_rem_index[col]]
            if len(erreurs_list) : probas_erreurs[col] = float(erreurs_values.loc[erreurs_list[0]])
            else: 
                probas_erreurs[col] = 1.
            added[col] = erreurs_list

        print("probas_erreurs:",probas_erreurs)

        for col in added:
            samples_nb = batch_size
            samples_nb= int(min(batch_size,samples_nb))
            repartition_rem_index[col] = repartition_rem_index[col].difference(added[col])
            repartition_rem_count[col] = len(repartition_rem_index[col])
            add_indices = add_indices + list(added[col][:samples_nb])
            cols_ordinal[col] += len(added[col])
        return add_indices

    # def add_indices_fun(**kwargs) :
    #     rl_weights = kwargs.get('rl_weights',{})
    #     erreurs = kwargs.get('error_fun',error_fun)(kwargs)

    #     cols_ordinal = get_occurence_count(kwargs["fy"])
    #     print("repartition training set:",cols_ordinal)
    #     print("repartition remaining set :",repartition_rem_count)
    #     add_indices = list()
    #     added = {}
    #     for col in cols_ordinal:
    #         if col in rl_weights:
    #             weight_ = float(rl_weights[col])
    #             sum_ = np.sum(list(cols_ordinal.values()))
    #             tol = int(weight_*sum_)-cols_ordinal[col]
    #             samples_nb= max(min(batch_size,tol),0)
    #         else:
    #             samples_nb = batch_size
    #         erreurs_list = erreurs[erreurs.index.isin(repartition_rem_index[col])].sort_values(by = col, ascending = False)
    #         if erreurs_list.shape[0] : probas_erreurs[col] = erreurs_list.iloc[0][col]
    #         else: 
    #             erreurs_list[col] = 0.
    #             probas_erreurs[col] = 0.
    #         added[col] = list(erreurs_list.index[:samples_nb])

    #     print("probas_erreurs:",probas_erreurs)

    #     for col in added:
    #         samples_nb = batch_size
    #         samples_nb= int(min(batch_size,samples_nb))
    #         repartition_rem_index[col] = repartition_rem_index[col].difference(added[col][:samples_nb])
    #         repartition_rem_count[col] = len(repartition_rem_index[col])
    #         add_indices = add_indices + list(added[col][:samples_nb])
    #         cols_ordinal[col] += len(added[col])
    #     return add_indices


#     score_fun_ = kwargs.get('score_fun',classifier_score_fun)
#     add_indices_fun_ = kwargs.get('add_indices_fun',add_indices_fun)

#     y,fy = pd.DataFrame(),pd.DataFrame()
#     if 'y' not in kwargs: kwargs['y']=y
#     cols_ordinal = get_occurence_count(kwargs["fz"])
#     if not kwargs['y'].shape[0]:
#         for col in cols_ordinal:
#             new_set = list(kwargs["fz"].loc[kwargs["fz"][col] == 1.].index)
#             random.Random(random_state).shuffle(new_set)
#             keep_indices.update(set(new_set[:min(batch_size,2)]))
#     else:
#         y,fy = kwargs['y'],kwargs['fy']


#     def update(keep_indices,kwargs):
#         kwargs['y'] = pd.concat([y,kwargs['z'].iloc[list(keep_indices)]],axis=0).reset_index(drop = True)
#         kwargs['fy'] = pd.concat([fy,kwargs['fz'].iloc[list(keep_indices)]],axis=0).reset_index(drop = True)
#         kwargs['x'],kwargs['fx']=kwargs['y'],kwargs['fy']
#         kwargs['f_z'] = proba_predictor_(**kwargs)
#         return kwargs
#     # def update(keep_indices,kwargs):
#     #     kwargs['x'],kwargs['fx']=kwargs['z'],kwargs['fz']
#     #     kwargs['y'] = pd.concat([y,kwargs['z'].iloc[list(keep_indices)]],axis=0).reset_index(drop = True)
#     #     kwargs['fy'] = pd.concat([fy,kwargs['fz'].iloc[list(keep_indices)]],axis=0).reset_index(drop = True)
#     #     kwargs['f_z'] = proba_predictor_(**kwargs)
#     #     return kwargs
#     kwargs = update(keep_indices,kwargs)

#     iteration = 0
#     add_indices = [0]
#     best_score = float("inf")
#     best_indices = {}
#     while kwargs['y'].shape[0] < max_number and len(add_indices) > 0:
#         add_indices = add_indices_fun_(**kwargs)
#         keep_indices = keep_indices | set(add_indices)
#         kwargs = update(keep_indices,kwargs)
#         score_ = score_fun_(**kwargs)
#         iteration = iteration+1
#         print("iteration: ", iteration, "training set size:", kwargs['y'].shape[0]," - score: ",score_," - best_score: ",best_score)
       
#     kwargs = update(keep_indices,kwargs)
#     # kwargs = update(best_indices,kwargs)
#     score_ = score_fun_(**kwargs)
#     print("final: training set size:", kwargs['x'].shape[0]," - score: ",score_)
#     return kwargs


# ############################ classifiers ###############################################
# class codpyprClassifier(codpyprRegressor,add_confusion_matrix): #label_codpy_predictor
#     def __init__(self,**kwargs):
#         super().__init__(**kwargs)
#         if 'accuracy_score_function' not in kwargs: 
#             from sklearn import metrics
#             self.accuracy_score_function = metrics.accuracy_score

#     def predictor(self,**kwargs):
#         if (self.D*self.Nx*self.Ny*self.Nz ):
#             get_proba = kwargs.get('get_proba',False)
#             kwargs['set_codpy_kernel'] = kwargs.get('set_codpy_kernel',self.set_kernel)
#             kwargs['rescale'] = kwargs.get('rescale',False)
#             fx = unity_partition(self.fx)
#             f_z = op.projection(x = self.x,y = self.y,z = self.z, fx = fx,**kwargs)
#             if get_proba:
#                 self.f_z = f_z
#             else:
#                 self.f_z = softmaxindice(f_z)
#     def id(self,name = ""):
#         return "codpy lab pred"
#     def copy(self):
#         return self.copy_data(codpyprClassifier())


# class codpyexClassifier(codpyprClassifier):
#     def copy(self):
#         return self.copy_data(codpyexClassifier())
#     def predictor(self,**kwargs):
#         if (self.D*self.Nx*self.Ny*self.Nz ):
#             get_proba = kwargs.get('get_proba',False)
#             fx = unity_partition(self.fx)
#             f_z = op.projection(x = self.x,y = self.x,z = self.z, fx = fx,set_codpy_kernel=self.set_kernel,rescale = True)
#             if get_proba:
#                 self.f_z = f_z
#             else:
#                 self.f_z = softmaxindice(f_z)
#     def id(self,name = ""):
#         return "codpy"
#         # return "codpy lab extra"

# ################### Semi_supervised ######################################""
# class codpyClusterClassifier(standard_cluster_predictor,add_confusion_matrix):
#     def copy(self):
#         return self.copy_data(codpyClusterClassifier())
#     def predictor(self,**kwargs):
#         kwargs['set_codpy_kernel'] = kwargs.get("set_codpy_kernel",self.set_kernel)
#         kwargs['rescale'] = kwargs.get("rescale",True)
#         kwargs['x'] = kwargs.get("x",self.x)
#         kwargs['z'] = kwargs.get("z",self.z)
#         self.y = alg.sharp_discrepancy( **kwargs)
#         kwargs['y'] = self.y
#         fx = alg.distance_labelling(self.x, self.y)
#         if (self.x is self.z):
#             self.f_z = fx
#         else: 
#             up = unity_partition(fx = fx)
#             debug = op.projection(fx = up,**kwargs)
#             self.f_z = softmaxindice(debug,axis=1)
#         if len(self.fx) : self.f_z = remap(self.f_z,get_surjective_dictionnary(fx,self.fx))
#         pass
#     def id(self,name = ""):
#         return "codpy"

# class codpyClusterPredictor(standard_cluster_predictor,add_confusion_matrix):
#     def copy(self):
#         return self.copy_data(codpyClusterPredictor())
#     def predictor(self,**kwargs):
#         self.y = alg.sharp_discrepancy(x = self.x, y = self.y,**kwargs)
#         fx = alg.distance_labelling(self.x, self.y)
#         if (self.x is self.z):
#             self.f_z = fx
#         else: 
#             up = unity_partition(fx = fx)
#             debug = op.projection(x = self.x,y = self.y,z = self.z,fx = up,**kwargs)
#             self.f_z = softmaxindice(debug,axis=1)
#         if len(self.fx) : self.f_z = remap(self.f_z,get_surjective_dictionnary(fx,self.fx))
#         pass

#     def id(self,name = ""):
#         return "codpy"        


# def test_predictor(my_fun):

#     D,Nx,Ny,Nz=2,2000,2000,2000
#     data_random_generator_ = data_random_generator(fun = my_fun,types=["cart","sto","cart"])
#     x, fx, y, fy, z, fz =  data_random_generator_.get_data(D=D,Nx=Nx,Ny=Ny,Nz=Nz)
#     multi_plot([(x,fx),(z,fz)],plotD,mp_title="x,f(x)  and z, f(z)",projection="3d")
#     fz_extrapolated = op.extrapolation(x,fx,x,set_codpy_kernel = kernel_setters.set_gaussian_kernel(0,1e-8,map_setters.set_standard_min_map),rescale = True)    
#     multi_plot([(x,fx),(x,fz_extrapolated)],plotD,mp_title="x,f(x)  and z, f(z)",projection="3d")
#     fz_extrapolated = op.extrapolation(x,fx,y,set_codpy_kernel = kernel_setters.set_gaussian_kernel(0,1e-8,map_setters.set_standard_min_map),rescale = True)    
#     multi_plot([(y,fy),(y,fz_extrapolated)],plotD,mp_title="x,f(x)  and z, f(z)",projection="3d")
#     fz_extrapolated = op.extrapolation(x,fx,z,set_codpy_kernel = kernel_setters.set_gaussian_kernel(0,1e-8,map_setters.set_standard_min_map),rescale = True)    
#     multi_plot([(x,fx),(z,fz_extrapolated)],plotD,mp_title="x,f(x)  and z, f(z)",projection="3d")

# def test_nablaT_nabla(my_fun,nabla_my_fun,set_kernel):
#     D,Nx,Ny,Nz=2,2000,2000,2000
#     data_random_generator_ = data_random_generator(fun = my_fun,nabla_fun = nabla_my_fun, types=["cart","cart","cart"])
#     x,y,z,fx,fy,fz,nabla_fx,nabla_fz,Nx,Ny,Nz =  data_random_generator_.get_raw_data(D=D,Nx=Nx,Ny=Ny,Nz=Nz)
#     f1 = op.nablaT(x,y,z,op.nabla(x,y,z,fx,set_codpy_kernel = set_kernel,rescale = True))
#     f2 = op.nablaT_nabla(x,y,fx)
#     multi_plot([(x,f1),(x,f2)],plot_trisurf,projection='3d')


# def test_withgenerator(my_fun):
#     set_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel,0,1e-8,map_setters.set_standard_min_map)
#     D,Nx,Ny,Nz=2,1000,1000,1000
#     scenarios_list = [ (D, 100*i, 100*i ,100*i ) for i in np.arange(1,5,1)]
#     if D!=1: projection="3d"
#     else: projection=""

#     data_random_generator_ = data_random_generator(fun = my_fun,types=["cart","sto","cart"])
#     x,y,z,Nx,Ny,Nz =  data_random_generator_.get_raw_data(D=1,Nx=5,Ny=1000,Nz=0)

#     x, fx, y, fy, z, fz =  data_random_generator_.get_data(D=D,Nx=Nx,Ny=Ny,Nz=Nz)
    
#     multi_plot([(x,fx),(z,fz)],plotD,mp_title="x,f(x)  and z, f(z)",projection=projection)

#     scenario_generator_ = scenario_generator()
#     scenario_generator_.run_scenarios(scenarios_list,data_random_generator_,codpyexRegressor(set_kernel = set_kernel),
# data_accumulator())
#     list_results = [(s.z,s.f_z) for s in scenario_generator_.accumulator.predictors]
#     multi_plot(list_results,plot1D,mp_max_items = 2)





if __name__ == "__main__":
    pass
    # set_kernel = kernel_setters.kernel_helper(
    # kernel_setters.set_tensornorm_kernel, 0,1e-8 ,map_setters.set_unitcube_map)
    # test_predictor(my_fun)
    # test_nablaT_nabla(my_fun,nabla_my_fun,set_kernel)
