import glob
import pickle
from preamble import *
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from unidecode import unidecode
import chardet
from operator import is_not
#from functools import partial
import random

cebpl_path =os.path.dirname(os.path.realpath(__file__)) 

def get_cebpl_param():

    cebpl_param = {###############################codpy internals
    'rescale_kernel':{'max': 1000,'seed':42},
    'discrepancy:xmax':500,
    'discrepancy:ymax':500,
    'discrepancy:zmax':500,
    'discrepancy:nmax':2000,
    'num_threads':25,
    # 'set_codpy_kernel' : kernel_setters.kernel_helper(kernel_setters.set_linear_regressor_kernel, 2,1e-8 ,None),
    # 'set_codpy_kernel' : kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 0,1e-8 ,map_setters.set_unitcube_map),
    # 'set_codpy_kernel' : kernel_setters.kernel_helper(kernel_setters.set_sumnorm_kernel, 0,1e-8 ,map_setters.set_unitcube_map),
    'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_matern_norm_kernel, 0,1e-8 ,map_setters.set_standard_mean_map),
    # 'set_codpy_kernel' : kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 0,1e-8 ,map_setters.set_standard_min_map),
    'rescale' : True,
    ############################### cebpl specific data loader
    ############################### cebpl specific data loader
    'transformer_dic' : {'DATE_NAIS_INCD_MAJR':get_float, 'DATE_FIN_INCD_MAJR':get_float,'TOP_RA':get_str,'CODE_SEGM_COMP':get_str,'TYPE_INCIDENT':get_str},
    ############################### debug = True : parallel, False = no parallel
    'debug':True,
    'batch_size':100,
    'random_state':42,
    # 'debug':False,
    ############################### to handle specific columns
    'cols_drop':['DATE_MISE_A_DISPO','ID_PERS','LIBL_CSP','DATE_R_AMIA', 'DATE_FIN_INCD_MAJR'], #'TOP_RA' 
    'cols_keep': [],
    'keep_columns':['SURF_FINN_GLBL_MOYE_MOIS','TYPE_INCIDENT'],
    'cat_cols_include': ['TOP_ARA'],
    ############################### cebpl full database and output files
    'data_path' : os.path.join(cebpl_path,"data","cebpl_data.csv"), ##############
    # 'data_path' : os.path.join(cebpl_path,"data","data_test.csv"),
    'transformed_data_path':os.path.join(cebpl_path,"data","data_test-transformed.gz"), 
    'rejected_ids_path' : os.path.join(cebpl_path,"data","rejected_ids.csv"),
    'error_data_path':os.path.join(cebpl_path,"data","errors.csv"),
    'raw_x_csv': os.path.join(cebpl_path,"data","raw_x.csv"),
    'raw_fx_csv': os.path.join(cebpl_path,"data","raw_fx.csv"),
    'raw_y_csv': os.path.join(cebpl_path,"data","raw_y.csv"),
    'raw_z_csv': os.path.join(cebpl_path,"data","raw_z.gz"),
    'raw_fz_csv': os.path.join(cebpl_path,"data","raw_fz.csv"),
            ############################# save training / test set / ground truth values for checking


            ############################# variable selector 
    'variables_selector': kbest_variable_selector,
    # 'error_fun' : get_mean_squared_error,
    'target_cols':['DUREE', 'DATE_FIN_INCD_MAJR','TOP_ARA'],
    'target_classifier_cols':['TOP_ARA'],
    # 'variables_cols_keep': ['DATE_NAIS_INCD_MAJR'],
    'variables_cols_keep': ['MT_EPRG_CONT_REEL_MOIS_PRCD'],
    'predictor':RF_classifier_predictor,
    'proba_predictor':RF_proba_predictor,
    # 'predictor':weighted_classifier,
    # 'proba_predictor':weighted_predictor,
            ############################# ftp params 
    'params_FTP' : {"FTP_USER" : "A44-PR-MPG-PARTNERS1@gce",
        "FTP_PASS" : "S'q4W<oy",
        "FTP_HOST": "ftp.esi.caisse-epargne.fr", 
        "FTP_PORT":990,
        },
    'params_ESI' : { 'date_format' : '%Y%m%d',
        'ESI_Path_ALLER' : ["/Fichiers","ALLER_CEBPL_MPG"],
        'ESI_Path_RETOUR' : ["/Fichiers","RETOUR_MPG_CEBPL"],
        'ESI_Path_RETOUR_ANALYSES' : ["/Fichiers","FICHIERS_RETOUR_ANALYSES"],
        "ESI_DIR": os.path.join(cebpl_path,"ESIMPG-DATA"),
        'Path_IN' : os.path.join(cebpl_path,"ESIMPG-DATA","in"),
        'Path_TRANS' : os.path.join(cebpl_path,"ESIMPG-DATA","TRANS"),
        'Path_OUT' : os.path.join(cebpl_path,"ESIMPG-DATA","out"),
        'Path_ARCHIVE' : os.path.join(cebpl_path,"ESIMPG-DATA","archive"),
        'suffix_ALLER' : "Flux_aller_IA_CEBPL_MPG.csv",
        'suffix_RETOUR' : "Flux_retour_IA_CEBPL_MPG.csv",
        "local_suppress_old_files_days": 7,
        "remote_suppress_old_files_days": 0,
        "clean_files":False
        },
    'CEBPL_update_params' : {
        'id_file_path': os.path.join(cebpl_path,"ESIMPG-DATA", 'update', 'IDs.csv'), 
        'pkl_processed': os.path.join(cebpl_path,"ESIMPG-DATA", 'update', 'processed_files_list.pkl'),
        'pkl_analysis_processed': os.path.join(cebpl_path,"ESIMPG-DATA", 'update', 'analysis_processed_files_list.pkl'),
        'data_folder_path' : os.path.join(cebpl_path,"data"),
        'data_path' : os.path.join(cebpl_path,"data","cebpl_data.csv"),
        'merged_data_path': os.path.join(cebpl_path,"ESIMPG-DATA", 'update', 'merged.csv'),
        'merged_analysis_path': os.path.join(cebpl_path,"ESIMPG-DATA",'update', 'analysis_merged.csv'),
        'archive_path':  os.path.join(cebpl_path,"ESIMPG-DATA", 'in'),
        'source_analysis_path' : os.path.join(cebpl_path,"ESIMPG-DATA",'analysis'),
        'update_folder_path' : os.path.join(cebpl_path,"ESIMPG-DATA",'update'),
        'last_merged_dates' : os.path.join(cebpl_path,"ESIMPG-DATA",'update','last_merged_dates.pkl'),
        'days' : 0
    },
    'sep':",",
    'output': os.path.join(cebpl_path,"data","output.csv")
    }
    return cebpl_param

def get_crossARA_params(kwargs = None):
    cross_validation_dic = {
        'training_x_csv': os.path.join(cebpl_path,"data","cross_ARA","x.csv"),
        'training_fx_csv': os.path.join(cebpl_path,"data","cross_ARA","fx.csv"),
        'training_y_csv': os.path.join(cebpl_path,"data","cross_ARA","y.csv"),
        'training_z_csv': os.path.join(cebpl_path,"data","cross_ARA","z.gz"),
        'training_fz_csv': os.path.join(cebpl_path,"data","cross_ARA","fz.csv"),
        'training_f_z_csv': os.path.join(cebpl_path,"data","cross_ARA","f_z.csv"),
        'test_z_csv': os.path.join(cebpl_path,"data","cross_ARA","test_z.gz"),
        'test_fz_csv': os.path.join(cebpl_path,"data","cross_ARA","test_fz.csv"),
        'test_f_z_csv': os.path.join(cebpl_path,"data","cross_ARA","test_f_z.csv"),
        'cross_validation_train_ARA_file_name': os.path.join(cebpl_path,"data","cross_ARA","cross_validation_train_ARA.csv"),
        'cross_validation_test_ARA_file_name': os.path.join(cebpl_path,"data","cross_ARA","cross_validation_test_ARA.csv"),
        'variables_selector_csv': os.path.join(cebpl_path,"data","cross_ARA","variables_selector.csv"),
        'rl_weights':{'TOP_ARA_False':0.5},
        'score_fun':mean_classifier_score_wrapper,
        'target_cols':['DUREE', 'TOP_ARA'],
        'target_classifier_cols':['TOP_ARA'],
        'cols_drop':['DATE_MISE_A_DISPO','ID_PERS','LIBL_CSP','DATE_R_AMIA','AFFECTATION','TOP_ARA','ARA','DUREE', 'DATE_FIN_INCD_MAJR','DATE_NAIS_INCD_MAJR'],
        'train_size':.9,
        #codpy learning algorithms
        # 'batch_size':100,
        # 'get_calibrated_training_set':codpy_rl_classifier,
        # 'max_number':5000
        #other learning machine parameters
        'get_calibrated_training_set':get_trivial_training_set,
        'proba_predictor':RF_proba_predictor,        
        'predictor':RF_classifier_predictor        
    }
    if kwargs is None:kwargs = {**get_cebpl_params(),**cross_validation_dic}
    else: kwargs = {**kwargs,**cross_validation_dic}
    return kwargs

def get_crossOPPO_params(kwargs = None):
    cross_validation_dic = {
        'training_x_csv': os.path.join(cebpl_path,"data","cross_OPPO","x.csv"),
        'training_fx_csv': os.path.join(cebpl_path,"data","cross_OPPO","fx.csv"),
        'training_y_csv': os.path.join(cebpl_path,"data","cross_OPPO","y.csv"),
        'training_z_csv': os.path.join(cebpl_path,"data","cross_OPPO","z.gz"),
        'training_fz_csv': os.path.join(cebpl_path,"data","cross_OPPO","fz.csv"),
        'training_f_z_csv': os.path.join(cebpl_path,"data","cross_OPPO","f_z.csv"),
        'test_z_csv': os.path.join(cebpl_path,"data","cross_OPPO","test_z.gz"),
        'test_fz_csv': os.path.join(cebpl_path,"data","cross_OPPO","test_fz.csv"),
        'test_f_z_csv': os.path.join(cebpl_path,"data","cross_OPPO","test_f_z.csv"),
        'cross_validation_train_OPPO_file_name': os.path.join(cebpl_path,"data","cross_OPPO","cross_validation_train_OPPO.csv"),
        'cross_validation_test_OPPO_file_name': os.path.join(cebpl_path,"data","cross_OPPO","cross_validation_test_OPPO.csv"),
        'variables_selector_csv': os.path.join(cebpl_path,"data","cross_OPPO","variables_selector.csv"),
        'rl_weights':{'AFFECTATION_ARA avec OPPO':0.6},
        'score_fun':mean_classifier_score_wrapper,
        'target_cols':['AFFECTATION'],
        'target_classifier_cols':['AFFECTATION'],
        'cols_drop':['DATE_MISE_A_DISPO','ID_PERS','LIBL_CSP','DATE_R_AMIA','AFFECTATION','TOP_ARA','ARA','DUREE', 'DATE_FIN_INCD_MAJR','DATE_NAIS_INCD_MAJR'],
        'train_size':.9,
        #codpy learning algorithms
        # 'batch_size':10,
        # 'get_calibrated_training_set':codpy_rl_classifier,
        # 'max_number':5000
        #other learning machine parameters
        'get_calibrated_training_set':get_trivial_training_set,
        'proba_predictor':RF_proba_predictor,        
        'predictor':RF_classifier_predictor        
    }
    if kwargs is None:kwargs = {**get_cebpl_params(),**cross_validation_dic}
    else: kwargs = {**kwargs,**cross_validation_dic}
    return kwargs

def get_cebpl_params():
    cebpl_params = get_cebpl_param()
    gm = global_merge(**cebpl_params)
    data_path = os.path.join(cebpl_path,"data") #cebpl_params['data_path']
    latest_file_date = gm._latest_file_date(data_path)
    if latest_file_date == '':
        cebpl_params['data_path'] = os.path.join(cebpl_path, "DATA", f"{gm.base_file}.csv")
    else:
        cebpl_params['data_path'] = os.path.join(cebpl_path, "DATA", f"{gm.base_file}_{latest_file_date.strftime('%Y%m%d')}.csv")
    return cebpl_params
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
############################################### Predictors ##########################################################################################
##########################################################################################################################################################
##########################################################################################################################################################


def RF_classifier_predictor(**kwargs):
    x,y,z,fx = kwargs.get('x',[]),kwargs.get('y',[]),kwargs.get('z',[]),kwargs.get('fx',[])
    x,y,z = column_selector([x,y,z],**kwargs)
    x,y,z,fx = get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx)
    fx = fx[:,1]
    model = RandomForestClassifier(n_estimators=400, n_jobs=-1)
    col = kwargs['fx'].columns
    def fun(x,z,fx):
        model.fit(x, fx)
        probabilities = model.predict(z)
        return pd.DataFrame(unity_partition(probabilities), columns = col)
    
    if isinstance(z,list):return[fun(x,z_,fx) for z_ in z]
    return fun(x,z,fx)

def RF_proba_predictor(**kwargs):
    x,z,fx = kwargs.get('x',[]),kwargs.get('z',[]),kwargs.get('fx',[])
    x,z = column_selector([x,z],**kwargs)
    x,z,fx = get_matrix(x),get_matrix(z),get_matrix(fx)
    if fx.shape[1] == 2:
        fx = fx = fx[:,1]
        col = kwargs['fx'].columns
    else:
        fx = fx[:,3]
        col = kwargs['fx'].iloc[:,2:].columns 
    model = RandomForestClassifier(n_estimators=400, n_jobs=-1)
    
    def fun(x,z,fx):
        model.fit(x, fx)
        probabilities = model.predict_proba(z)
        return pd.DataFrame(probabilities, columns = col)
    
    if isinstance(z,list):return[fun(x,z_,fx) for z_ in z]
    return fun(x,z,fx)


##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
########################################################## Database builder ##############################################################################
##########################################################################################################################################################
##########################################################################################################################################################


def format_time_deltas_days(x):
    type_ = type(x)
    if type_ == list: return [format_time_deltas_days(y) for y in x]
    if type_ == np.array: return np.array([format_time_deltas_days(y) for y in x])
    if type_ == np.ndarray: 
        out = [int(get_float(y)) for y in x]
        return np.array(out)
    if type_ == datetime.timedelta:
        days = x.days
        return "{:d}".format(days)
    # return "{:d}:{:02d}:{:02d}".format(hours, minutes, seconds)

########### global variables

date_format = '%d/%m/%Y'
transformed_data_path = os.path.join(cebpl_path,"data","data-transformed.gz")

############  global functions

def get_str(vals):
    if isinstance(vals,list): return [get_str(v) for v in vals]
    v=unidecode(str(vals))
    if len(v) == 0:
        return 'nan'
    return v

def get_train_test(datas,train_size=0.5, random_state=42, id_key = "ID_PERS"):
    ids = list(get_ids(datas,id_key))
    ids_train, ids_test = train_test_split(ids, train_size = train_size, random_state = random_state)
    return datas.loc[datas[id_key].isin(ids_train)], datas.loc[datas[id_key].isin(ids_test)]

def get_ids(datas, id_key = "ID_PERS"):
    temp = get_idgroups(datas,id_key)
    if my_len(temp) :
        return list(temp.groups.keys())

def get_idgroups(datas, id_key = "ID_PERS"):
    if my_len(datas): return datas.groupby(id_key)
    return[]

def get_min_date(group,format=date_format):
    return get_float(np.min(pd.to_datetime(group[1]["DATE_NAIS_INCD_MAJR"], format=format)))

def get_min_dates(df,format=date_format):
    return get_float(np.min(pd.to_datetime(df["DATE_NAIS_INCD_MAJR"], format=format)))

def get_max_dates(df,format=date_format):
    return get_float(np.max(pd.to_datetime(df["DATE_FIN_INCD_REEL"], format=format)))

def get_max_date(group,nan = np.nan,format=date_format):
    ####temp = get_float(np.max(pd.to_datetime(group[1]["DATE_FIN_INCD_REEL"], format=format)), nan)
    return get_float(np.max(pd.to_datetime(group[1]["DATE_FIN_INCD_REEL"], format=format)), nan_val = nan)

def get_time_delta(group,nan = np.nan,format=date_format):
    min_ = get_min_date(group,format=format)
    max_ = get_max_date(group,nan = nan, format=format)
    delta = max_ - min_
    return max_ - min_

def split_values(datas,columns): 
    x,fx = datas.drop(columns, axis = 1),datas[columns]
    return x,fx


class cebpl_data_transformer():
    x_data  = []
    time_grid_ = []
    rejected_ids_ = pd.DataFrame()
    min_date_ = []
    max_date_ = []
    def get_affectation(datas):
        def helper(data):
            if data == 'AGENCE': return 0.
            if data == 'ARA sans OPPO': return 1.
            if data == 'ARA avec OPPO': return 2.
            raise ValueError('unknown AFFECTATION')
        out= datas.apply(helper)
        return out
    def build_parallel_helper(col_name,**kwargs):
        #print(group[0]) # print ID_PERS
        datas = kwargs['data']
        transformer_dic = dict(kwargs['transformer_dic'])
        if col_name in transformer_dic: 
            fun = transformer_dic[col_name]
            datas = list(datas[col_name])
            out = fun(datas)
            out = pd.Series(data = out, name = col_name)
            return out
        return pd.Series(datas[col_name].to_list(), name = col_name)

    def build(kwargs): 
        sep = kwargs.get('sep',",")
        transformed_data_path = kwargs["transformed_data_path"]
        def fun(data,transformed_data_path):
            params = kwargs.copy()
            if len(transformed_data_path) and os.path.exists(transformed_data_path):
                transformed_data = read_csv(transformed_data_path,sep = sep)
            else:
                if not isinstance(data,pd.DataFrame):
                    data = read_csv(data,**kwargs)
                params['data'] = data
                transformed_data = parallel_task(param_list = list(data.columns),fun = cebpl_data_transformer.build_parallel_helper,**params)
                # [print(s.name,":",len(s),"**") for s in transformed_data]
                transformed_data = pd.concat(transformed_data,axis=1)
                # [print(s,":",len(transformed_data[s]),"**") for s in transformed_data]
                params['split'] = True
                params['cols_keep'] = get_cebpl_params()['cols_keep']
                transformed_data,other = column_selector(transformed_data,**params)


                cat_cols_include = kwargs.get('cat_cols_include',['TOP_ARA','ARA sans OPPO'])
                transformed_data = hot_encoder(transformed_data,cat_cols_include=cat_cols_include)
                transformed_data = pd.concat([transformed_data,other],axis=1)
                transformed_data.fillna(0.,inplace=True)
                # df = pd.concat(transformed_data,axis=1)
                # columns = [s.replace(' ','_') for s in transformed_data.columns]
                # transformed_data = pd.DataFrame(transformed_data.values,columns=columns)
                if len(transformed_data_path):
                    transformed_data.to_csv(transformed_data_path,sep = sep, index=False,compression='gzip')
                    pass
            return transformed_data
        if isinstance(transformed_data_path,dict): 
            kwargs['transformed_data'] = [fun(r,t) for r,t in transformed_data_path.items()]
        else: 
            kwargs['transformed_data']=fun(kwargs["data"],transformed_data_path)
        return kwargs


    def set_raw_data(kwargs):
        kwargs = load_csv('data_path', **kwargs)
        def fun(funx,**kwargs):
            if isinstance(funx,pd.DataFrame):
                if 'AFFECTATION' in funx.columns: 
                    funx['TOP_ARA'] = funx['AFFECTATION'] != 'AGENCE'
                    funx['TOP_ARA'] = funx['AFFECTATION'] != 'AGENCE'
                    funx['ARA sans OPPO'] = funx['AFFECTATION'] != 'ARA sans OPPO'
                    funx.dropna(subset=['AFFECTATION'],inplace=True)
            return funx
        if isinstance(kwargs['data'],list) : kwargs['data']=[fun(x,**kwargs) for x in kwargs['data']]
        else: kwargs['data'] = fun(kwargs['data'],**kwargs)
        return kwargs

    def get_raw_data(kwargs):
        if kwargs.get('data',False) == 0: kwargs = cebpl_data_transformer.set_raw_data(kwargs)
        return kwargs

    def get_transformed_data(kwargs):
        if "data" not in kwargs.keys():
            kwargs = cebpl_data_transformer.set_raw_data(kwargs)
        kwargs = cebpl_data_transformer.build(kwargs)
        return kwargs
        

    def get_datas(**kwargs):
        x,fx = cebpl_data_transformer.get_raw_xfx(**kwargs)
        y = cebpl_data_transformer.get_raw_y(**kwargs)
        z,fz = cebpl_data_transformer.get_raw_zfz(**kwargs)
        return x,fx,y,[],z,fz
    def set_datas(**kwargs):
        datas = kwargs.get("transformed_data",cebpl_data_transformer.get_transformed_data(kwargs)["transformed_data"])
        # print(datas.columns)
        datas = datas.sample(frac=1.).reset_index(drop = True)
        target_cols = kwargs['target_cols']
        target_cols = get_matching_cols(datas,target_cols)

        weight = 1./len(target_cols)
        
        Nx = kwargs.get('Nx',datas.shape[0])
        size_ = int(Nx*weight)

        z,fz = split_values(datas,target_cols)

        x,fx = list(),list()
        index_taken = set()
        for col in target_cols :
            test = datas.loc[datas[col] == 1.]
            c,d = split_values(test,target_cols)
            c,d = c.head(size_),d.head(size_)
            index_taken.update(c.index)
            x.append(c),fx.append(d)

        x,fx = pd.concat(x),pd.concat(fx)
        y = x

      
        def save(datas,**kwargs):
            arg_file = kwargs.get('file',"")
            compression = kwargs.get('compression',None)
            if len(kwargs.get(arg_file)) and len(datas) :  datas.to_csv(kwargs[arg_file],sep = kwargs.get('sep',';'), index=False,compression=compression)
        save(x,file='raw_x_csv',**kwargs)
        save(fx,file='raw_fx_csv',**kwargs)
        save(z,file='raw_z_csv',compression='gzip',**kwargs)
        save(y,file='raw_y_csv',**kwargs)
        save(fz,file='raw_fz_csv',**kwargs)

    def get_raw_xfx(**kwargs):
        raw_x_csv,raw_fx_csv = kwargs.get('raw_x_csv',[]),kwargs.get('raw_fx_csv',[])
        if not os.path.exists(raw_x_csv) or not os.path.exists(raw_fx_csv):cebpl_data_transformer.set_datas(**kwargs)
        return read_csv(raw_x_csv,**kwargs),read_csv(raw_fx_csv,**kwargs)

    def get_raw_y(**kwargs):
        raw_y_csv = kwargs.get('raw_y_csv',[])
        if not os.path.exists(raw_y_csv) :cebpl_data_transformer.set_datas(**kwargs)
        return read_csv(raw_y_csv,**kwargs)

    def get_raw_zfz(**kwargs):
        raw_z_csv,raw_fz_csv = kwargs.get('raw_z_csv',[]),kwargs.get('raw_fz_csv',[])
        if not os.path.exists(raw_z_csv) or not os.path.exists(raw_fz_csv) :self.set_datas(**kwargs)
        return read_csv(raw_z_csv,**kwargs),read_csv(raw_fz_csv,**kwargs)
   

def read_csv(file_path,**kwargs):
    sep = kwargs.get('sep',',')
    # encoding = kwargs.get('encoding',"ISO-8859-1")
    # encoding = kwargs.get('encoding',"UTF-8")
    encoding = kwargs.get('encoding',"unicode_escape")
    
    if file_path and os.path.exists(file_path):
        try : return pd.read_csv(file_path,sep=sep,encoding=encoding)
        except: 
            print("unable to load file:",file_path)
            print(sys.exc_info()[0])
    else:return pd.DataFrame()

def load_csv(arg_file = "",**kwargs):
    if isinstance(arg_file,list):
        kwargs['data'] = [load_csv(a,**kwargs)['data'] for a in arg_file]
        return kwargs
    files_path = kwargs.get(arg_file,None)
    if isinstance(files_path,dict):
        files_path = list(files_path.keys())
    if isinstance(files_path,str):
        kwargs['data'] = read_csv(files_path,**kwargs) 
    if isinstance(files_path,list):
        out = [read_csv(f,**kwargs) for f in files_path]
        for o,f in zip(out,files_path):
            if o is None:
                kwargs[arg_file].remove(f)
        kwargs['data'] = list(filter(partial(is_not, None), out))
    return kwargs

def to_csv(datas,file_path,**kwargs): 
    if isinstance(datas,list): return [to_csv(d,a,**kwargs) for d,a in zip(datas,file_path)]
    datas.to_csv(file_path,sep = kwargs.get('sep',','), index= kwargs.get("index",False), compression = kwargs.get("compression",None))
def save_csv(datas,arg_file = "",**kwargs): 
    if len(kwargs.get(arg_file)) and len(datas) :  
        to_csv(file_path = kwargs[arg_file],datas =datas,sep = kwargs.get('sep',','), index= kwargs.get("index",False), compression = kwargs.get("compression",None))

def transform(kwargs = None):
    if kwargs is None: kwargs = get_cebpl_params()
    cebpl_data_transformer_ = cebpl_data_transformer(**kwargs)
    print("number of lines in raw data:", cebpl_data_transformer_.get_raw_data(**kwargs).shape[0])
    print("number of columns in raw data:", cebpl_data_transformer_.get_raw_data(**kwargs).shape[1])
    kwargs['datas'] = cebpl_data_transformer_.get_transformed_data(**kwargs)
    return kwargs

def get_raw_ARA_set(kwargs = None):
    if kwargs is None: kwargs = get_cebpl_params()
    x,fx,y,fy,z,fz = cebpl_data_transformer.get_datas(**kwargs)
    kwargs = {**kwargs,**get_crossARA_params()}
    kwargs['fx'],kwargs['fy'],kwargs['fz'] = fx,fy,fz
    kwargs['x'],kwargs['y'],kwargs['z'] = x,y,z
    return kwargs

def get_raw_OPPO_set(kwargs = None) :
    if kwargs is None: kwargs = get_raw_ARA_set()
    kwargs = {**kwargs,**get_crossOPPO_params()}
    x,y,z,fx,fy,fz = kwargs['x'],kwargs['y'],kwargs['z'],kwargs['fx'],kwargs['fy'],kwargs['fz']
    x,y,z=x.loc[x['AFFECTATION_AGENCE'] == 0],y.loc[y['AFFECTATION_AGENCE'] == 0],z.loc[z['AFFECTATION_AGENCE'] == 0]
    x,y,z = x.drop('AFFECTATION_AGENCE', axis = 1),y.drop('AFFECTATION_AGENCE', axis = 1),z.drop('AFFECTATION_AGENCE', axis = 1)
    target_cols = get_matching_cols(x,kwargs['target_cols'])

    fx,fz = x[target_cols],z[target_cols]
    kwargs['x'],kwargs['y'],kwargs['z'],kwargs['fx'],kwargs['fy'],kwargs['fz'] = x,y,z,fx,fy,fz
    return kwargs


def raw_reproductibility_test(kwargs = None):
    if kwargs is None: kwargs = get_raw_ARA_set()
    f_z = cebpl_predictor_var(kwargs)
    print("reproductibility test with raw datas:",get_mean_squared_error(f_z,kwargs['fz']))
    return kwargs


def get_calibration_sets(kwargs = None):
    if kwargs is None: kwargs = get_variable_selector()
    x,fx,y,z,fz = load_csv(['x_csv','fx_csv','y_csv','z_csv','fz_csv'],**kwargs)
    if True in [x.empty,fx.empty,y.empty,z.empty,fz.empty]:
        x,fx,y,z,fz = kwargs['x'],kwargs['fx'],kwargs['y'],kwargs['z'],kwargs['fz']
        save_csv(x,'x_csv',**kwargs)
        save_csv(fx,'fx_csv',**kwargs)
        save_csv(z,'z_csv',**kwargs)
        save_csv(y,'y_csv',**kwargs)
        save_csv(fz,'fz_csv',**kwargs)
    kwargs['x'],kwargs['fx'],kwargs['y'],kwargs['z'],kwargs['fz'] = x,fx,y,z,fz            

    return kwargs

def reproductibility_test(kwargs = None):
    if kwargs is None: kwargs = get_variable_selector()
    f_z = cebpl_predictor_var(kwargs)
    print("reproductibility test with selected vars:",get_mean_squared_error(f_z,kwargs['fz']))
    return kwargs

def output_confusion_scores(kwargs):
    z,fz,f_z = kwargs['z'],kwargs['fz'],kwargs['f_z']
    fz,f_z = softmaxindice(mat = fz),softmaxindice(mat = f_z)
    out = metrics.confusion_matrix(fz,f_z)
    def plot_fun(out,**kwargs):plot_confusion_matrix(out,labels = ['AGENCE','SANS OPPO','OPPO'],fmt=".1f",**kwargs, fontsize=6)
    outper = out * (100./np.sum(out))
    # print("confusion matrix:")
    # print(out)
    print("overall score:", np.trace(out)/np.sum(out)*100)
    print("class average score:", (1.-mean_classifier_score_fun(kwargs['fz'],kwargs['f_z']))*100)
    # multi_plot([out,outper],fun_plot = plot_fun,mp_figsize=(10, 10))
    pass

def balanced_classifier_train_test_split(**kwargs):
    target_cols = get_matching_cols(kwargs["fz"],kwargs["target_cols"])
    train_size, random_state=kwargs.get("train_size",0.5), kwargs.get("random_state",42)
    z,fz = kwargs["z"].reset_index(drop=True),kwargs["fz"].reset_index(drop=True)
    z,fz = kwargs["z"].reset_index(drop=True),kwargs["fz"].reset_index(drop=True)
    if kwargs.get("shuffle",True):
        zindex = list(z.index)
        random.Random(random_state).shuffle(zindex)
        z,fz = z.iloc[zindex].reset_index(drop=True),fz.iloc[zindex].reset_index(drop=True)
    repartition,global_cut = {},{}
    for col in target_cols:
        repartition[col] = list(fz[fz[col] == 1.].index)
        global_cut[col]  = int(len(repartition[col])*train_size)
    repartition = dict(sorted(repartition.items(),key=lambda x:len(x[1])))
    def get_cut(**kwargs):
        cut = {}
        max_ = kwargs.get('max',10e+10)
        for col in target_cols:
            cut_ = int(len(repartition[col])*train_size)
            cut_ = min(max_,cut_)
            cut[col] = min(cut_,global_cut[col])
        return cut

    cut = kwargs.get("cut",get_cut(**kwargs))
    outz,outfz,test_z,test_fz = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    for col in target_cols:
        outz,test_z = pd.concat([outz,z.iloc[repartition[col][:cut[col]]]]),pd.concat([test_z,z.iloc[repartition[col][ max(cut[col],global_cut[col]): ] ] ])
        outfz,test_fz = pd.concat([outfz,fz.iloc[repartition[col][:cut[col]]]]),pd.concat([test_fz,fz.iloc[repartition[col][max(cut[col],global_cut[col]):]]])


    return outz.reset_index(drop=True),outfz.reset_index(drop=True),test_z.reset_index(drop=True),test_fz.reset_index(drop=True)

def get_cross_validation(arg_file,sortcol,compare_col,kwargs = None):
    proba_predictor_ = kwargs.get('proba_predictor',RF_proba_predictor)
    f_z = proba_predictor_(**kwargs)
    columnsfz = list(kwargs["fz"].columns)
    out = pd.DataFrame(f_z,columns = columnsfz)
    columnsf_z = [col + "_ref" for col in columnsfz]
    out['ID_PERS'],out[columnsf_z] = kwargs['z']['ID_PERS'],kwargs['fz'][columnsfz]
    out.sort_values(by = sortcol, ascending = False,inplace = True)
    out = out.assign(ABAC=out.loc[:, compare_col]).reset_index(drop = True)
    out['ABAC'] = np.cumsum(out[compare_col].to_numpy())
    out['ABAC'] /= list(range(1,out.shape[0]+1))
    save_csv(out,arg_file = arg_file,**kwargs) 
    f_z = kwargs.get('predictor',RF_proba_predictor)(**kwargs)
    mean_classifier_score_fun(kwargs["fz"],f_z)      
    return out

def get_cross_validation_train_ARA(kwargs = None):
    kwargs = get_calibrated_set_ARA(kwargs)
    cross_validation_train_ARA= load_csv('cross_validation_train_ARA_file_name',**kwargs)['data']
    if cross_validation_train_ARA.empty:
        cross_validation_train_ARA = get_cross_validation('cross_validation_train_ARA_file_name','TOP_ARA_True','TOP_ARA_True_ref',kwargs)
    kwargs['cross_validation_train_ARA']=cross_validation_train_ARA
    return kwargs
def get_cross_validation_train_OPPO(kwargs = None):
    kwargs = get_calibrated_set_OPPO(kwargs)
    cross_validation_train_OPPO= load_csv('cross_validation_train_OPPO_file_name',**kwargs)['data']
    if cross_validation_train_OPPO.empty:
        cross_validation_train_OPPO = get_cross_validation('cross_validation_train_OPPO_file_name','AFFECTATION_ARA sans OPPO','AFFECTATION_ARA sans OPPO_ref',kwargs)
    kwargs['cross_validation_train_OPPO']=cross_validation_train_OPPO
    return kwargs

def get_cross_validation_test_ARA(kwargs = None):
    kwargs = get_crossARA_test_set(kwargs)
    cross_validation_test_ARA= load_csv('cross_validation_test_ARA_file_name',**kwargs)['data']
    if cross_validation_test_ARA.empty:
        cross_validation_test_ARA = get_cross_validation('cross_validation_test_ARA_file_name','TOP_ARA_True','TOP_ARA_True_ref',kwargs)
    kwargs['cross_validation_test_ARA']=cross_validation_test_ARA
    return kwargs
def get_cross_validation_test_OPPO(kwargs = None):
    kwargs = get_crossOPPO_test_set(kwargs)
    cross_validation_test_OPPO= load_csv('cross_validation_test_OPPO_file_name',**kwargs)['data']
    if cross_validation_test_OPPO.empty:
        cross_validation_test_OPPO = get_cross_validation('cross_validation_test_OPPO_file_name','AFFECTATION_ARA sans OPPO','AFFECTATION_ARA sans OPPO_ref',kwargs)
    kwargs['cross_validation_test_OPPO']=cross_validation_test_OPPO
    return kwargs


def get_cross_validationARA(kwargs = None): return get_cross_validation_test_ARA(get_cross_validation_train_ARA(kwargs))
def get_cross_validationOPPO(kwargs = None): return get_cross_validation_test_OPPO(get_cross_validation_train_OPPO(kwargs))

def get_crossARA_training_set(kwargs = None):
    if kwargs is None: kwargs = get_crossARA_params(get_cebpl_params())
    else:kwargs = {**kwargs,**get_crossARA_params(get_cebpl_params())}

    z,fz= load_csv(['training_z_csv','training_fz_csv'],**kwargs)['data']
    if True in [z.empty,fz.empty]:
        kwargs = {**kwargs,**get_raw_ARA_set()}
        z,fz,test_z,test_fz = balanced_classifier_train_test_split(shuffle= True,**kwargs)
        save_csv(z,arg_file ='training_z_csv',compression="gzip",**kwargs)
        save_csv(fz,arg_file ='training_fz_csv',**kwargs)
        save_csv(test_z,arg_file ='test_z_csv',compression="gzip",**kwargs)
        save_csv(test_fz,arg_file ='test_fz_csv',**kwargs)
    kwargs["z"],kwargs["fz"] = z,fz
    test = kwargs['variables_selector']
    return get_variable_selector(kwargs)


def get_crossOPPO_training_set(kwargs = None):
    if kwargs is None: kwargs = get_crossOPPO_params(get_cebpl_params())
    else:kwargs = {**kwargs,**get_crossOPPO_params(get_cebpl_params())}
    z,fz= load_csv(['training_z_csv','training_fz_csv'],**kwargs)['data']
    if True in [z.empty,fz.empty]:
        kwargs = {**kwargs,**get_raw_OPPO_set()}
        z,fz,test_z,test_fz = balanced_classifier_train_test_split(shuffle= True,**kwargs)
        save_csv(z,arg_file ='training_z_csv',compression="gzip",**kwargs)
        save_csv(fz,arg_file ='training_fz_csv',**kwargs)
        save_csv(test_z,arg_file ='test_z_csv',compression="gzip",**kwargs)
        save_csv(test_fz,arg_file ='test_fz_csv',**kwargs)
    kwargs["z"],kwargs["fz"] = z,fz
    return get_variable_selector(kwargs)


def get_crossARA_test_set(kwargs = None):
    if kwargs is None: kwargs = get_crossARA_params()
    else: kwargs = {**kwargs,**get_crossARA_params()}
    test_z,test_fz= load_csv(['test_z_csv','test_fz_csv'],**kwargs)['data']
    if True in [test_z.empty,test_fz.empty]:
        kwargs = get_crossARA_training_set(kwargs)
        test_z,test_fz= load_csv(['test_z_csv','test_fz_csv'],**kwargs)['data']
    kwargs["z"],kwargs["fz"] = test_z,test_fz
    return get_calibrated_set_ARA(kwargs)

def get_crossOPPO_test_set(kwargs = None):
    if kwargs is None: kwargs = get_crossOPPO_params()
    else: kwargs = {**kwargs,**get_crossOPPO_params()}
    test_z,test_fz= load_csv(['test_z_csv','test_fz_csv'],**kwargs)['data']
    if True in [test_z.empty,test_fz.empty]:
        kwargs = get_crossOPPO_training_set(kwargs)
        test_z,test_fz= load_csv(['test_z_csv','test_fz_csv'],**kwargs)['data']
    kwargs["z"],kwargs["fz"] = test_z,test_fz
    return get_calibrated_set_OPPO(kwargs)


def get_trivial_training_set(kwargs = None):
    if kwargs is None: kwargs = get_crossARA_training_set()
    kwargs['x'],kwargs['fx'] = kwargs['z'],kwargs['fz']
    return kwargs

def errorARA_fun(kwargs): 
    out= pd.DataFrame.abs(kwargs['fz']['TOP_ARA_True']-kwargs['f_z']['TOP_ARA_True']).sort_values(ascending = False)
    return out

def get_calibrated_set_ARA(kwargs = None):
    if kwargs is None: kwargs = get_crossARA_params()
    else:kwargs = {**kwargs,**get_crossARA_params()}
    x,fx,y= load_csv(['training_x_csv','training_fx_csv','training_y_csv'],**kwargs)['data']
    kwargs['x'],kwargs['fx'],kwargs['y'],kwargs['error_fun'] = x,fx,y,errorARA_fun
    get_calibrated_training_set = kwargs.get('get_calibrated_training_set',codpy_rl_classifier)
    if True in [x.empty,fx.empty]:
        kwargs = get_crossARA_training_set(kwargs)
        kwargs = get_calibrated_training_set(kwargs)
        save_csv(kwargs['x'],arg_file ='training_x_csv',**kwargs)
        save_csv(kwargs['fx'],arg_file ='training_fx_csv',**kwargs)
        save_csv(kwargs['y'],arg_file ='training_y_csv',**kwargs)
    return get_variable_selector(kwargs)

def get_calibrated_set_OPPO(kwargs = None):
    if kwargs is None: kwargs = get_crossOPPO_params()
    else:kwargs = {**kwargs,**get_crossOPPO_params()}
    x,fx,y= load_csv(['training_x_csv','training_fx_csv','training_y_csv'],**kwargs)['data']
    kwargs['x'],kwargs['fx'],kwargs['y'] = x,fx,y
    get_calibrated_training_set = kwargs.get('get_calibrated_training_set',RF_classifier_predictor)
    if True in [x.empty,fx.empty]:
        kwargs = {**kwargs,**get_crossOPPO_training_set()}
        kwargs = get_calibrated_training_set(kwargs)
        save_csv(kwargs['x'],arg_file ='training_x_csv',**kwargs)
        save_csv(kwargs['fx'],arg_file ='training_fx_csv',**kwargs)
        save_csv(kwargs['y'],arg_file ='training_y_csv',**kwargs)
    return get_variable_selector(kwargs)


def mean_classifier_score_fun(fz, f_z) :
    fz,f_z = softmaxindice(mat = fz),softmaxindice(mat = f_z)
    out = metrics.confusion_matrix(fz,f_z)
    averages = np.zeros([out.shape[0]])
    for n in range(out.shape[0]):
            averages[n] = out[n,n] / np.sum(out[n,:])   
    print(out)
    print("averages:",averages)
    return 1.-np.mean(averages)
def mean_classifier_score_wrapper(**kwargs) : return mean_classifier_score_fun(kwargs['fz'], kwargs['f_z'])

def trivial_variable_selector(**params):
    kwargs = params.copy()
    z = kwargs['z']
    variable_selector_csv = kwargs.get('variables_selector_csv',[])
    matching_cols = kwargs.get('selector_cols',[])

    cols_drop = params.get('cols_drop',[])
    cols_drop = set(cols_drop) | set(select_constant_columns(z))
    params['cols_drop'] = list(cols_drop)

    z = column_selector(z,**params)
    xyzcolumns = list(z.columns)
    output = { 'keep_columns' : xyzcolumns}
    
    if len(variable_selector_csv): 
        pd.DataFrame(output).to_csv(variable_selector_csv,sep = ',')
    return output['keep_columns']


def kbest_variable_selector(**params):

    def kbest(x_train, y_train):
        k = 160
        y_train = softmaxindice(mat = y_train)
        selector = SelectKBest(f_classif, k=k)
        selector.fit(x_train, y_train)
        return selector.get_feature_names_out(list(x_train.columns))

    kwargs = params.copy()
    z,fz = kwargs['z'],kwargs['fz']
    cols_drop = params.get('cols_drop',[])
    cols_drop = set(cols_drop) | set(select_constant_columns(z))
    params['cols_drop'] = list(cols_drop)

    z = column_selector(z,**params)
    kbest_columns = kbest(z, fz)
    xyzcolumns = list(z.columns)
    output = { 'keep_columns' : kbest_columns}
    
    variable_selector_csv = kwargs.get('variables_selector_csv',[])
    if len(variable_selector_csv): 
        pd.DataFrame(output).to_csv(variable_selector_csv,sep = ',')
    return output['keep_columns']


def get_variable_selector(kwargs = None):
    # return kwargs
    if kwargs is None: kwargs = get_crossARA_validation_set()
    predictor = kwargs.get('predictor',RF_proba_predictor)
    kwargs['predictor'] = kwargs.get('proba_predictor',RF_proba_predictor)
    variable_selector_csv = kwargs.get('variables_selector_csv',[])
    cols_keep = kwargs.get('cols_keep',[])
    variable_selector_ = kwargs.get('variables_selector',variable_selector)
    if len(variable_selector_csv) == 0 or not os.path.exists(variable_selector_csv):
        # kwargs['error_fun'] = mean_classifier_score_fun
        cols_keep = set(variable_selector_(**kwargs))
    else: 
        cols_keep = set(read_csv(variable_selector_csv,sep = ",")['keep_columns'].to_list())
    kwargs['cols_keep'] = list(cols_keep | set(kwargs.get('cols_keep',[])))
    kwargs['predictor'] = predictor
    return kwargs

def selector_cross_validation(kwargs = None):
    if kwargs is None: kwargs = get_variable_selector()
    output(kwargs)

def get_cebpl_ARA_params():
    file_name = datetime.date.today().strftime('%Y%m%d')+"Flux_retour_IA_MPG_CEBPL_V23.csv"
    cross_validation_dic = get_crossARA_params()
    cross_validation_dic['test_f_z_csv'] = os.path.join(cebpl_path,"data","output",file_name)
    return cross_validation_dic

def get_cebpl_OPPO_params():
    file_name = datetime.date.today().strftime('%Y%m%d')+"Flux_retour_IA_MPG_CEBPL_V23.csv"
    cross_validation_dic = get_crossOPPO_params()
    cross_validation_dic['test_f_z_csv'] = os.path.join(cebpl_path,"data","output",file_name)
    return cross_validation_dic

def cebpl_output(kwargs = None):
    def fun(o,z,sortcol,cross_df,col):
        def fun(i):
            return cross_df.loc[(cross_df[col]-i).abs().argsort()[0]]['ABAC']
        test = list(o[sortcol])
        o['ABAC_'+col] = [fun(i) for i in test]

    if kwargs is None : kwargs = cebpl_predict_OPPO()
    else: kwargs = {**kwargs,**cebpl_predict_OPPO()}
    outOPPO = kwargs["f_z"]
    cross_validation_test_OPPO = get_cross_validation_test_OPPO(kwargs)['cross_validation_test_OPPO']
    sortcol = 'AFFECTATION_ARA sans OPPO'
    if isinstance(outOPPO,list):[fun(o,z,sortcol,cross_validation_test_OPPO,'AFFECTATION_ARA sans OPPO') for o,z in zip(outOPPO,kwargs['z'])]
    else:fun(outOPPO,kwargs['z'],sortcol,cross_validation_test_OPPO,'AFFECTATION_ARA sans OPPO')


    if kwargs is None : kwargs = cebpl_predict_ARA()
    else: kwargs = {**kwargs,**cebpl_predict_ARA()}
    outARA = kwargs["f_z"]
    cross_validation_test_ARA = get_cross_validation_test_ARA(kwargs)['cross_validation_test_ARA']
    sortcol = 'TOP_ARA_True'
    if isinstance(outARA,list):[fun(o,z,sortcol,cross_validation_test_ARA,'TOP_ARA_True') for o,z in zip(outARA,kwargs['z'])]
    else:fun(outARA,kwargs['z'],sortcol,cross_validation_test_ARA,'TOP_ARA_True')

    if isinstance(outOPPO,list):out = [pd.merge(l,r,left_index=True,right_index=True) for l,r in zip(outARA,outOPPO)]
    else: out = pd.merge(outARA,outOPPO,left_index=True,right_index=True)

    if isinstance(outOPPO,list):[o.sort_values(by = sortcol, ascending = False,inplace = True) for o in out]
    else: out.sort_values(by = sortcol, ascending = False,inplace = True)

    def fun(o,z):o['ID_PERS'] = z['ID_PERS']
    if isinstance(out,list):[fun(o,z) for o,z in zip(out,kwargs['z'])]
    else: out['ID_PERS'] = kwargs['z']['ID_PERS']
        
    def fun(x):
        x.drop(['TOP_ARA_False','AFFECTATION_ARA avec OPPO'], axis = 1,inplace=True)
        x.drop_duplicates('ID_PERS',keep='first',inplace=True)
    if isinstance(out,list):[fun(x) for x in out]
    else:fun(out)

    to_csv(out,file_path = kwargs['filelistout'],**kwargs) 
    return kwargs



def setup_cebpl(kwargs = None):
    if kwargs is None : kwargs = get_cebpl_params()
    # else: kwargs = {**kwargs,**get_cebpl_params()}
    kwargs = download_cebpl_test_set(kwargs)
    kwargs['data_path'] = kwargs['filelistin']
    kwargs = load_csv('data_path', **kwargs)
    dict_ = {}
    if isinstance(kwargs['data_path'],list):
        def fun(x,y):dict_[x] = y
        [ fun(x,y) for x,y in zip(kwargs['filelistin'],kwargs['filelistDATA']) ]
    else: 
        dict_[kwargs['filelistin']] = kwargs['filelistDATA']
    kwargs['transformed_data_path'] = dict_

    def fun(funx,**kwargs):
        print("number of lines in raw data:", funx.shape[0])
        print("number of columns in raw data:", funx.shape[1])
    if isinstance(kwargs['data'],list): 
        [fun(x,**kwargs) for x in kwargs['data']]
    else: fun(kwargs['data'],**kwargs)
    kwargs = cebpl_data_transformer.get_transformed_data(kwargs)
    def fun(zfun,kwargs):
        if isinstance(zfun,list):return [fun(z,kwargs) for z in zfun]
        x = kwargs["x"]
        columnsx = list(x.columns)
        columnsz = list(zfun.columns)
        shape = list(x.shape)
        shape[0] = zfun.shape[0]
        out = pd.DataFrame(np.zeros(shape),columns = columnsx)
        columnin = [v for v in columnsz if v in columnsx]
        trash = set(columnsz).difference(set(columnin))
        print("trash columns on new data:",trash)
        out[columnin] = zfun[columnin]
        return out
    kwargs['z'] = fun(kwargs['transformed_data'],kwargs)
    return kwargs


def cebpl_predict(kwargs):
    kwargs = setup_cebpl(kwargs)
    kwargs['f_z'] = kwargs['proba_predictor'](**kwargs)
    return kwargs
    z = kwargs['z']
    kwargs = setup_cebpl(kwargs)
    if isinstance(kwargs['z'],list): 
        lens        = [y.shape[0] for y in kwargs['z']]
        kwargs['z'] = [pd.concat([y,z]) for y in kwargs['z']]
    else:
        lens = kwargs['z'].shape[0]
        kwargs['z'] = pd.concat([kwargs['z'],z])

    kwargs['f_z'] = kwargs['proba_predictor'](**kwargs)
    if isinstance(kwargs['z'],list): 
        kwargs['z'],kwargs['f_z'] = [y.iloc[0:l].reset_index(drop = True) for y,l in zip(kwargs['z'],lens)],[y.iloc[0:l].reset_index(drop = True) for y,l in zip(kwargs['f_z'],lens)]
    else:
        kwargs['z'],kwargs['f_z'] = kwargs['z'].iloc[0:lens].reset_index(drop = True),kwargs['f_z'].iloc[0:lens].reset_index(drop = True)
    return kwargs

def cebpl_predict_ARA(kwargs = None):
    if kwargs is None : kwargs = get_crossARA_test_set()
    else: kwargs = {**kwargs,**get_crossARA_test_set()}
    return cebpl_predict(kwargs)

def cebpl_predict_OPPO(kwargs = None):
    if kwargs is None : kwargs = get_crossOPPO_test_set()
    else: kwargs = {**kwargs,**get_crossOPPO_test_set()}
    return cebpl_predict(kwargs)


def read_file(**kwargs):
    kwargs = get_cebpl_params()
    kwargs['data_path'] = os.path.join(cebpl_path,"data","data_test-transformed.gz")
    datum = load_csv('data_path', **kwargs)['data'].columns
    #data = datum['data'].dropna()#['DATE_FIN_INCD_MAJR']#.apply(days_transform)
    pass

def get_data_set(kwargs = None):
    if kwargs is None: kwargs = get_crossOPPO_params()
    else: kwargs = {**kwargs,**get_crossOPPO_params()}
    x,fx= load_csv(['raw_x_csv','raw_fx_csv'],**kwargs)['data']
    z,fz= load_csv(['raw_z_csv','raw_fz_csv'],**kwargs)['data']
    # if True in [test_z.empty,test_fz.empty]:
    #     kwargs = get_crossOPPO_training_set(kwargs)
    #     test_z,test_fz= load_csv(['test_z_csv','test_fz_csv'],**kwargs)['data']
    kwargs["z"],kwargs["fz"] = z,fz
    kwargs["x"],kwargs["fx"] = x,fx
    return  kwargs #get_calibrated_set_OPPO(kwargs)



##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
######################## Database updater with CEBPL analysis ##########################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
def days_to_date(days):
    start_date = datetime.datetime(1, 1, 1)
    date = start_date + datetime.timedelta(days=days+7)
    return date.strftime('%d/%m/%Y')

def download_cebpl_analysis(kwargs):
    ESI_Path_ALLER = kwargs["params_ESI"]["ESI_Path_RETOUR_ANALYSES"]
    Path_in_a = kwargs['CEBPL_update_params']['source_analysis_path']
    FT = FTP_connector(**kwargs)
    for n in ESI_Path_ALLER: FT.ftps.cwd(n)
    out = []
    files = FT.ftps.nlst()
    files_to_remove = ['BASE_APPRENTISSAGE_SORTIE.csv', 'BASE_APPRENTISSAGE_SORTIE_V2.csv', 'BASE_APPRENTISSAGE_SORTIE_V3.csv', '&Date_analy_inst.-Flux_retour_IA_MPG_CEBPL_Analyse.csv']
    files = [element for element in files if element not in files_to_remove]
    print(FT.ftps.dir())
    for file in files:
        local_file  = os.path.join(Path_in_a, file)
        if not os.path.exists(local_file):
            FT.ftps.retrbinary("RETR "+file,open(local_file, 'wb').write)
            print("Download status:success", file)
        out.append(local_file)
    FT.ftps.close()
    pass

class DataUpdater:
    def __init__(self, kwargs):
        params = kwargs['CEBPL_update_params']
        self.data_path = params['data_path']
        self.id_file_path = params['id_file_path']
        self.merged_data_path = params['merged_data_path']
        self.merged_analysis_path = params['merged_analysis_path']
        self.pkl_processed =  params['pkl_processed']
        self.pkl_analysis_processed = params['pkl_analysis_processed']
        self.archive_path = params['archive_path']
        self.source_analysis_path = params['source_analysis_path']
        self.update_folder_path = params['update_folder_path']
        self.encoding = 'UTF-8'
        download_cebpl_analysis(kwargs)
    def get_encoding(self, file):
        with open(file, 'rb') as f:
            result = chardet.detect(f.read())
        return result['encoding']

    def update_data(self, source = None, extension='*.old', update_path='merged.csv', pkl_file='processed_files.pkl', **kwargs):
        condition = kwargs.get('condition', None)
        capital_letters = kwargs.get('capital_letters', None)
        try:
            with open(pkl_file, 'rb') as f:
                processed_files = pickle.load(f)
        except FileNotFoundError:
            processed_files = []
        all_files = glob.glob(os.path.join(source, extension))
        if condition:
            all_files = condition(all_files)
        new_files = list(set(all_files) - set(processed_files))
        if new_files:
            encoding = self.get_encoding(new_files[0])
            try:
                new_dfs = [pd.read_csv(file, encoding=encoding) for file in new_files]
            except UnicodeDecodeError:
                new_dfs = [pd.read_csv(file, encoding='ISO-8859-1') for file in new_files]
            if capital_letters:
                for df in new_dfs:
                    df.columns = df.columns.str.upper()
            new_data = pd.concat(new_dfs, ignore_index=True)

            try:
                merged_df = pd.read_csv(update_path)
                merged_df = pd.concat([merged_df, new_data], ignore_index=True)
                if 'merged.csv' in self.merged_data_path:
                    merged_df['AFFECTATION'] = 'AGENCE'
            except FileNotFoundError:
                merged_df = new_data
                if 'merged.csv' in self.merged_data_path:
                    merged_df['AFFECTATION'] = 'AGENCE'
            merged_df.to_csv(update_path, index=False)
            processed_files += new_files
            with open(pkl_file, 'wb') as f:
                pickle.dump(processed_files, f)

    def process_id(self):
        def get_new_ids(df, processed_ids):
            client_ids = df.loc[(df['Analyse_ARA'] == 'Agence') & (df['ID_Client'] != 0), 'ID_Client']
            new_ids = client_ids.loc[~client_ids.isin(processed_ids)]
            new_ids = new_ids.to_frame('ID_Client')
            return new_ids

        df = pd.read_csv(self.merged_analysis_path, encoding=self.encoding)
        try:
            processed_ids = pd.read_csv(self.id_file_path, squeeze=True, encoding=self.encoding)
        except FileNotFoundError:
            processed_ids = pd.DataFrame(columns=['ID_Client'])
        new_ids = get_new_ids(df, processed_ids)
        if not new_ids.empty:
            new_ids = pd.concat([processed_ids, new_ids])
            new_ids.to_csv(self.id_file_path, index=False, encoding=self.encoding)

    def extract_lines_IDs(self):
        '''
        The method allows :
            to rename lines in the Analyse_ARA column
            remove the lines that have 0 in the Analyse_ARA column
            rename column names
            Identify clients in merged file and analysis update file and merge into 
            unified dataframe
        '''
        df = pd.read_csv(os.path.join(self.update_folder_path,'analysis_merged.csv'), encoding=self.encoding)
        df_update = df[['ID_Client', 'Analyse_ARA']]
        # Rename the entries in the column 'Analyse_ARA'
        df_update.loc[:, 'Analyse_ARA'] = df_update['Analyse_ARA'].replace({
            'Ara avec Oppo 102': 'ARA avec OPPO',
            'Ara sans Oppo 102': 'ARA sans OPPO',
            'Agence': 'AGENCE'
        })
        # Drop the rows where 'Analyse_ARA' is 0
        df_update = df_update[df_update['Analyse_ARA'].astype(str) != '0']
        # Rename the columns
        df_update = df_update.rename(columns={
            'ID_Client': 'ID_PERS',
            'Analyse_ARA': 'AFFECTATION'
        })

        merged_df = pd.read_csv(self.merged_data_path, encoding=self.encoding)
        df2 = merged_df[merged_df['ID_PERS'].isin(df_update['ID_PERS'])]
        df2 = pd.merge(df2, df_update, left_on='ID_PERS', right_on='ID_PERS', how='left')
        df2.to_csv(os.path.join(self.update_folder_path,'update_agence.csv'), index=False)
        pass

    def merge_data(self):
        self.update_data(source = self.archive_path, extension='*.old', update_path=self.merged_data_path, pkl_file=self.pkl_processed, 
                         **{'capital_letters': True})
    def merge_analysis(self):
        condition = lambda file_paths:  [path for path in file_paths if 'Flux_retour_IA_MPG_CEBPL_Analyse' in path]
        self.update_data(source = self.source_analysis_path, extension='*.csv', update_path=self.merged_analysis_path, 
                         pkl_file= self.pkl_analysis_processed, **{'condition': condition})
 
    def latest_merge_date(self):
        # Check for all files starting with 'merged_' prefix
        merged_files = [f for f in os.listdir(self.update_folder_path) if f.startswith('merged_')]
        if not merged_files:
            return None
        latest_file = max(merged_files, key=lambda s: s.split('_')[1].split('.csv')[0])
        date_str = latest_file.split('_')[1].split('.csv')[0]
        return datetime.datetime.strptime(date_str, '%Y%m%d').date()

    def latest_downloaded_file_date(self, file_list):
        file_list = [f for f in file_list if not '_Fichier contre analyse.csv' in f]
        #file_list.remove('C:\\Users\\shohr\\Desktop\\Github\\CodpyCEBPL\\cebpl-dev\\ESIMPG-DATA\\analysis\\20230928_Fichier contre analyse.csv')
        date_list = [datetime.datetime.strptime(os.path.basename(f).split('-')[0], '%Y%m%d').date() for f in file_list]
        return max(date_list)

    def run(self):
        extension='*.old'
        try:
            with open(self.pkl_processed, 'rb') as f:
                processed_files = pickle.load(f)
        except FileNotFoundError:
            processed_files = []
        try:
            with open(self.pkl_analysis_processed, 'rb') as f:
                analysis_processed_files = pickle.load(f)
        except FileNotFoundError:
            analysis_processed_files = []
        all_files = glob.glob(os.path.join(self.archive_path, extension))
        new_files = list(set(all_files) - set(processed_files))
        all_analysis_processed_files = glob.glob(os.path.join(self.source_analysis_path, '*.csv'))
        new_analysis_processed_files = list(set(all_analysis_processed_files) - set(analysis_processed_files))
        if len(new_files) == 0 and len(new_analysis_processed_files) == 0:
            return
        else:
            analysis_files = glob.glob(os.path.join(self.source_analysis_path,'*.csv'))
            last_merged_date = self.latest_merge_date()
            latest_downloaded_date = self.latest_downloaded_file_date(analysis_files)
            
            # Only proceed if the downloaded or merged files have a newer date
            if not last_merged_date or latest_downloaded_date > last_merged_date:
                if last_merged_date == None:
                    self.merged_data_path = os.path.join(self.update_folder_path, f"merged_{latest_downloaded_date.strftime('%Y%m%d')}.csv")
                else:
                    self.merged_data_path = os.path.join(self.update_folder_path, f"merged_{last_merged_date.strftime('%Y%m%d')}.csv")
                self.merge_data() #merge *.old data
                self.merge_analysis()
                self.process_id()
                self.extract_lines_IDs()
            else:
                print(f"No new files to process. Latest merged data is from {last_merged_date}.")


class global_merge():
    def __init__(self, **kwargs) -> None:
        self.base_file = kwargs.get('base_file', 'cebpl_data')
        self.data_path = kwargs['data_path']
        self.encoding ='utf-8'
        self.update_folder_path = kwargs['CEBPL_update_params']['update_folder_path']
        self.update_agence_path = os.path.join(self.update_folder_path,'update_agence.csv')

    def _latest_file_date(self, folder_path):
        # The expected pattern for files with dates
        pattern_with_date = self.base_file
        
        # Filtering of the directory for files that match the pattern
        files = [f for f in os.listdir(folder_path) if f.startswith(pattern_with_date) and f.endswith('.csv')]
        
        # Sorting by the actual date and stack the latest, if files with date are found
        try:
            if files:
            # Extraction of  dates from the filenames, sort them, and get the latest
                files.sort(key=lambda f: datetime.datetime.strptime(f.split('_')[-1].replace('.csv', ''), '%Y%m%d'))
                date_str = files[-1].split('_')[-1].replace('.csv', '')
                return datetime.datetime.strptime(date_str, '%Y%m%d').date()
        except ValueError:
            pass
        
        pattern_with_nodate = self.base_file.rstrip('_')
        
        files = [f for f in os.listdir(folder_path) if f.startswith(pattern_with_nodate) and f.endswith('.csv')]
        # If no dated files are found, checking for 'cebpl_short.csv'
        if self.base_file.rstrip('_') + '.csv' in files:
            return ''
        # If neither condition is met, return None
        return ''

    def can_merge(self, pkl_file, days_since_last_merge, folder_path):
        pkl_date = None
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as file:
                try:
                    pkl_date = pickle.load(file)
                except EOFError:
                    pass

        file_date = self._latest_file_date(folder_path)

        # If both dates don't exist, merge can proceed
        if not pkl_date and not file_date:
            return True

        # If one of the dates doesn't exist, use the available one
        last_merge_date = pkl_date or file_date

        current_date = datetime.datetime.now().date()
        days_passed = (current_date - last_merge_date).days
        return days_passed > days_since_last_merge
    @staticmethod
    def correct_encoding(s):
        return s.replace('PrÃªt', 'Prêt')
    def merge_files(self, file1, file2, folder_path):
        df1 = pd.read_csv(file1, encoding=self.encoding, usecols=lambda column: column not in ['Unnamed: 0'])  # Here we exclude the 'Unnamed: 0' column which is typically the index column from pandas saved CSVs
        df2 = pd.read_csv(file2, encoding=self.encoding)

        # Merge keys from original dataset
        merge_keys = ['ID_PERS', 'MT_CRNC_INCD_RISQ_MAJR', 'MT_TRSR_MOYE_MOIS', 'EVOL_SURF_FIN']
        df1_keys = df1[merge_keys]
        
        # Get the indices of df2 that have matching rows in df1 based on merge keys
        indices_to_remove = df2.merge(df1_keys, on=merge_keys, how='inner', indicator=True).query('_merge == "both"').index
        
        indices_to_remove = [idx for idx in indices_to_remove if idx < len(df2)] # Drop those indices from df2
        df2 = df2.drop(indices_to_remove)

        combined = pd.concat([df1, df2]).reset_index(drop=True)  # Reset index to ensure continuous numbering
        combined['TYPE_INCIDENT'] = combined['TYPE_INCIDENT'].apply(self.correct_encoding)
        new_file_name = f"{self.base_file}_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
        save_path = os.path.join(folder_path, new_file_name)
        combined.to_csv(save_path, index=True, encoding=self.encoding)  # We set index=True to save with continuous numbering

        # Remove the original file after updating
        if os.path.exists(file1) and file1 != save_path:
            os.remove(file1)
        return save_path

    def update_last_merge_date(self, pkl_file):
        with open(pkl_file, 'wb') as file:
            pickle.dump(datetime.datetime.now().date(), file)
    
    def run(self, **kwargs):
        params = kwargs.get('CEBPL_update_params', None)
        pkl_file = params.get('last_merged_dates', None)
        D = params.get('days', 1)
        folder_path = params.get('data_folder_path', None)
        original_file_path = os.path.join(folder_path, f"{self.base_file}_{self._latest_file_date(folder_path)}.csv")
        
        latest_file_date = self._latest_file_date(folder_path)
        if latest_file_date:
            # If a date was found in the filenames, set the original_file_path to that dated file
            original_file_path = os.path.join(folder_path, f"{self.base_file}_{latest_file_date.strftime('%Y%m%d')}.csv")
        else:
            # If no dated file was found, check for the base file
            if os.path.exists(self.data_path):
                original_file_path = self.data_path
            else:
                # If neither a dated file nor the base file exists, raise a warning and set a new filename with the current date
                print("Warning: No existing CSV with the latest date found. Merging with new file.")
                original_file_path = os.path.join(folder_path, f"{self.base_file}_{datetime.datetime.now().strftime('%Y%m%d')}.csv")

        if self.can_merge(pkl_file, D, folder_path):
            self.merge_files(original_file_path, self.update_agence_path, folder_path)
            self.update_last_merge_date(pkl_file)
        else:
            print(f"At least {D} days must pass since the last merge.")



if __name__ == "__main__":
    # get_cross_validation_train_ARA()
    data_updater = DataUpdater(get_cebpl_params()).run()
    global_merge(**get_cebpl_params()).run(**get_cebpl_params())
    cebpl_output()
    # upload_ESI(**get_cebpl_params())
    # get_cross_validationARA()
    # cebpl_output(cebpl_predict_ARA(cebpl_predict_OPPO(get_cebpl_params())))
    # get_cross_validationOPPO()
    # cebpl_output(cebpl_predict_ARA(cebpl_predict_OPPO()))
    pass
    # out = cebpl_files_update(**get_cebpl_paramss())
