from preamble import *
import pathlib

def add_suffix(suffix = None,file_names="",path=None):
    if isinstance(file_names,list):return [add_suffix(file_names=file_name,suffix=suffix,path=path) for file_name in file_names]
    base_name = os.path.basename(file_names)
    if path is None: path = os.path.dirname(os.path.realpath(file_names))
    if suffix is not None:
        split_tup = os.path.splitext(base_name)
        return os.path.join(path,split_tup[0] + suffix + split_tup[1])
    return os.path.join(path,base_name)
def replace_string(strings,first,second):
    if isinstance(strings,list):return [replace_string(string,first,second) for string in strings]
    return strings.replace(first,second)


def get_ESI_params(**ESI_params):
    ESI_Path_ALLER,ESI_Path_RETOUR, ESI_DIR = ESI_params.get('ESI_Path_ALLER'),ESI_params.get('ESI_Path_RETOUR'),ESI_params.get('ESI_DIR')
    Path_IN,Path_TRANS, Path_OUT,Path_ARCHIVE = ESI_params.get('Path_IN'),ESI_params.get('Path_TRANS'),ESI_params.get('Path_OUT'),ESI_params.get('Path_ARCHIVE')
    suffix_ALLER, suffix_RETOUR,n_files_host, clean_files= ESI_params.get('suffix_ALLER'),ESI_params.get('suffix_RETOUR'),ESI_params.get('n_files_host'),ESI_params.get('clean_files')
    return ESI_Path_ALLER,ESI_Path_RETOUR, ESI_DIR, Path_IN, Path_TRANS,Path_OUT, Path_ARCHIVE, suffix_ALLER, suffix_RETOUR,n_files_host, clean_files



def download_cebpl_test_set(kwargs):
    import datetime,time
    from os import listdir
    from os.path import isfile, join

    params_FTP = kwargs.get('params_FTP')
    params_ESI = kwargs.get('params_ESI')
    ESI_Path_ALLER,ESI_Path_RETOUR, ESI_DIR, Path_IN, Path_TRANS,Path_OUT, Path_ARCHIVE, suffix_ALLER, suffix_RETOUR,n_files_host, clean_files = get_ESI_params(**params_ESI)

    if not os.path.exists(ESI_DIR): os.makedirs(ESI_DIR)
    if not os.path.exists(Path_IN): os.makedirs(Path_IN)
    if not os.path.exists(Path_TRANS): os.makedirs(Path_TRANS)
    if not os.path.exists(Path_OUT): os.makedirs(Path_OUT)

    FT = FTP_connector(**kwargs)
    for n in ESI_Path_ALLER: FT.ftps.cwd(n)
    print(pd.DataFrame(FT.ftps.nlst(), columns=["list of files on host: ALLER"]))
    def filter_in(distant_file,local_file,filelist, **kwargs):
        # test = list(filter(lambda x: distant_file in x,filelist.filein.values))
        out_file = replace_string(add_suffix(file_names=local_file,path = Path_OUT),'aller','retour')
        return  not os.path.exists(local_file) or not os.path.exists(out_file)
    out = []
    files = FT.ftps.nlst()
    filelistin = [join(Path_IN, f) for f in listdir(Path_IN) if isfile(join(Path_IN, f))]
    print(FT.ftps.dir())
    for file in files:
        local_file  = os.path.join(Path_IN, file)
        if filter_in(distant_file = file,local_file=local_file, filelist = filelistin, **kwargs):
            if not os.path.exists(local_file):
                FT.ftps.retrbinary("RETR "+file,open(local_file, 'wb').write)
            out.append(local_file)
            today_date = datetime.date.today()
            print("Download status:success", file)
    FT.ftps.close()

    def fun(file):
        head, tail = os.path.splitext(file)
        return isfile(join(file)) and tail=='.csv'

    kwargs['filelistin'] = [join(Path_IN, f) for f in listdir(Path_IN) if fun(join(Path_IN, f))]
    kwargs['filelistDATA'] = add_suffix("-transformed",kwargs['filelistin'],Path_TRANS)
    kwargs['filelistDATA'] = replace_string(kwargs['filelistDATA'],"csv","gz")
    kwargs['filelistout'] = replace_string(add_suffix(None,kwargs['filelistin'],Path_OUT),'aller','retour')
    return kwargs
    pass

def upload_ESI(**kwargs):
    import datetime,time
    params_FTP = kwargs.get('params_FTP')
    params_ESI = kwargs.get('params_ESI')
    clean_files = kwargs.get('clean_files')
    out = []

    ESI_Path_ALLER,ESI_Path_RETOUR, ESI_DIR, Path_IN, Path_TRANS,Path_OUT,Path_ARCHIVE, suffix_ALLER, suffix_RETOUR,n_files_host, clean_files = get_ESI_params(**params_ESI)
    FT = FTP_connector(**params_FTP)
    for n in ESI_Path_RETOUR: FT.ftps.cwd(n)

    local_files = os.listdir(Path_OUT)
    distant_files = FT.ftps.nlst()
    params_ESI = kwargs.get('params_ESI')

    def filter_out(distant_file,local_file, **kwargs):
        import pathlib
        filename, file_extension = os.path.splitext(local_file)
        today_date = get_float(datetime.date.today())
        n = kwargs.get('remote_suppress_old_files_days', 7)
        local_timestamp = get_float(datetime.datetime.fromtimestamp(pathlib.Path(local_file).stat().st_ctime))
        return today_date - local_timestamp <= n and file_extension==".csv"

    for file in local_files:
        file_name = os.path.join(Path_OUT,file)
        if filter_out(distant_file=file,local_file=file_name, **params_ESI):
            output_name = os.path.basename(file)
            with open(file_name, "rb") as file_:
                FT.ftps.storbinary("STOR " + output_name, file_)
                out.append(file)


    def remote_clean(path,**kwargs):
        today_date = get_float(datetime.date.today())
        n = kwargs.get('remote_suppress_old_files_days', 7)
        for p in path: FT.ftps.cwd(p)
        host_files = FT.ftps.nlst()
        for remoteFile in host_files:
            my_file = [remoteFile]
            def helper(n):my_file.insert(0,n+"/")
            [helper(n) for n in reversed(path)]
            my_file = ''.join(my_file)
            remote_datetime = FT.ftps.voidcmd(r"MDTM " + my_file)
            remote_datetime = remote_datetime[4:-10].strip() 
            remote_timestamp = get_float(get_date(remote_datetime, date_format = '%Y%m%d'))
            if today_date - remote_timestamp >= n:
                print('delete old file :'+ remoteFile)
                FT.ftps.delete(remoteFile)
            pass
    
    def local_clean(path, **kwargs):
        today_date = datetime.date.today()
        n = kwargs.get('local_suppress_old_files_days', 7)
        local_files = os.listdir(path)

        for local_file in local_files:
            my_file = os.path.join(path, local_file)
            local_timestamp = datetime.datetime.fromtimestamp(pathlib.Path(my_file).stat().st_ctime).date()
            date_difference = (today_date - local_timestamp).days
            file_name = os.path.join(path,local_file)
            if date_difference >= n:
                print('delete old file: ' + my_file)
                os.remove(my_file)
            else:
                filename, file_extension = os.path.splitext(file_name)
                if file_extension == ".csv":
                    print('rename file: ' + my_file)
                    os.rename(file_name,filename +".old")


    if clean_files:
        remote_clean(path = ESI_Path_ALLER,remote_suppress_old_files_days = 0)
        remote_clean(path = ESI_Path_RETOUR,remote_suppress_old_files_days = 1)
        local_clean(path = Path_IN,local_suppress_old_files_days = params_ESI.get('local_suppress_old_files_days', 7))
        local_clean(path = Path_OUT,local_suppress_old_files_days = params_ESI.get('local_suppress_old_files_days', 7))

   # filelist.extend(out)


    print("upload status:success")
    print(pd.DataFrame(FT.ftps.nlst(), columns=["list of files on host: RETOUR"]))
    FT.ftps.close()
    # if kwargs.get('save_file_list',True): 
    #     with open(filelistout, 'wb') as f:
    #         pickle.dump(filelist, f)        

    return out    

def get_ESI_transform_params(**kwargs) :
    params_ESI = kwargs.get('params_ESI')
    ESI_Path_ALLER,ESI_Path_RETOUR, filelistin,filelistout, ESI_DIR, Path_IN, Path_OUT, suffix_ALLER, suffix_RETOUR,n_files_host, clean_files = get_ESI_params(**params_ESI)
    kwargs['data_predictor']['variables_selector']['variables_cols_keep']= os.path.join(ESI_DIR,"variables_selector.csv")
    kwargs['data_path']= os.path.join(ESI_DIR,"data.csv")
    kwargs['timegrid_csv'] = os.path.join(ESI_DIR,"time_grid.csv")
    data_csv = kwargs['data_csv']
    training_set = kwargs['data_path']

    kwargs['transformed_data_path'] = add_suffix(file_names=data_csv,path = Path_IN,suffix = "-transformed")
    kwargs['rejected_ids_path'] = add_suffix(file_names=data_csv,path = Path_IN,suffix ="-rejected_ids")
    kwargs['error_data_path'] = add_suffix(file_names=data_csv,path = Path_IN,suffix ="-errors")
    kwargs['raw_x_csv'] =  add_suffix(path = ESI_DIR,suffix ="raw_x.csv")
    kwargs['raw_fx_csv'] =  add_suffix(path = ESI_DIR,suffix ="raw_fx.csv")
    kwargs['raw_z_csv'] =  add_suffix(file_names=data_csv,suffix ="_raw_z",path=Path_IN)
    kwargs['raw_fz_csv'] =  add_suffix(file_names=data_csv,suffix ="_raw_fz",path=Path_IN)
    kwargs['x_csv'] =  add_suffix(path = ESI_DIR,suffix ="x.csv")
    kwargs['fx_csv'] =  add_suffix(path = ESI_DIR,suffix ="fx.csv")
    kwargs['z_csv'] =  add_suffix(file_names=data_csv,suffix ="z",path=Path_IN)
    kwargs['fz_csv'] =  add_suffix(file_names=data_csv,suffix ="fz",path=Path_IN)
    kwargs['f_z_csv'] =  add_suffix(file_names=data_csv,suffix ="f_z",path=Path_IN)
    kwargs['out_path'] = replace_string(add_suffix(file_names=data_csv,path = Path_OUT),'aller','retour')
    return kwargs
    pass


if __name__ == "__main__":
    download_cebpl_test_set()
    cebpl_files_update(**get_cebpl_param())
