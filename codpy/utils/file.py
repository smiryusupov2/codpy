import os, sys
import pandas as pd
import errno

################file utilities######################################
save_to_file_switchDict = { pd.DataFrame: lambda x,file_name,**kwargs :  x.to_csv(file_name,**kwargs),
                            list: lambda x,file_name,**kwargs :  [save_to_file(y,f,**kwargs) for y,f in zip(x,file_name)]
                    }

def save_to_file(x,file_name,**kwargs):
    type_debug = type(x)
    method = save_to_file_switchDict.get(type_debug,lambda x,file_name,**kwargs: save_to_file(pd.DataFrame(x),file_name,**kwargs))
    return method(x,file_name,**kwargs)


def find_helper(pathname, matchFunc = os.path.isfile):
    for dir in sys.path:
        candidate = os.path.join(dir,pathname)
        # print(candidate)
        if matchFunc(candidate):
            return candidate
    raise FileNotFoundError(errno.ENOENT,os.strerror(errno.ENOENT),pathname)

def find_dir(pathname):
    return find_helper(pathname, matchFunc = os.path.isdir)

def find_file(pathname):
    return find_helper(pathname)

def files_indir(dirname,extension=".png"):
    out=[]
    for root, directories, files in os.walk(dirname):
        for file in files:
            if not len(extension):out.append(os.path.join(root,file))
            else: 
                if(file.endswith(extension)):
                    out.append(os.path.join(root,file))
    return out