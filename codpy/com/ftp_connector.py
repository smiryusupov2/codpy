import os
currentdir = os.path.dirname(os.path.realpath(__file__))
import ftplib
import ssl
from datetime import datetime
import pandas as pd

class FTP_TLSb(ftplib.FTP_TLS):
    def ntransfercmd(self, cmd, rest=None):
        conn, size = ftplib.FTP.ntransfercmd(self, cmd, rest)
        if self._prot_p:
            session = self.sock.session
            if isinstance(self.sock, ssl.SSLSocket):
                    session = self.sock.session
            conn = self.context.wrap_socket(conn,server_hostname=self.host,
            session=session)
        return conn, size

class FTP_connector(FTP_TLSb):
    def __init__(self, **kwargs):
        FTP_HOST = kwargs.get("FTP_HOST", "ftp.esi.caisse-epargne.fr")
        FTP_USER = kwargs.get("FTP_USER", "A44-PR-MPG-PARTNERS1@gce")
        FTP_PASS = kwargs.get("FTP_PASS", "S'q4W<oy")
        FTP_PORT = kwargs.get("FTP_PORT",990)
        self.ftps = FTP_TLSb()
        print(self.ftps.connect(FTP_HOST,FTP_PORT))
        print(self.ftps.login(FTP_USER, FTP_PASS))
        self.ftps.prot_p()

    def FTP_download(self, **kwargs):
        t = datetime.today().strftime('%Y-%m-%d')
        path = kwargs.get('path_download','')
        self.ftps.cwd("Fichiers") 
        print("current path", self.ftps.pwd())    
        self.ftps.cwd("ALLER CEBPL MPG")  
        print("current path", self.ftps.pwd()) 
        savedFileName = path + t + '.csv'
        filename = self.ftps.nlst()[0]
        with open(savedFileName, 'wb') as f:
            self.ftps.retrbinary('RETR ' + filename, f.write)
            self.ftps.quit()
            print("File has been downloaded")
        self.ftps.close()
    def FTP_upload(self, **kwargs):
        path = kwargs.get('path_file','')
        upload_name = kwargs.get('upload_name','')
        t = datetime.today().strftime('%Y-%m-%d')
        upload_name = upload_name + t + '.csv'
        self.ftps.cwd("Fichiers") 
        print("current path", self.ftps.pwd())    
        self.ftps.cwd("RETOUR MPG CEBPL")  
        print("current path", self.ftps.pwd())
        with open(path, "rb") as file:
            self.ftps.storbinary("STOR " + upload_name, file)
            print("File has been uploaded")
        self.ftps.close()


def download_cebpl_test_set(**kwargs):
    import datetime,time
    params = kwargs
    params_FTP = params.get('params_FTP')
    params_ESI = params.get('params_ESI')
    input_path = params_FTP.get('input_path')
    Path_ALLER = params_ESI.get('Path_ALLER',None)
    clean_files = params_FTP.get('clean_files', False)
    n_files_host = params_FTP.get('n_files_host', 4)
    filelist_path = params_FTP.get('filelist_path',None)
    filelist_csv = "filelist.csv"

    filelist_name = os.path.join(filelist_path, "filelist.csv")
    if not os.path.exists(filelist_name): 
        filelist = pd.DataFrame(columns = ['Date','filein','fileout'])
        filelist.to_csv(filelist_name,index=False)

    filelist = pd.read_csv(os.path.join(filelist_path, "filelist.csv"))
    
    FT = FTP_connector(**kwargs)
    for n in Path_ALLER: FT.ftps.cwd(n)
    FT.ftps.dir()
    def filter_in(distant_file,local_file,filelist, **kwargs):
        file_list = "".join(str(filelist.filein.to_list()))
        return  not os.path.exists(local_file) or distant_file not in file_list
    def clean(n):
        host_files = FT.ftps.nlst()
        local_files = os.listdir(input_path)
        if set(host_files) == set(local_files) and len(set(host_files)) > (n+1) :
            files_del = host_files[-n:] # keep last n files
            for i in range(n):
                FT.ftps.delete(files_del[i])
        else:
            if set(host_files) > set(local_files): 
                print('Files are not on local disc:',set(local_files) - set(host_files))
    out = []
    files = FT.ftps.nlst()
    for file in files:
        local_file  = os.path.join(input_path, file)
        if filter_in(file,local_file, filelist, **kwargs):
            FT.ftps.retrbinary("RETR "+file,open(local_file, 'wb').write)
            out.append(local_file)
            today_date = datetime.date.today()
            filelist = filelist.append(dict(zip(filelist.columns,[today_date,str(local_file),""])),ignore_index=True)
        print("Download status:success", file)
    if clean_files:
        clean(n_files_host)
    print(pd.DataFrame(FT.ftps.nlst(), columns=["list of files on host"]))
    FT.ftps.close()
    filelist.to_csv(filelist_name, index=False)
    return out
    pass

def upload_ESI(**kwargs):
    import datetime,time
    params = kwargs
    params_FTP = params.get('params_FTP')
    params_ESI = params.get('params_ESI')
    output_path = params_FTP.get('output_path')
    Path_RETOUR = params_ESI.get('Path_RETOUR')
    filelist_path = params_FTP.get('filelist_path',None)
    filelist_csv = "filelist.csv"

    filelist_name = os.path.join(filelist_path, filelist_csv)
    if not os.path.exists(filelist_name): 
        filelist = pd.DataFrame(columns = ['Date','filein','fileout'])
        filelist.to_csv(filelist_name,index=False)
    filelist = pd.read_csv(os.path.join(filelist_path, "filelist.csv"))
    def exist(file):
        if file in FT.ftps.nlst():
            return True
        else:
            return False
    def filter_out(file):
        file_list = "".join(str(filelist.fileout.to_list()))
        return  not exist(file) and file not in file_list
    out = []
    FT = FTP_connector(**params)
    for n in Path_RETOUR: FT.ftps.cwd(n)
    files = os.listdir(output_path)
    for file in files:
        if filter_out(file):
            today_date = datetime.date.today()
            path = os.path.join(output_path,file)
            with open(path, "rb") as file_:
                FT.ftps.storbinary("STOR " + file, file_)
                out.append(file)
                today_date = datetime.date.today()
                filelist = filelist.append(dict(zip(filelist.columns,[today_date,"",str(file)])),ignore_index=True)
        print("upload status:success", file)
    print(pd.DataFrame(FT.ftps.nlst(), columns=["list of files on host"]))
    FT.ftps.close()
    filelist.to_csv(filelist_name, index=False)
    pass    

if __name__ == "__main__":
    pass




    