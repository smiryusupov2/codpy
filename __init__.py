import sys,os,ctypes
def mkl_path():
    mkl_path = sys.exec_prefix
    mkl_path = os.path.join(mkl_path,"Library","bin")
    os.environ["PATH"] += os.pathsep + mkl_path
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    mkl_path = os.path.join(mkl_path,"mkl_rt.2.dll")
    return mkl_path
hllDll = ctypes.WinDLL(mkl_path())
import codpypyd


# project_root = os.path.dirname(os.path.dirname(__file__))
# # Adding the project root to the sys.path
# if project_root not in sys.path:
#     sys.path.append(project_root)
# common_path = os.path.join(project_root, "codpy", "src")
# if common_path not in sys.path:
#     sys.path.append(common_path)

# predictors_path = os.path.join(project_root, "codpy", "utils")
# if predictors_path not in sys.path:
#     sys.path.append(predictors_path)
# #######################################