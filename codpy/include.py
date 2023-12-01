import os,sys
project_root = os.path.dirname(os.path.dirname(__file__))

# Adding the project root to the sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
common_path = os.path.join(project_root, "codpy", "src")
if common_path not in sys.path:
    sys.path.append(common_path)

predictors_path = os.path.join(project_root, "codpy", "utils")
if predictors_path not in sys.path:
    sys.path.append(predictors_path)
#######################################

