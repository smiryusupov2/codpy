import math
import warnings
import functools
import numpy as np

import codpy.algs
from codpy.core import _requires_rescale,KerInterface,kernel_setter,get_matrix
from codpy.data_conversion import get_matrix
from codpy.selection import column_selector
from codpy.kernel import Kernel
from codpy.lalg import LAlg
from codpy.plot_utils import multi_plot
import pandas as pd
import codpy.conditioning

class copy_distrib:
    def __init__(self,distrib,**kw):
        self.distrib = distrib
        self.x, self.y = distrib.x,distrib.y_original
    def __call__(self,**kw):
        return self.distrib.dist_ref

def cond_data(df,col_x,col_y=None,**kwargs):
    def remove_special_char(z):
        import re
        out= re.sub('[^a-zA-Z0-9 \n\.]', ' ', z)
        return out
    if col_y is None:
        col_y = set(df.columns) - set(col_x)
    col_x = [remove_special_char(col) for col in col_x]
    col_y = [remove_special_char(col) for col in col_y]
    columns = col_x+col_y
    df = pd.DataFrame(df[columns],columns = columns)


    x = pd.DataFrame(df[col_x])
    y = pd.DataFrame(df[col_y])

    threshold = kwargs.get("threshold",.2)
    threshold = np.power(threshold,1./len(col_x))
    means = df.mean()
    vars = df.var()
    for i,col in enumerate(col_x):
        if i==0: bools = np.array(np.abs(df[col] - means[col]) < vars[col]*threshold)
        else: bools = np.logical_and(bools, np.array(np.abs(df[col] - means[col]) < vars[col]*threshold))

    dist_ref = df[bools][col_y]
    return x, y, dist_ref,df,get_matrix(means[col_x]).T


class cond_marginal_plot:
    def __init__(self,distrib,**kwargs):
        class distrib_helper:
            def __init__(self,distrib,**kwargs):
                self.x,self.y_original,self.dist_ref,self.data,self.cond =  distrib(**kwargs)
        self.distrib_ = distrib_helper(distrib,**kwargs)

    def __call__(self,**kwargs):
        N= kwargs.get('N',500)
        samplers = kwargs['samplers']
        cond = self.distrib_.cond
        samplers_instances = [sampler(x=self.distrib_.x,y=self.distrib_.y_original,**kwargs) for sampler in samplers]
        # dists = [sampler(z= get_matrix(cond), N=N) for sampler in samplers_instances]
        dists = [self.distrib_.dist_ref for sampler in samplers_instances]
        dist_columns = self.distrib_.y_original.columns.to_numpy()
        dists_out = [pd.DataFrame(dist,columns =dist_columns).T for dist in dists]
        table = pd.concat(dists_out)
        index_name = [d_name.__class__.__name__+":"+c_name for d_name in samplers_instances for c_name in dist_columns]
        table.index = index_name
        self.graphic(self.distrib_,dists_out)

        return table

    @staticmethod
    def get_list(dist_ref,dists_sampled):
        out = []
        for dist in dists_sampled:
            for n,col in enumerate(dist_ref.dist_ref):
                cond_cols = dist_ref.x.columns.to_numpy()
                title = ""
                for c in cond_cols: title += c+","
                ref_col,samp_row_col = " ref. dist.", dist.__class__.__name__
                df1 = pd.DataFrame(data = dist_ref.dist_ref[col].values,columns=[col])
                df1["dist"] = ref_col
                df2 = pd.DataFrame(data = dist.out[:,n],columns=[col])
                df2["dist"] = samp_row_col
                df = pd.concat([df1,df2])
                df = {'data':df,'cond_cols':title}
                out.append((df))
                # out.append((pd.DataFrame(data = dist_ref.dist_ref[col],columns=[ref_col]),pd.DataFrame(data = dist_ref.y_original[col],columns=[samp_row_col])))
        return out
    def graphic(self,dist_ref,sampler_instances):
        def fun_plot(params,**kwargs):
            import seaborn as sns
            ax = params["ax"]
            data = params['data']
            cond_cols = params['cond_cols']
            col,hue = data.columns[0],data.columns[1]
            sns.histplot(data=data, x=col,hue=hue, bins=100,cumulative = True,ax=ax,stat="density", common_norm=False,fill=False,element="step")
            title = data.columns[0]+ "|"+ cond_cols
            ax.set_title(title)
            # ax.legend()
        multi_plot(self.get_list(dist_ref,sampler_instances),mp_ncols=len(dist_ref.dist_ref.columns),mp_nrows=len(sampler_instances),fun_plot=fun_plot)

class cond_pairgrid_plot(cond_marginal_plot):
    @staticmethod
    def get_list(dist_ref,dists_sampled):
        out = []
        for dist in dists_sampled:
            # cond_cols = dist_ref.x.columns.to_numpy()
            title = ""
            df1 = pd.DataFrame(data = dist,columns=dist_ref.dist_ref.columns)
            ref_col,samp_row_col = " ref. dist.", dist.__class__.__name__
            df1["dist"] = samp_row_col
            df2 = dist_ref.dist_ref
            df2["dist"] = ref_col
            df = pd.concat([df1,df2])
            df = {'data':df,'cond_cols':title}
            out.append(df)
            # out.append((pd.DataFrame(data = dist_ref.dist_ref[col],columns=[ref_col]),pd.DataFrame(data = dist_ref.y_original[col],columns=[samp_row_col])))
        return out
    def graphic(self,dist_ref,sampler_instances):
        def fun_plot(params):
            if isinstance(params,list):return [fun_plot(p) for p in params]
            # ax = params.get("ax",plt.figure(figsize = (8,4)))
            import seaborn as sns
            data = params['data']
            g = sns.PairGrid(data, diag_sharey=False,hue = "dist")
            g.map_upper(sns.scatterplot)
            # g.map_lower(sns.kdeplot,stat="density", common_norm=False)
            g.map_diag(sns.histplot,stat="density", common_norm=False,bins=100,cumulative = True)
            cond_cols = params['cond_cols']
            title = data.columns[0]+ "|"+ cond_cols
            # g.set_title(title)
            g.add_legend()
        fun_plot(self.get_list(dist_ref,sampler_instances))

def breast_data(**kwargs):
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data[:,0:4],columns = data["feature_names"][0:4])
    X['class'] = data.target
    # X.to_csv("breast.csv")
    x, y, dist_ref,df,means = cond_data(X,["class"],**kwargs)
    from sklearn.model_selection import train_test_split
    x0,y0 = x[x["class"] == 0],y[x["class"] == 0]
    x1,y1 = x[x["class"] == 1],y[x["class"] == 1]
    x0,x_ref, y0, y_ref = train_test_split(x0, y0, test_size=0.5, random_state=42) 
    x,y = pd.concat([x0,x1],axis=0,ignore_index=True),pd.concat([y0,y1],axis=0,ignore_index=True)  
    return x, y, y_ref,df,0

def breast_example(**kwargs):
    return cond_pairgrid_plot(breast_data,threshold = 0.25)(samplers = [codpy.conditioning.ConditionerKernel],**kwargs)


if __name__ == "__main__":
    print(breast_example())
    pass 

