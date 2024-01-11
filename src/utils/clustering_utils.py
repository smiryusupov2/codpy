from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from codpy.src.utils.graphical import get_representation_function
import seaborn as sns

############################clustering_utilities

class graphical_cluster_utilities:
    def plot_clusters(predictor ,ax, **kwargs):
        fun = get_representation_function(**kwargs)

        xlabel = kwargs.get('xlabel',"pca1")
        ylabel = kwargs.get('ylabel',"pca2")
        cluster_label = kwargs.get('cluster_label',"cluster:")


        x = np.asarray(predictor.z)
        fx = np.asarray(predictor.f_z)
        centers = np.asarray(predictor.y) 
        ny = len(centers)
        if (len(x)*len(fx)*len(centers)):
            colors = plt.cm.Spectral(fx / ny)
            x,y = fun(x)
            num = len(x)
            df = pd.DataFrame({'x': x, 'y':y, 'label':fx}) 
            groups = df.groupby(fx)
            for name, group in groups:
                ax.plot(group.x, group.y, marker='o', linestyle='', ms=50/np.sqrt(num), mec='none')
                ax.set_aspect('auto')
                ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
                ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
            if len(centers):
                c1,c2 = fun(centers)
                ax.scatter(c1, c2,marker='o', c="black", alpha=1, s=200)
                for n in range(0,len(c1)):
                    a1,a2 = c1[n],c2[n]
                    ax.scatter(a1, a2, marker='$%d$' % n, alpha=1, s=50)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.title.set_text(cluster_label + str(ny))


def plot_confusion_matrix(confusion_matrix ,ax = None, **kwargs):
    if ax is None: ax = plt.axes()
    fmt = kwargs.get('fmt',"d")
    cmap = kwargs.get('cmap',plt.cm.copper)
    sns.heatmap(confusion_matrix, ax=ax, annot=True, fmt=fmt, cmap=cmap)
    labels = kwargs.get('labels',"")
    title = kwargs.get('title',"Conf. Mat.:")
    fontsize,rotationx,rotationy = kwargs.get('fontsize',14),kwargs.get('rotationx',90),kwargs.get('rotationy',360)
    if ax is not None:
        ax.set_title(title, fontsize=fontsize)
        ax.set_xticklabels(labels, fontsize=fontsize, rotation=rotationx)
        ax.set_yticklabels(labels, fontsize=fontsize, rotation=rotationy)


class add_confusion_matrix:
    def confusion_matrix(self):
        out = []    
        if len(self.fz)*len(self.f_z):out = metrics.confusion_matrix(self.fz, self.f_z)
        return out
    def plot_confusion_matrix(predictor ,ax, **kwargs):
        sns.heatmap(predictor.confusion_matrix(), ax=ax, annot=True, fmt="d", cmap=plt.cm.copper)
        labels = kwargs.get('labels',[str(s) for s in np.unique(predictor.fz)])
        title = kwargs.get('title',"Conf. Mat.:")
        ax.set_title(title, fontsize=14)
        ax.set_xticklabels(labels, fontsize=14, rotation=90)
        ax.set_yticklabels(labels, fontsize=14, rotation=360)