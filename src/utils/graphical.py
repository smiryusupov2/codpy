from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
from codpy.src.utils.data_conversion import *

#########################graphical utilities###################

########################################

def scatter_plot(param,**kwargs) -> None:
    x,y = get_matrix(param[0]),get_matrix(param[1])
    if x.shape[0]*x.shape[1]*y.shape[0]*y.shape[1] == 0: return
    color = kwargs.get('color',["blue","red"])
    label = kwargs.get('label',["first","second"])
    type = kwargs.get('type',["o","o"])
    markersize = kwargs.get('markersize',[2,2])
    plt.plot(x[:,0], x[:,1],'o',color = color[0], label = label[0], markersize = markersize[0])
    plt.plot(y[:,0], y[:,1],'o',color = color[1], label = label[1], markersize = markersize[1])
    # plt.plot(y[:,0], y[:,1],'o',color = 'red', label = "Sampling", markersize=2, alpha = 0.5)
    plt.legend()

def graph_plot(param, **kwargs):
    scatter_plot(param, **kwargs)
    x,y = param[0],param[1]
    x,y = get_matrix(param[0]),get_matrix(param[1])
    if x.shape[0]*x.shape[1]*y.shape[0]*y.shape[1] == 0: return
    N = min(len(x),len(y))
    color = kwargs.get('color_edge','black')
    plt.plot([y[0:N,0], x[0:N,0] ], [ y[0:N,1], x[0:N,1] ], linewidth=1,color = color)


def multiple_plot1D_fun(x,fx,title = 'Figure',labelx='fx-axis:',labely='fz-axis:',legend=[]):
    fig = plt.figure()
    if (len(x)):
        if (x.ndim == 1): x = x.reshape(len(x),1)
        if (fx.ndim == 1): fx = fx.reshape(len(fx),1)
        N,D = len(x), fx.shape[1]
        N = len(x)
        # print("N,D:",N,D)
        plotfx,plotx,permutation = lexicographical_permutation(fx,x)
        plotx = plotx.reshape(N)
        for i in range(D):
            curve = plotfx[:,i]
            plt.plot(plotx,curve,marker = 'o',ls='-',markersize=2)
    if (len(legend) != D):
        legend = []
        for i in range(D):
            legend.append('line plot '+str(i))
    plt.legend(legend)
    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)

def multiple_plot1D(x,fx,title = 'Figure',labelx='fx-axis:',labely='fz-axis:',legend=[]):
    multiple_plot1D_fun(flattenizer(x),flattenizer(fx),title,labelx,labely,legend)

def multiple_norm_error(x,fx,ord=None):
    N,D = fx.shape
    out = []
    if (len(x)):
        for i in range(D):
            out.append(np.linalg.norm(x-fx[:,i],ord))
    return out

def show_imgs(images, ax=None,**kwargs):
    j = 0
    if isinstance(images, list) :
        pixels = []
        for image in images:
            if not len(pixels): pixels = image
            else: pixels = np.concatenate( (pixels,image),axis=1)
        ax.imshow(pixels, cmap='gray')
    else: plt.imshow(images, cmap='gray')

def compare_plot_lists_ax(listxs, listfxs, ax, **kwargs):
    index = kwargs.get("index",0)
    labelx=kwargs.get("labelx",'x-units')
    fun_x=kwargs.get("fun_x",get_data)
    extra_plot_fun=kwargs.get("extra_plot_fun",None)
    labely=kwargs.get("labely",'f(x)-units')
    listlabels=kwargs.get("listlabels",[None for n in range(len(listxs))])
    listalphas=kwargs.get("alphas",np.repeat(1.,len(listxs)))
    xscale =kwargs.get("xscale",None)
    yscale =kwargs.get("yscale",None)
    figsize =kwargs.get("figsize",(2,2))
    loc =kwargs.get("loc",'upper left')
    prop =kwargs.get("prop",{'size': 6})


    for x,fx,label,alpha in zip(listxs, listfxs,listlabels,listalphas):
        plotx = fun_x(x)
        plotfx = get_data(fx)
        plotx,plotfx,permutation = lexicographical_permutation(plotx,plotfx,**kwargs)
        if extra_plot_fun is not None: extra_plot_fun(plotx,plotfx)
        ax.plot(plotx,plotfx,marker = 'o',ls='-',label= label, markersize=12 / len(plotx),alpha = alpha)
        ax.legend(prop={'size': 6})
    title = kwargs.get("title",'')
    ax.title.set_text(title)
    if yscale is not None: ax.set_yscale(yscale)
    if yscale is not None: ax.set_xscale(xscale)
    ax.title.set_text(title)
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)


def compare_plot_lists(kwargs):
    listxs,listfxs,ax = kwargs['listxs'],kwargs['listfxs'],kwargs['ax']
    from matplotlib.dates import date2num
    if ax: return compare_plot_lists_ax(fun_axvspan = None, **kwargs)
    index = kwargs.get("index",0)
    labelx=kwargs.get("labelx",'x-units')
    fun_x=kwargs.get("fun_x",get_data)
    extra_plot_fun=kwargs.get("extra_plot_fun",None)
    labely=kwargs.get("labely",'f(x)-units')
    listlabels=kwargs.get("listlabels",[None for n in range(len(listxs))])
    listalphas=kwargs.get("alphas",np.repeat(1.,len(listxs)))
    xscale =kwargs.get("xscale",None)
    yscale =kwargs.get("yscale",None)
    figsize =kwargs.get("figsize",(2,2))
    plt.figure(figsize=figsize)
    for x,fx,label,alpha in zip(listxs, listfxs,listlabels,listalphas):
        plotx = fun_x(x)
        plotfx = get_data(fx)
        plotx,plotfx,permutation = lexicographical_permutation(x=plotx,fx=plotfx,**kwargs)
        if extra_plot_fun is not None: extra_plot_fun(plotx,plotfx)
        plt.plot(plotx,plotfx,marker = 'o',ls='-',label= label, markersize=12 / len(plotx),alpha = alpha)
        plt.legend(prop={'size': 6})
    title = kwargs.get("title",'')
    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)

def matrix_to_cartesian(x,fx):
    fx = fx.reshape((len(fx)))
    # print('matrix_to_cartesian:x:', x.shape)
    # print('matrix_to_cartesian:fx:', fx.shape)
    size = int(len(x)**(1/2))
    Xc = sorted(list(set(x[:,0])))
    Yc = sorted(list(set(x[:,1])))

    assert len(fx)==len(Xc)*len(Yc)
    X,Y = np.meshgrid(Xc,Yc)
    Z = np.zeros((len(Xc),len(Yc)))
    for i in range(len(fx)):
       indx = Xc.index(x[i,0])
       indy = Yc.index(x[i,1])
       Z[indx,indy] = fx[i]
       #Z[indx,indy] = x[i,0]+x[i,1]
       Y[indx,indy] = x[i,1]
       X[indx,indy] = x[i,0]
    #print('X:',X)
    #print('X.shape:',X.shape)
    #print('Y:',Y)
    #print('Y.shape:',Y.shape)
    #print('Z:',Z)
    #print('Z.shape:',Z.shape)
    return X,Y,Z
    
def multi_compare1D(x,fx, title = 'Figure',labelx='x-axis',labely='y-axis:',figsizex=6.4, figsizey=4.8, flip=True):
    if (len(x)):
        if (x.ndim == 1): x = x.reshape(len(x),1)
        D = x.shape[1]
        fig = plt.figure(figsize=(figsizex,figsizey))
        fig.suptitle(title)
        for d in np.arange(0,D): 
            if (flip):
                plotfx,plotx,permutation = lexicographical_permutation(fx.flatten(),x[:,d])
            else:
                plotfx,plotx,permutation = lexicographical_permutation(x[:,d],fx.flatten())
            plt.subplot(D, 1, d + 1)
            plt.plot(plotx,plotfx,marker = 'o',ls='-',label= labelx,markersize=2)
            plt.xlabel(labelx)
            plt.ylabel(labely + str(d))

class fun_pca:
    pca = None
    def __call__(self, x):
        if not self.pca:
            self.pca = PCA()
            principal_components = self.pca.fit_transform(x)
        else: principal_components = self.pca.transform(x)
        return principal_components[:,0], principal_components[:,1]

def get_representation_function(**kwargs):
    fun = fun_pca()
    index1 = int(kwargs.get('index1',0))
    index2 = int(kwargs.get('index2',0))
    def fun_index(x):return x[:,index1], x[:,index2]
    if (index1 != index2): fun = fun_index
    return fun


def plotD(xfx,ax=None,**kwargs):
    x,fx=xfx[0],xfx[1]
    if isinstance(x,tuple):
        color = ['b','r','g','c','m','y','k','w']
        [plotD( (y,fy),ax,color = c,markersize = 2,**kwargs) for y,fy,c in zip(x,fx,color)]
    else: 
        if x.ndim == 1: return plot1D(xfx,ax,**kwargs)
        if x.shape[1] >= 2:return plot_trisurf(xfx,ax,**kwargs)
        if x.shape[1] == 1:return plot1D(xfx,ax,**kwargs)


def plot1D(xfx,ax=None,**kwargs):
    if isinstance(xfx,pd.DataFrame):return plot1D(xfx = xfx.values.T,ax=ax,**kwargs)
    x,fx=xfx[0],xfx[1]
    if len(xfx) == 3:
        kwargs = {**kwargs,**xfx[2]}
        pass
    title = kwargs.get('title',"")
    legend = kwargs.get('legend',"")
    suptitle = kwargs.get('suptitle',"")
    markersize = kwargs.get('markersize',3)
    markerfacecolor = kwargs.get('markerfacecolor','r')
    color  = kwargs.get('color','b')
    fmt = kwargs.get('fmt','-'+color+'o')
    figsize = kwargs.get('figsize',(4, 4))
    if ax == None: fig, ax = plt.subplots(figsize=figsize)
    ax.title.set_text(suptitle)
    if (len(x)):
        plotx,plotfx,permutation = lexicographical_permutation(x=get_matrix(x),fx=get_matrix(fx),indexfx = 0)
        ax.plot(plotx.flatten(),plotfx.flatten(),fmt, markersize = markersize,markerfacecolor=markerfacecolor)
        if len(legend): ax.legend([legend])
    title = kwargs.get("title",'')
    labelx=kwargs.get("labelx",'x-units')
    labely=kwargs.get("labely",'f(x)-units')
    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)

def get_ax_helper(ax,**kwargs):
    if ax is None:
        projection = kwargs.get('projection')
        fig = plt.figure()
        if len(projection): ax = fig.add_subplot(projection = projection)
        else: ax = fig.add_subplot()
    return ax



def plot_trisurf(xfx,ax,**kwargs):
    xp,fxp = xfx[0],xfx[1]
    x,fx=get_matrix(xp),get_matrix(fxp)
    if x.shape[1] > 2: fun = fun_pca()
    elif x.shape[1] == 2: fun = lambda x : [get_data(x)[:,0], get_data(x)[:,1]]

    legend = kwargs.get('legend',"")
    elev = kwargs.get('elev',90)
    azim = kwargs.get('azim',-100)
    linewidth  = kwargs.get('linewidth',None)
    antialiased  = kwargs.get('antialiased',False)
    cmap = kwargs.get('cmap',cm.jet)
    ax = get_ax_helper(ax,**kwargs)

    if len(fx)>0:
        X, Y = fun(x)
        Z = fx.flatten()
        ax.plot_trisurf(X, Y, Z, antialiased = antialiased, cmap = cmap, linewidth=linewidth)
        ax.view_init(azim = azim, elev = elev)
        ax.title.set_text(legend)
        if isinstance(xp,pd.DataFrame):
            ax.set_xlabel(xp.columns[0])
            ax.set_ylabel(xp.columns[1])
        if isinstance(fxp,pd.DataFrame):
            ax.set_zlabel(fxp.columns[0])

def plot2D_CartesianMesh(xfx,ax,**kwargs):
    x,fx=xfx[0],xfx[1]

    suptitle = kwargs.get('suptitle',"")
    elev = kwargs.get('elev',15)
    azim = kwargs.get('azim',-100)
    cmap = kwargs.get('cmap',cm.jet)

    if len(fx)>0:
        X, Y, Z = matrix_to_cartesian(x,fx)
        ax.plot_surface(X, Y, Z, cmap=cmap)
        ax.view_init(elev = elev, azim=azim)
        ax.title.set_text(suptitle)

def flattenizer(x):
    if is_primitive(x) : return x
    if (type(x) == type([])): 
        if len(x) == 1: return flattenizer(x[0])
        else:return [flattenizer(s) for s in x]
    if (type(x) == np.ndarray): 
        x = np.squeeze(x)
        return np.asarray([flattenizer(s) for s in x])
    s = []
    for n in range(0,x.ndim):
        if x.shape[n] > 1 : s.append(x.shape[n])
    if len(s): return x.reshape(s)
    return x
    

def plot_data_distribution(df_x,df_fx):
    nCol = df_x.columns.size
    # num_cols = df_x._get_numeric_data().columns                          
    # cat_cols=list(set(df_x.columns) - set(num_cols))
    # numerical_features = df_x[num_cols]
    fig = plt.figure(figsize=(12,18))
    for i in range(len(df_x.columns)):
        fig.add_subplot(nCol/4,4,i+1)
        sns.scatterplot(df_x.iloc[:, i],df_fx)
        plt.xlabel(df_x.columns[i])
    plt.tight_layout()

def plot_data_correlation(df_x, thres = 0.8):
    num_correlation = df_x.select_dtypes(exclude='object').corr()
    plt.figure(figsize=(20,20))
    plt.title('High Correlation')
    sns.heatmap(num_correlation > thres, annot=True, square=True)

class multi_plot:
    def get_subplot(self,nrows,ncols,j,**kwargs):
        if 'projection' in kwargs : return self.fig.add_subplot(nrows,ncols,j, projection=kwargs['projection'])
        return self.fig.add_subplot(nrows,ncols,j)
    def __init__(self,params ,fun_plot, **kwargs):
        max_items = kwargs.get('mp_max_items',min(len(params),4))
        if max_items == -1: max_items = len(params)
        title = kwargs.get('mp_title','')
        ncols = kwargs.get('mp_ncols',len(params))
        nrows = kwargs.get('mp_nrows',None)
        f_names = kwargs.get('f_names',[None for n in range(len(params)) ])
        fontsize = kwargs.get('fontsize',10)
        numbers = min(len(params),max_items)
        ncols = min(ncols,numbers)
        projection = kwargs.get('projection','')
        legends = kwargs.get('legends', ["" for n in range(len(params)) ])
        if numbers == 0:return
        j = 0
        if nrows is None:
            nrows = max(int(np.ceil(numbers/ncols)),1)
        figsize= kwargs.get('mp_figsize',(8,4))
        if figsize is not None:self.fig = plt.figure(figsize = figsize)
        else: self.fig = plt.figure()
        if not isinstance(fun_plot,list):fun_plot = [fun_plot for n in range(0,len(params))]

        for param, f_name, legend,fun in zip(params, f_names, legends,fun_plot):
            if len(projection): ax = self.get_subplot(nrows,ncols,j+1, projection = projection)
            else: ax = self.get_subplot(nrows,ncols,j+1 ,**kwargs)
            if isinstance(param,dict):fun({**param,**kwargs,**{'ax':ax,'fig':self.fig}} )
            else:fun(param,**{**kwargs,**{'legend':legend,'fig':self.fig,'ax':ax}} )
            if f_name is not None: plt.title(f_name, fontsize=fontsize)

            j = j+1
            if j== ncols*nrows:
                break
        # fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        self.fig.tight_layout()

        if title: self.fig.suptitle(title, fontsize=12, fontweight='bold')

class multi_plot_figs(multi_plot):
    subfigs = None
    def __init__(self,params ,fun_plot, **kwargs):
        super().__init__(params ,fun_plot, **kwargs)
    def get_subplot(self,nrows,ncols,j,**kwargs):  # ,**kwargs
        out= self.fig.add_subplot(nrows,ncols,j)
        display_ax=kwargs.get('display_ax','off')
        #plt.axis('off')
        plt.axis(display_ax)
        return out

class multi_plot_pics(multi_plot_figs):
    class fun_pic :
        def __init__(self,fun_plot):
            self.fun_plot = fun_plot
        def __call__(self,param,**kwargs):
            import io
            from PIL import Image
            fig = self.fun_plot(param,**kwargs)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            param['ax'].imshow(Image.open(img_buf))
            # plt.close(fig)
            pass
    def __init__(self,params ,fun_plot, **kwargs):
        super().__init__(params ,multi_plot_pics.fun_pic(fun_plot), **kwargs)