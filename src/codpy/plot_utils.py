from matplotlib import pyplot as plt
import numpy as np

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
