import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import codpy.algs
import codpy.conditioning
from codpy.core import get_matrix
from codpy.data_conversion import get_matrix
from codpy.plot_utils import multi_plot


class copy_distrib:
    def __init__(self, distrib, **kw):
        self.distrib = distrib
        self.x, self.y = distrib.x, distrib.y_original

    def __call__(self, **kw):
        return self.distrib.dist_ref


def mnist_data(num=1):
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = (
        x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]) / 255.0
    )
    shape = (x_train.shape[0], x_train.shape[1])
    x_train = np.ndarray(shape=shape, dtype=float, buffer=x_train)
    idx = np.random.choice(len(x_train), 500, replace=False)
    X = x_train[idx]
    y = y_train[idx]
    one_hot_labels = hot_encoder(pd.DataFrame(y, dtype=str))
    columns = list(one_hot_labels.columns)
    index = columns.index("0_" + str(num))
    marg2 = X
    marg1 = one_hot_labels
    x_vals = np.array([1 if i == index else 0 for i in range(10)])
    original_1_samples = X[y == num]
    dist_ref = np.mean(original_1_samples, axis=0)
    return (
        get_matrix(x_vals).T,
        get_matrix(marg1),
        get_matrix(marg2),
        dist_ref,
        X,
        y,
        num,
    )


def iris_data(**kwargs):
    from sklearn.datasets import load_iris

    data = load_iris()
    feature_names = ["sep.len", "sep.wid.", "pet.len", "pet.wid."]
    X = pd.DataFrame(data.data, columns=feature_names)
    return cond_data(X, [feature_names[3]], **kwargs)


def wine_data(**kwargs):
    from sklearn.datasets import load_wine

    data = load_wine()
    X = pd.DataFrame(data.data[:, 0:4], columns=data["feature_names"][0:4])

    return cond_data(X, ["alcohol", "malic_acid"], **kwargs)


def breast_data(**kwargs):
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = pd.DataFrame(data.data[:, 0:4], columns=data["feature_names"][0:4])
    X["class"] = data.target
    # X.to_csv("breast.csv")
    x, y, dist_ref, df, means = cond_data(X, ["class"], **kwargs)
    from sklearn.model_selection import train_test_split

    x0, y0 = x[x["class"] == 0], y[x["class"] == 0]
    x1, y1 = x[x["class"] == 1], y[x["class"] == 1]
    x0, x_ref, y0, y_ref = train_test_split(x0, y0, test_size=0.5, random_state=42)
    x, y = (
        pd.concat([x0, x1], axis=0, ignore_index=True),
        pd.concat([y0, y1], axis=0, ignore_index=True),
    )
    return x, y, y_ref, df, 0


def cond_data(df, col_x, col_y=None, **kwargs):
    def remove_special_char(z):
        import re

        out = re.sub("[^a-zA-Z0-9 \n\.]", " ", z)
        return out

    if col_y is None:
        col_y = set(df.columns) - set(col_x)
    col_x = [remove_special_char(col) for col in col_x]
    col_y = [remove_special_char(col) for col in col_y]
    columns = col_x + col_y
    df = pd.DataFrame(df[columns], columns=columns)

    x = pd.DataFrame(df[col_x])
    y = pd.DataFrame(df[col_y])

    threshold = kwargs.get("threshold", 0.2)
    threshold = np.power(threshold, 1.0 / len(col_x))
    means = df.mean()
    vars = df.var()
    for i, col in enumerate(col_x):
        if i == 0:
            bools = np.array(np.abs(df[col] - means[col]) < vars[col] * threshold)
        else:
            bools = np.logical_and(
                bools, np.array(np.abs(df[col] - means[col]) < vars[col] * threshold)
            )

    dist_ref = df[bools][col_y]
    return x, y, dist_ref, df, get_matrix(means[col_x]).T


def stats_d(empirical, samples):
    from scipy.stats import ks_2samp

    empirical_quantiles = np.percentile(empirical, [5, 95], axis=0)
    sampled_quantiles = np.percentile(samples, [5, 95], axis=0)
    ks_feature1 = ks_2samp(empirical[:, 0], samples[:, 0])
    ks_feature2 = ks_2samp(empirical[:, 1], samples[:, 1])
    data = {
        "Feature1 5%": [empirical_quantiles[0, 0], sampled_quantiles[0, 0]],
        "Feature2 5%": [empirical_quantiles[0, 1], sampled_quantiles[0, 1]],
        "Feature1 95%": [empirical_quantiles[1, 0], sampled_quantiles[1, 0]],
        "Feature2 95%": [empirical_quantiles[1, 1], sampled_quantiles[1, 1]],
        "Feature1 KS p-value": [ks_feature1.pvalue, "-"],
        "Feature2 KS p-value": [ks_feature2.pvalue, "-"],
    }
    quant = pd.DataFrame(data, index=["Empirical", "Sampled"])

    empirical_cov = np.cov(empirical, rowvar=False)
    sampled_cov = np.cov(samples, rowvar=False)

    cov_difference = empirical_cov - sampled_cov
    cov = pd.DataFrame(
        cov_difference[:2, :2],
        index=["Empirical", "Sampled"],
        columns=["Empirical", "Sampled"],
    )
    return quant, cov


class cond_marginal_plot:
    def __init__(self, distrib, **kwargs):
        class distrib_helper:
            def __init__(self, distrib, **kwargs):
                self.x, self.y_original, self.dist_ref, self.data, self.cond = distrib(
                    **kwargs
                )

        self.distrib_ = distrib_helper(distrib, **kwargs)

    def __call__(self, samplers, **kwargs):
        N = kwargs.get("N", 500)
        cond = self.distrib_.cond
        samplers_instances = [
            sampler(x=self.distrib_.x, y=self.distrib_.y_original, **kwargs)
            for sampler in samplers
        ]

        def helper(sampler):
            dist_columns = self.distrib_.y_original.columns.to_numpy()
            sampled = sampler.sample(x=cond, n=N)[0]
            df_out = pd.DataFrame(sampled, columns=dist_columns)
            index_name = sampler.__class__.__name__
            df_out["class"] = index_name
            df_ref = self.distrib_.dist_ref.copy()
            df_ref["class"] = "ref. dist."
            df_out = pd.concat([df_out, df_ref])
            return df_out

        dists = [helper(sampler) for sampler in samplers_instances]

        self.graphic(dists)

        return dists

    def get_list(self, dists):
        out = []
        cond_col = self.distrib_.x.columns[0]
        for dist in dists:
            for n, col in enumerate(dist.columns):
                if col != "class":
                    dic = {"data": dist, "col": col, "cond_col": cond_col}
                    out.append(dic)
        return out

    def graphic(self, dists):
        len_ = len(dists[0].columns) - 1

        def fun_plot(params, **kwargs):
            ax = params["ax"]
            data = params["data"]
            col = params["col"]
            cond_col = params["cond_col"]
            sns.histplot(
                data=data,
                x=col,
                hue="class",
                bins=100,
                cumulative=True,
                ax=ax,
                stat="density",
                common_norm=False,
                fill=False,
                element="step",
            )
            title = col + "|" + cond_col
            ax.set_title(title)
            ax.legend()

        multi_plot(
            self.get_list(dists), mp_ncols=len_, mp_nrows=len(dists), fun_plot=fun_plot
        )
        plt.savefig("cond_marginal_plot.png")


class cond_pairgrid_plot(cond_marginal_plot):
    @staticmethod
    def get_list(dist_ref, dists_sampled):
        out = []
        for dist in dists_sampled:
            # cond_cols = dist_ref.x.columns.to_numpy()
            title = ""
            df1 = pd.DataFrame(data=dist, columns=dist_ref.dist_ref.columns)
            ref_col, samp_row_col = " ref. dist.", dist.__class__.__name__
            df1["dist"] = samp_row_col
            df2 = dist_ref.dist_ref
            df2["dist"] = ref_col
            df = pd.concat([df1, df2])
            df = {"data": df, "cond_cols": title}
            out.append(df)
            # out.append((pd.DataFrame(data = dist_ref.dist_ref[col],columns=[ref_col]),pd.DataFrame(data = dist_ref.y_original[col],columns=[samp_row_col])))
        return out

    def graphic(self, dist_ref, sampler_instances):
        def fun_plot(params):
            if isinstance(params, list):
                return [fun_plot(p) for p in params]
            # ax = params.get("ax",plt.figure(figsize = (8,4)))
            import seaborn as sns

            data = params["data"]
            g = sns.PairGrid(data, diag_sharey=False, hue="dist")
            g.map_upper(sns.scatterplot)
            # g.map_lower(sns.kdeplot,stat="density", common_norm=False)
            g.map_diag(
                sns.histplot,
                stat="density",
                common_norm=False,
                bins=100,
                cumulative=True,
            )
            cond_cols = params["cond_cols"]
            title = data.columns[0] + "|" + cond_cols
            # g.set_title(title)
            g.add_legend()

        multi_plot(
            self.get_list(dist_ref, sampler_instances),
            mp_ncols=len(dist_ref.dist_ref.columns),
            mp_nrows=len(sampler_instances),
            fun_plot=fun_plot,
        )
        # fun_plot(self.get_list(dist_ref,dist_sampled))


def breast_data(**kwargs):
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = pd.DataFrame(data.data[:, 0:4], columns=data["feature_names"][0:4])
    X["class"] = data.target
    # X.to_csv("breast.csv")
    x, y, dist_ref, df, means = cond_data(X, ["class"], **kwargs)
    from sklearn.model_selection import train_test_split

    x0, y0 = x[x["class"] == 0], y[x["class"] == 0]
    x1, y1 = x[x["class"] == 1], y[x["class"] == 1]
    x0, x_ref, y0, y_ref = train_test_split(x0, y0, test_size=0.5, random_state=42)
    x, y = (
        pd.concat([x0, x1], axis=0, ignore_index=True),
        pd.concat([y0, y1], axis=0, ignore_index=True),
    )
    return x, y, y_ref, df, 0


def iris_example(**kwargs):
    return cond_marginal_plot(iris_data, threshold=0.25)(
        samplers=[
            codpy.conditioning.ConditionerKernel,
            codpy.conditioning.NadarayaWatsonKernel,
        ],
        **kwargs,
    )
    # return cond_marginal_plot(iris_data,threshold = 0.25)(samplers = [codpy.conditioning.NadarayaWatsonKernel],**kwargs)


if __name__ == "__main__":
    iris_example(order=1)
    pass
