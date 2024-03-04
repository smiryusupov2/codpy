import os,sys
parent_path = os.path.dirname(__file__)
parent_path = os.path.dirname(parent_path)
if parent_path not in sys.path: sys.path.append(parent_path)
from include import *
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.datasets import fetch_california_housing
from math import pi, factorial
import unittest

import os
import ssl

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

def func(x, order = 1):
    return 1/factorial(order) * x ** order


def test_extrapolation_linear(func, decimal = 3):
    x = np.random.randn(100, 1)
    z = np.random.randn(100, 1)
    fx =func(x)
    fz = func(z)
    
    f_z = op.extrapolation(x, z, fx, kernel_fun="linear", map=None)

    np.testing.assert_almost_equal(f_z, fz, decimal=decimal)

def test_denoiser(func, decimal = 3):
    x = np.random.randn(100, 1)
    z = np.random.randn(100, 1)
    fx =func(x)
    fz = func(z)
    f_z_den = op.denoiser(x,z,fx, kernel_fun = "maternnorm", map = "standardmean")

    np.testing.assert_almost_equal(fz, f_z_den, decimal=decimal)

def test_norm(func, decimal = 3):
    x = np.random.randn(100, 1)
    z = np.random.randn(100, 1)
    fx =func(x)
    norm_ =  op.norm(x,x,x,fx, kernel_fun="linear", map=None, rescale=True)
    alpha, residuals, rank, s = np.linalg.lstsq(x, fx, rcond=None)
    alpha = np.squeeze(alpha) 
    norm = alpha** 2
    np.testing.assert_almost_equal(norm, norm_, decimal=decimal)

def test_Knm():
    x = np.random.randn(100, 1)
    kernel.rescale(x)
    Knm = op.Knm(x=x,y=x)

def test_Knm_inv(decimal = 3):
    x = np.random.randn(10, 2)
    fx = np.random.randn(10, 3)
    kernel.rescale(x)
    Knm = op.Knm(x=x,y=x)
    Knm_inv = lalg.cholesky(x=Knm,eps=1e-2)
    Kinv = op.Knm_inv(x=x,y=x,fx=fx)
    Kinv1 = np.linalg.solve(x.T @ x, fx.T).T

    np.testing.assert_almost_equal(Kinv, Kinv1, decimal=decimal)

def test_diff_matrix(decimal = 3):
    x = np.random.randn(100, 1)
    z = np.random.randn(100, 1)
    DiffM = np.sum((x[:, np.newaxis, :] - z[np.newaxis, :, :]) ** 2, axis=2)
    D = op.Dnm(x = x, y = z, distance = "norm22", kernel_fun="linear", map=None)
    np.testing.assert_almost_equal(DiffM, D, decimal=decimal)


def test_distance_labelling(decimal = 3):
    x = np.random.randn(100, 1)
    z = np.random.randn(100, 1)
    ind = distance_labelling(x,z)
    DiffM = np.sum((x[:, np.newaxis, :] - z[np.newaxis, :, :]) ** 2, axis=2)
    ind2 = np.argmax(DiffM, axis = 1)
    np.testing.assert_almost_equal(ind, ind2, decimal=decimal)

def test_discrepancy(decimal = 0):
    x = np.random.randn(100, 1)
    z = np.random.randn(100, 1)
    xx = x.dot(x.T)
    zz = z.dot(z.T)
    xz = x.dot(z.T)

    MMD =  xx.mean() - 2 * xz.mean() + zz.mean()
    disc = discrepancy(x,x,z, kernel_fun="linear", map=None)

    np.testing.assert_almost_equal(disc, MMD, decimal=decimal)

def test_lsap(decimal = 3):
    cost_matrix = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    lsap1 = lsap(cost_matrix)
    lsap2 = scipy_lsap(cost_matrix)

    np.testing.assert_almost_equal(lsap2, lsap1, decimal=decimal)

def test_encoder_decoder(alpha=0.05):
    x = np.random.randn(100, 1)
    z = np.random.rand(100, 1)
    enc = encoder(x,z)
    dec = decoder(enc)
    decoded_x = dec(z)

    # Flatten arrays for KS test (KS test requires 1D arrays)
    x_flattened = x.flatten()
    decoded_x_flattened = decoded_x.flatten()

    # Perform KS test
    ks_statistic, p_value = stats.ks_2samp(x_flattened, decoded_x_flattened)

    # Check if p-value is significant
    assert p_value > alpha, f"KS test failed with p-value {p_value}. The distributions of x and decoded_x may be different."

def test_match(alpha=0.05):
    x = np.random.randn(1000, 1)
    matched_x = match(x, Ny = 300)

    # Flatten arrays for KS test (KS test requires 1D arrays)
    x_flattened = x.flatten()
    decoded_x_flattened = matched_x.flatten()

    # Perform KS test
    _, p_value = stats.ks_2samp(x_flattened, decoded_x_flattened)

    # Check if p-value is significant
    assert p_value > alpha, f"KS test failed with p-value {p_value}. The distributions of x and decoded_x may be different."


def test_nabla(func, decimal = 3):
    x = np.random.randn(100, 1)
    z = np.random.randn(100, 1)
    fx = func(x, order = 1)
    _nabla = diffops.nabla(x,x,z,fx,kernel_fun="linear", map=None)
    np.testing.assert_almost_equal(_nabla, 1, decimal=decimal)

def test_nablaTnabla(func, decimal = 3):
    x = np.random.randn(100, 1)
    z = np.random.randn(100, 1)
    fx = func(x, order = 2)
    _nabla = diffops.nabla(x,x,z,fx,kernel_fun="linear", map=None)
    _nabla1 = diffops.nablaT(x,x,z,fz = _nabla,kernel_fun="linear", map=None)
    _nablaTnabla = diffops.nablaT_nabla(x,x,fx,kernel_fun="linear", map=None)
    np.testing.assert_almost_equal(_nablaTnabla, _nabla1, decimal=decimal)

def test_LerayT(func, decimal = 3):
    x = np.random.randn(100, 1)
    z = np.random.randn(100, 1)
    fx = func(x, order = 2)
    _nabla = diffops.nabla(x,x,z,fx,kernel_fun="linear", map=None)
    _nabla_inv = diffops.nabla_inv(x,x,z,fz = _nabla, fx = fx,kernel_fun="linear", map=None)
    _LerayT_ = diffops.nabla(x,x,z,fx = _nabla_inv,kernel_fun="linear", map=None)
    _LerayT = diffops.Leray_T(x,x,fx = _nabla,kernel_fun="linear", map=None)
    np.testing.assert_almost_equal(_LerayT_, _LerayT, decimal=decimal)


def test_Leray(func, decimal = 3):
    x = np.random.randn(100, 1)
    z = np.random.randn(100, 1)
    fx = func(x, order = 2)
    # fz = func(z, order = 2)
    _nabla = diffops.nabla(x,x,z,fx,kernel_fun="linear", map=None)
    _Leray_fz = _nabla - diffops.nabla(x=x,y=x,z=z,fx=diffops.nabla_inv(x=x,y=x,z=z,
                    fz= _nabla,kernel_fun="gaussian", map="standardmin"), kernel_fun="linear", map=None)
    _Leray = diffops.Leray(x,x,fx = _nabla,kernel_fun="linear", map=None)
    np.testing.assert_almost_equal(_Leray_fz, _Leray, decimal=decimal)

def test_get_normals(decimal = 2):
    normals = np.var(get_normals(100,100))
    np.testing.assert_almost_equal(normals, 1, decimal=decimal)


def test_hot_encoder():
    # Fetch a sample dataset
    california_housing = fetch_california_housing()
    california_housing_df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)

    california_housing_df['Income Category'] = pd.qcut(california_housing_df['MedInc'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    california_housing_df['Rooms per Household Category'] = pd.cut(california_housing_df['AveRooms'], bins=[0, 2, 4, 6, 100], labels=['Few', 'Normal', 'Many', 'Too Many'])

    # Assuming 'credit-g' dataset, and knowing the categorical columns in advance (for this dataset)
    # cat_cols = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker']
    cat_cols =  ['Income Category', 'Rooms per Household Category']
    print('init length:', len(cat_cols))
    # Applying the hot_encoder function
    encoded_df = hot_encoder(california_housing_df, cat_cols_include=cat_cols)
    print('columns length:', len(encoded_df.columns))

def testkernel():
    # set_kernel("maternnorm",1e-2)
    set_kernel("tensornorm",1e-2)
    test = cd.kernel.get_regularization()
    set_map("scale_to_unitcube")

if __name__ == "__main__":
    testkernel()
    test_Knm()
    test_Knm_inv()
    test_extrapolation_linear(func = func)
    test_norm(func, decimal = 3)
    test_diff_matrix(decimal = 3)
    test_denoiser(func=func)
    test_discrepancy(decimal = 0)
    test_distance_labelling()


    test_nabla(func = func, decimal = 3)
    test_nablaTnabla(func = func)
    test_LerayT(func=func)
    test_Leray(func=func)
    test_hot_encoder()


    print("Base tests passed !")


# test_lsap(decimal = 3)
# test_encoder_decoder()
# test_match()
# test_get_normals(decimal=2)


# x = np.random.randn(100, 1)
# z = np.random.randn(100, 1)
# fx = func(x)
# fz = func(z)

# # alpha = op.coefficients(x,x,fx, kernel_fun="linear", map = None)
# # #ker_den = kernel_density_estimator(x = x, y = z)
# # data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 500)
# # xx, yy = data.T
# # #ker_cor_den = kernel_conditional_density_estimator(x_vals=x_vals, y_vals=y_vals, x_data=xx, y_data=yy)


# disc_func = discrepancy_functional(x, fx, kernel_fun="linear", map=None).eval(x,z, fz, kernel_fun="linear", map=None)

# xx, zz, perm = reordering(x,z)

# _nablaTnabla_inv = diffops.nablaT_nabla_inv(x,x,fx,kernel_fun="linear", map=None)
# _hessian = diffops.hessian(x,x,fx,kernel_fun="linear", map=None).squeeze().squeeze().squeeze()
# _hessian1 = compute_hessian(x, func).squeeze().squeeze()
# print(np.linalg.norm(_hessian - _hessian1))
# pass