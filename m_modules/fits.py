import sys, os
sys.path.append('/Users/meet/Desktop/Courses/Research/Chem/Code/QA/g_modules')
sys.path.append('/Users/meet/Desktop/Courses/Research/Chem/Code/QA/m_modules')
# Add the parent directory to the system path
sys.path.append('/Users/meet/Desktop/Courses/Research/Chem/Code/QA')

from pyscf import gto,scf, cc
import numpy as np
import pyscf
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import basis_set_exchange as bse
from g_modules.FcMole import FcM, FcM_like
from g_modules.AP_class import APDFT_perturbator as AP
from g_modules.alch_deriv import first_deriv_elec,DeltaV
from m_modules.stat_funcs import *
from scipy.optimize import curve_fit
import numpy.fft as fft
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from m_modules import energy
from m_modules.energy import *
from m_modules.fits import *
from m_modules.config import *
# from m_modules.plots import *
import scipy
from scipy.fftpack import idct, idst

def alc_fitting(atomic_number, frac_energies, step, max_lam_err, x_axis, prediction):

    pre_restricted = np.array([ prediction[i][:max_lam_err*int(1/step) + 1] for i in range(3)])
    dft_restricted = frac_energies[:max_lam_err*int(1/step) + 1]
    # print(dft_restricted == frac_energies, dft_restricted.shape)
    # free_restricted = free_energies[:max_lam_err*int(1/step) + 1]

    '''Defining the new x axis for fitting errors'''
    x_axis_err = reflect(x_axis[:max_lam_err*int(1/step) + 1],-1)
    '''Fitting away 2nd order errors'''
    first_order_err = reflect(dft_restricted - pre_restricted)
    # first_order_err = dft_restricted - prediction_7_3[:,:max_lam_err*int(1/step) + 1]
    err_fits = [] # across the whole x-axis
    err_res_fits = [] # across restricted x-axis
    # fitting for only equi. diatomic

    fit = quad_fit
    for i in range(1):
        popt_, pcov_ = curve_fit(fit, x_axis_err, first_order_err[i],absolute_sigma=True,maxfev=100000)
        err_res_fits.append(np.array(fit(x_axis_err, *popt_)))
        err_fits.append(np.array(fit(reflect(x_axis,-1), *popt_)))

    quad_adjusted_pre = reflect(prediction) + np.array(err_fits)
    quad_adjusted_pre_res = reflect(pre_restricted) + np.array(err_res_fits)
    # np.array([quad_adjusted_pre[i][max_lam_err*int(1/step):2*max_lam_err*int(1/step) + 1] for i in range(3)])

    # plotting error between actual and prediction
    comps = []
    for i in range(3):
        comps.append(get_mol_symbol(atomic_number, i,-i))


    # h_real_n ,f_real_n, A_real,f_real_p, n_real_p, arr_fft,inv_fft = FT(reflect(dft_restricted),quad_adjusted_pre_res, max_lam_err, comps,atomic_number)

    second_order_err = reflect(dft_restricted) - quad_adjusted_pre_res

    my_params = [-2.63982116e-02 , 4.03299727e-02 , 2.52607851e-01,  0,0,0]
    manual_fit = beat(x_axis_err,*my_params)

    err_fits = [] # across the whole x-axis
    err_res_fits = [] # across restricted x-axis
    for i in range(1):
        popt_, pcov_ = curve_fit(beat, x_axis_err, second_order_err[i],absolute_sigma=True,maxfev=100000,\
            p0 =my_params) #  (A_real, f_real_p,f_real_n,0, 0)
        err_res_fits.append(np.array(beat(x_axis_err, *popt_)))
        err_fits.append(np.array(beat(reflect(x_axis,-1), *popt_)))

    product_sines_pre = quad_adjusted_pre + np.array(err_fits)
    product_sines_pre_res =  quad_adjusted_pre_res + np.array(err_res_fits)


    fig1 = plot_errors(fit.__name__,x_axis_err ,reflect(dft_restricted),\
        reflect(pre_restricted),\
            quad_adjusted_pre_res , comps, atomic_number, product_sines_pre_res\
                -quad_adjusted_pre_res, manual_fit)

    plt.show()



    # correction = closed_form(x_axis_err,h_real_n ,f_real_n, A_real,f_real_p, n_real_p)
    # subplots = 2
    # fig, ax = plt.subplots(subplots, 1, figsize=(5*subplots, 5*subplots), sharex=True)
    # ax[0].plot(inv_fft, label='inverse fft')
    # ax[0].plot(arr_fft, label=' arr fft')
    # ax[1].plot(correction, label='correction')
    # ax[0].legend()
    # ax[1].legend()

    # l_nl_plot(atomic_number, exponent, d, frac_energies, pre_NN_l, pre_CO_l, pre_BF_l, prediction, x_axis)
