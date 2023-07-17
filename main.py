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
from m_modules.config import *
from m_modules.plots import *
import scipy
from scipy.fftpack import idct, idst
from scipy.signal import unit_impulse as dirac_delta
exponent = round(2.33, 2)
step = 0.1
def initiate(atomic_number):
    max_lam = atomic_number
    steps =  int(atomic_number / step  + 1)
    d = 2.1
    energy.max_lam, energy.steps, energy.step, energy.exponent, energy.d = max_lam, steps, step, exponent, d
    # return_max_lam_steps() # ensures a 0.1 step size

    NN = gto.M(atom= f"{atomic_number} 0 0 0; {atomic_number} 0 0 {d}",unit="Bohr",basis='unc-ccpvdz') # uncontracted cc-pvdz
    mf_NN, e_NN = new_mol(NN, exponent, 0,0)[1],new_mol(NN,exponent, 0,0)[2]
    CO, mf_CO, e_CO = new_mol(NN, exponent, -1,1)[:3]
    BF, mf_BF, e_BF = new_mol(NN, exponent, -2,2) [:3]

    # e_BeNe, e_LiNa,e_HeMg, e_HAl, e_Si = new_mol(NN,exponent,-3,3)[2],new_mol(NN,exponent,-4,4)[2], new_mol(NN,exponent,-5,5)[2],\
    #                                     new_mol(NN,exponent,-6,6)[2],new_mol(NN,exponent,-7,7)[2]

    '''Data for max_lambda = 5 and = 7 is stored in data folder'''
    ## Calculating actual fractional charge energies

    # data = get_symmetric_change_data(0,max_lam,atomic_number)
    data = get_symmetric_change_data(0,max_lam,atomic_number)
    frac_energies,free_energies =  data[0], data[1]

    '''getting / generating data for protonation'''

    # gen = False
    # data = get_asymmetric_change_data(0,max_lam, gen)
    # R_P,L_P = data[0],data[1]

    AG_NN = AG(mf_NN) # the alchemical gradient i.e. d_E / d_Z_I for where Z_I are N atoms
    AG_CO = AG(mf_CO)
    AG_BF = AG(mf_BF)

    # Evaluating the linearized energy gradient at lambda = 0. See argument fo d_Z_lambda in get_pred function in energy.py
    pre_NN_l, pre_NN_nl = gen_data(NN,AG_NN,exponent,e_NN,0, max_lam)
    pre_CO_l, pre_CO_nl = gen_data(CO,AG_CO,exponent,e_CO,-1, max_lam -1)
    pre_BF_l, pre_BF_nl = gen_data(BF,AG_BF,exponent,e_BF,-2, max_lam -2)
    prediction_7_3 = np.array([pre_NN_nl, pre_CO_nl, pre_BF_nl])

    '''
    Preparing prediction data for plotting for symmetrical alchemical changes for n in (0.5, 3) in steps of 0.1'''
    '''
    Structure of predictions_n:
        predictions_n = [...,[pre_NN_nl_n, pre_CO_nl_n, pre_BF_nl_n],...]
        pre_NN_nl_n = non-linear Z prediction over all lambda at n from NN
    '''
    # predictions_n = []
    # for n in np.linspace(0.5,5,46):
    #     pre_NN_nl_n = gen_data(NN,AG_NN,n,e_NN,0, max_lam)[1]
    #     pre_CO_nl_n = gen_data(CO,AG_CO,n,e_CO,-1, max_lam -1)[1]
    #     pre_BF_nl_n = gen_data(BF,AG_BF,n,e_BF,-2, max_lam -2)[1]
    #     predictions_n.append(np.array([pre_NN_nl_n, pre_CO_nl_n, pre_BF_nl_n]))
    # predictions_n = np.array(predictions_n)
    # prediction_2 = predictions_n[15]

    # fitting errors only upto a certain lambda
    max_lam_err = max_lam
    # store difference between numpy arrays of size (51,) and (3,71) in a numpy array of size (3,51) in one line
    prediction = prediction_7_3
    pre_restricted = np.array([ prediction[i][:max_lam_err*int(1/step) + 1] for i in range(3)])
    dft_pred_restricted = frac_energies[:max_lam_err*int(1/step) + 1]
    '''Defining the total x_axis'''
    x_axis = np.linspace(0,max_lam,steps)

    '''Defining the new x axis for fitting errors'''
    x_axis_err = reflect(x_axis[:max_lam_err*int(1/step) + 1],-1)
    '''Fitting away 2nd order errors'''
    quad_err = reflect(dft_pred_restricted - pre_restricted)
    # quad_err = dft_pred_restricted - prediction_7_3[:,:max_lam_err*int(1/step) + 1]
    err_fits = [] # across the whole x-axis
    err_res_fits = [] # across restricted x-axis
    # fitting for only equi. diatomic
    fit = quad_fit
    for i in range(1):
        popt_, pcov_ = curve_fit(fit, x_axis_err, quad_err[i],absolute_sigma=True,maxfev=100000)
        err_res_fits.append(np.array(fit(x_axis_err, *popt_)))
        err_fits.append(np.array(fit(reflect(x_axis,-1), *popt_)))

    quad_adjusted_prediction = reflect(prediction) + np.array(err_fits)
    quad_adjusted_pre_restriced = reflect(pre_restricted) + np.array(err_res_fits)
    # np.array([quad_adjusted_prediction[i][max_lam_err*int(1/step):2*max_lam_err*int(1/step) + 1] for i in range(3)])

    # plotting error between actual and prediction
    comps = []
    for i in range(3):
        comps.append(get_mol_symbol(atomic_number, i,-i))
    fig1 = plot_pol_errors(fit.__name__,x_axis_err ,reflect(dft_pred_restricted),\
        reflect(pre_restricted),\
            quad_adjusted_pre_restriced , comps, atomic_number)

    h_real_n ,f_real_n, A_real,p_real_p, n_real_p,fft_real = FT(reflect(dft_pred_restricted),quad_adjusted_pre_restriced, max_lam_err, comps,atomic_number)
    print('All good')
    correction = closed_form(x_axis_err,h_real_n ,f_real_n, A_real,p_real_p, n_real_p)
    subplots = 2
    fig, ax = plt.subplots(subplots, 1, figsize=(5*subplots, 5*subplots), sharex=True)
    ax[0].plot(fft.ifft(fft.ifftshift(fft_real)), label='inverse fft')
    ax[1].plot(correction, label='correction')
    ax[0].legend()
    ax[1].legend()



def FT(actual, new_prediction, max_lam_err, comps, atomic_number):
    err = actual - new_prediction
    subplots = 2
    fig, ax = plt.subplots(subplots, 1, figsize=(5*subplots, 5*subplots), sharex=True)
    imag_peak_freqs = []
    imag_peak_heights = []

    # put 1 to only plot error from equilibrium diatomic.
    # for i in range(1):
    i = 0
    data = err[i][:-1]
    N = len(data)  # Number of data points
    normalization = 1   # Normalization factor
    fft_data = normalization * fft.fftshift(fft.fft(data))
    fft_sine_data = fft.fftshift(normalization * scipy.fft.dst(data))[N//2:]
    fft_cos_data = fft.fftshift(normalization * scipy.fft.dct(data))[N//2:]

    print(check_even(fft_data[1:]))
    # Compute the frequencies
    frequencies = fft.fftshift(fft.fftfreq(N, d = step))

    # plot only half of the frequencies
    frequencies = frequencies[N//2:]
    fft_data = fft_data[N//2:]



    ax[0].plot(frequencies, np.real(fft_data),label=comps[i]) #,  label=f'{frequencies[peaks_r[0]]} {peaks_r[1]}'


     # # finding peaks for feeding curve-fit parameters
    # positive peaks
    # def fitting():
    peaks_r = find_peaks(np.real(fft_data), height = 0.01)
    real_peak_freq = frequencies[peaks_r[0][0]]
    real_peak_height = peaks_r[1]['peak_heights'][0]

    # negative peaks
    peaks_r_n = find_peaks(-np.real(fft_data)[:7], height = 2e-3)
    real_peak_freq_n = frequencies[peaks_r_n[0][0]]
    real_peak_height_n = -peaks_r_n[1]['peak_heights'][0]

    fit_func = exp

    p0r = (real_peak_height, real_peak_freq,-1)
    popt_r, pcov_r = curve_fit(fit_func, frequencies[peaks_r[0][0]:], np.real(fft_data)[peaks_r[0][0]:],\
        p0 = p0r, bounds = ((real_peak_height,real_peak_freq,-np.inf),(real_peak_height*1.001,real_peak_freq*1.001,0)))

    real_fit = fit_func(frequencies[peaks_r[0][0]:], *popt_r)

    ax[0].plot(frequencies[peaks_r[0][0]:], real_fit)
    ax[0].plot(real_peak_freq_n, real_peak_height_n, 'ro', label=f'neg peak {real_peak_freq_n}')

    # fitting()

    # ax[1].plot(imag_peak_freq_n, imag_peak_height_n, 'ro', label=f'neg peak {imag_peak_freq_n}')
    # ax[1].plot(frequencies[peaks_i[0][0]:], imag_fit)
    ax[1].plot(frequencies, fft_sine_data,label=comps[i]) # , label=f'{frequencies[peaks_i[0]]} {peaks_i[1]}'

    ax[1].set_xlabel('Frequency')
    ax[1].set_title(f'Imaginary comp. of Fourier Transform of Actual - Quadratic global correction  \n till lambda = {max_lam_err} @ n={exponent}\
        for {get_element_symbol(atomic_number)}-{get_element_symbol(atomic_number)}')
    ax[0].set_title(f'Real comp. of Fourier Transform of Actual - Quadratic global correction  \n till lambda = {max_lam_err} @ n={exponent} ')
    ax[0].legend()
    ax[1].legend()
    fig.text(0.04, 0.5, 'amplitude', va='center', rotation='vertical',size=20)
    plt.show()

    return real_peak_height_n,real_peak_freq_n, *popt_r, np.real(normalization * fft.fftshift(fft.fft(data)))


'''Fitting quarttic error'''
def perform_quart_err_correction():
    quart_err = dft_pred_restricted - quad_adjusted_pre_restriced
    fits = []
    for i in range(3):
        popt_, pcov_ = curve_fit(quart_fit, x_axis_err, quart_err[i],absolute_sigma=True)
        fitted_err_ = np.array(quart_fit(x_axis, *popt_))
        fits.append(fitted_err_)

    quart_adjusted_prediction = quad_adjusted_prediction +np.array(fits)
    quart_adjusted_pre_restriced = np.array([quart_adjusted_prediction[i][:max_lam_err*int(1/step) + 1] for i in range(3)])

    plot_pol_errors(quad_adjusted_prediction, quart_adjusted_prediction,'quartic')

def plot_predictions():
    '''Plotting graph with linear and non-linear grads for symmetrical alchemical changes'''
    l_vals = np.array([np.array(x) for x in [pre_NN_l,pre_CO_l,pre_BF_l]])
    l_keys = ['L. Z @ NN', 'L. Z @ CO','L. Z @ BF']
    nl_keys = ['NL. Z @ NN','NL. Z @ CO','NL. Z @ BF']

    l_lines = dict(zip(l_keys, l_vals))
    nl_lines = dict(zip(nl_keys,prediction_7_3))
    all_lines = {**l_lines, **nl_lines}

    format = ['r:','g:', 'b:','r--','g--', 'b--']

    # figure(figsize=(8, 6), dpi=80)
    fig, ax = plt.subplots()
    ax.scatter([0,1,2,3,4,5,6,7],[e_NN, e_CO, e_BF, e_BeNe, e_LiNa, e_HeMg, e_HAl, e_Si],label='Comps.')
    ax.plot(x_axis,np.array(frac_energies),'black',label='Actual curve')
    # ax.plot(x_axis,np.array(R_P),'brown',label='R. protonated curve')
    # ax.plot(x_axis,np.array(L_P),'grey',label='L. protonated curve')

    i = 0
    for line in all_lines:
        # plt.scatter(np.linspace(0,2,len(lines[line])),lines[line],label=line)
        plt.plot(x_axis,all_lines[line],format[i],label=line)
        i +=1
    ax.legend()
    ax.set_ylabel('Total Elec. enegy in Ha.')
    ax.set_xlabel(r' ($\Delta \lambda$)')
    ax.set_xlim(0,7)
    ax.set_title('Predictions with a quadratic error adjusted')

    # add another x axis in plot
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels(['NN','CO','BF','BeNe','LiNa','HeMg','HAl','Si'])
    plt.show()
    plt.savefig('data/figs/main_prediction_plot.png')

def finite_fourier_fit():
    # fitting the eror after adjusting quadratic error.
    err_to_fit = dft_pred_restricted - quad_adjusted_pre_restriced

    def get_fourier_pred(N):
        '''Gets the lambda dependent fourier prediction upto Nth order for the error after adjusting the quadratic error.'''
        fits = []
        funcs = [ff1, ff2, ff3, ff4,ff5]
        base = (3.977507e-02,  0.28, -1.20482758e+01,  1.16835373e-01)
                # 5.40220558e-03)
        for i in range(1,N):
            base += (3.977507e-02,  0.28, -1.20482758e+01,  1.16835373e-01)
                # 5.40220558e-03)
        popts = []
        for j in range(3):
            popt, pcov = curve_fit(funcs[N-1], x_axis_err, err_to_fit[j],p0=base,absolute_sigma=True,maxfev=10000000)
            fits.append(funcs[N-1](x_axis,*popt))
            # print(f'b is {popt[0]} for {comps[j]}')
            popts.append(np.array(popt))
        return quad_adjusted_prediction + np.array(fits), np.array(popts)

    for i in range(5):
        popts = get_fourier_pred(i+1)[1]
        plt.scatter(np.linspace(1,i+1,i+1),np.abs(popts[0][1::4]), label = f'{i+1}th order fourier terms for N2 @ n=2')
        # print(f'popt for {i+1}th order is {popts}')
        # get every 5th value of the popt and plot it against the order in one line
        # plot each point in popts[0][1::5] as scatter plot with individual labels
        print(i,len(popts[0][1::4]))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    # to check whether the fits accurately model the error
    for i in range(3):
        plt.scatter(x_axis, frac_energies - quad_adjusted_prediction[i], label = f'quad err for {comps[i]}')
        plt.plot(x_axis, (get_fourier_pred(1)[0]-quad_adjusted_prediction)[i], label = '1 fourier term')
        plt.plot(x_axis, (get_fourier_pred(2)[0]-quad_adjusted_prediction)[i], label = '2 fourier term')
        plt.plot(x_axis, (get_fourier_pred(3)[0]-quad_adjusted_prediction)[i], label = '3 fourier term')
        # plt.plot(x_axis, ff(x_axis, *[ 2.19714862e-02,  1.70264763e+00 ,-11.10,  1.07301660e+00,7.97399174e-03]),label='Real')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Lambda')
    plt.ylabel('Error in Ha.')
    plt.title('Fitting quad fit error')
    # get legend out of the plot
    plt.show()

def all_models_all_errs():
    '''PLotting all the errors for the different models'''
    comps = ['NN', 'CO', 'BF']
    error_funcs = [rmse, max_error, mae, std]
    models = [quad_adjusted_prediction, \
        quart_adjusted_prediction, get_fourier_pred(1)[0], get_fourier_pred(2)[0]]
    labels = ['quad adj.', 'quart adj.', 'fourier 1', 'fourier 2']
    all_errors = []
    fig, (ax1, ax2,ax3, ax4) = plt.subplots(4,1, figsize = (15,15))
    axes = [ax1, ax2, ax3, ax4]
    for f in range(len(error_funcs)):
        for model in range(len(models)):
            err = []
            for i in range(3):
                # if labels[model] not in ['n = 2 adjusted quad. err','fourier 1', 'fourier 2']:
                err.append(error_funcs[f](frac_energies, models[model][i]))
                # else:
                #     err.append(error_funcs[f](frac_energies[:61], models[model][i]))

            # changing to milli hartree
            err = np.array(err)*1000
            axes[f].scatter(comps,err, label=f'{labels[model]}')
            axes[f].set_ylabel(f'{error_funcs[f].__name__} (milli Ha.)')
            axes[f].set_title(f'{error_funcs[f].__name__} for different models')
            # shift legend outside of plot
            box = axes[f].get_position()
            axes[f].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            axes[0].legend()
            # axes[f].legend()
    plt.savefig('data/figs/all_models_err.png')
    # fig.suptitle(r'Errors for different n where $Z(\lambda) = (Z_i^n + \lambda (Z_f^n-Z_i^n))^{\frac{1}{n}}$', fontsize=25)

def final_err():
    x_axis = np.linspace(0,max_lam,steps)
    err = np.abs(frac_energies - get_fourier_pred(3)[0])
    for i in range(3):
        plt.plot(x_axis,err[i],label=f'for {comps[i]} pred')
    plt.xlabel('Lambda')
    plt.ylabel('Absolute error')
    plt.title('Error after fitting 3 fourier terms and a quadratic term \n to n=2 global fitting till lambda = 7')
    plt.legend()
    plt.savefig('data/figs/3_fourier_terms.png')
    # create numpy array of same value
    # print(f'MAE errors for 3 fourier terms {mae(frac_energies,get_fourier_pred(3)[0]),mae(frac_energies,get_fourier_pred(3)[1]),mae(frac_energies,get_fourier_pred(3)[2])}')
    plt.show()