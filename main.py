import sys, os
sys.path.append('/Users/meet/Desktop/Courses/Research/Chem/Code/QA/g_modules')
sys.path.append('/Users/meet/Desktop/Courses/Research/Chem/Code/QA/m_modules')
# Add the parent directory to the system path
sys.path.append('/Users/meet/Desktop/Courses/Research/Chem/Code/QA')

from pyscf import gto,scf, cc
import numpy as np
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
import scipy
import pandas as pd



def init_alc(atomic_number, exponent, frac_energies,
             mol_objs: list, args):

    max_d_lam, steps, step, exponent, d = args
    # energy.max_d_lam, energy.steps, energy.step,\
    #     energy.exponent, energy.d = max_d_lam, steps, step, exponent, d

    # collecting molecule, converged object, electronic energy into arrays

    mols = mol_objs[0::3]
    mfs = mol_objs[1::3]
    e_mol = mol_objs[2::3]
    '''Data for max_lambda = 5 and = 7 is stored in data folder'''

    '''getting / generating data for protonation'''

    prediction =  []
    lin_pred = []

    for i in range(len(mols)):
        AG_mol = AG(mfs[i]) # the alchemical gradient i.e. d_E / d_Z_I for where Z_I are N atoms
        # Evaluating the linearized energy gradient at lambda = 0. See argument of d_Z_lambda in get_pred function in energy.py
        pre_l, pre_nl = gen_data(mols[i],AG_mol,e_mol[i],-i, max_d_lam-i, args)
        prediction.append(pre_nl)
        lin_pred.append(pre_l)

    # for ghost_Atom prediction
    # NN = gto.M(atom= f"{atomic_number*2} 0 0 0; ghost_{get_element_symbol(7)} 0 0 {d}",unit="Bohr",basis='unc-ccpvdz') # uncontracted cc-pvdz
    # mol, mf_mol, elec_energy = new_mol(NN, 0,0)
    # pre_l, pre_nl = gen_data(mol,AG(mf_mol),elec_energy,-7, max_d_lam-7, args)
    # prediction.append(pre_nl)
    # lin_pred.append(pre_l)

    '''
    Preparing prediction data for plotting for symmetrical alchemical changes for n in (0.5, 3) in steps of 0.1'''
    '''
    Structure of predictions_n:
        predictions_n = [...,[pre_NN_nl_n, pre_CO_nl_n, pre_BF_nl_n],...]
        pre_NN_nl_n = non-linear Z prediction over all lambda at n from NN
    '''

    # fitting errors only upto a certain lambda
    max_lam_err = max_d_lam
    # store difference between numpy arrays of size (51,) and (3,71) in a numpy array of size (3,51) in one line

    x_axis = np.linspace(0,max_d_lam,steps)

    # alc_fitting(atomic_number, frac_energies, step, max_lam_err, x_axis, prediction)
    # l_nl_plot(atomic_number, exponent, d, frac_energies, pre_NN_l, pre_CO_l, pre_BF_l, prediction, x_axis)

    return prediction, x_axis

def init_sep(Z1,Z2):
    min = 0.1
    max = 10
    stepsize = 0.1
    from_ = .1
    to_ = 6

    atoms = np.array([Z1,Z2])
    sep = np.arange(min, max+stepsize, stepsize)
    data_total = gen_sep_data(atoms[0],atoms[1], min, max, stepsize) # total energy curve
    data = data_total - np.prod(atoms) / sep # gives total electronic energy curve

    # restricting data
    data_total = data_total[(sep > from_) & (sep < to_)]
    data = data[(sep > from_) & (sep < to_)]
    sep = sep[(sep > from_) & (sep < to_)]
    exponent = 2.0

    # fitting
    C = -np.sum(atoms**exponent)

    popt_, pcov_ = curve_fit(pol_fit, sep, data,absolute_sigma=True,p0 = (-50, 0.9,C),maxfev=100000)
    fit_ = pol_fit(sep, *popt_)
    fitted_exponent = round(popt_[1],3)

    d_i = 2.07
    elec_equi, elec_grad, tot_grad = get_mol_energy_grad(atoms[0],atoms[1],d_i)
    gradient = elec_grad[1] # d E / d sep --- keeping left atom fixed and shifting right atom

    prediction = elec_equi + gradient*d_S_gamma(d_i, sep, fitted_exponent)

    # new_prediction = prediction + (data[-1] - prediction[-1])

    # get_values_uc(pol_fit, popt_, pcov_)
    # try:
    min_index, min_val = find_min(data_total)
    r_e = sep[min_index]
    D_e = data_total[-1] - min_val
    a = 1.42308
    index, hts = find_peaks(-data_total, height = -min_val)
    morse, morse_params = fit_me([sep, data_total], morse_potential, [sep[index[0]], D_e, a, data_total[-1]])
    a = morse_params[2]
    morse_potential_elec_fit = morse - np.prod(atoms) / sep

    # fourier transforming
    ft_i = reflect(data - prediction)
    N = len(ft_i)  # Number of data points
    normalization = 1 #/ (np.sqrt(2*np.pi))
    fft_data = normalization * fft.rfft(ft_i)#[3:]
    real_fft_data = np.real(fft_data)
    frequencies = fft.rfftfreq(N, d = stepsize)#[3:]

    # plotting peaks in fft
    peak_indices, peak_dic = find_peaks(real_fft_data,height= 0)
    peak_freqs = frequencies[peak_indices]
    peaks = real_fft_data[peak_indices]

    # frequency of the frequencies
    f_ = 1 / np.diff(peak_freqs)[0]

    model_x = frequencies[2:] #[peak_indices[1]:]
    model_y = real_fft_data[2:] #[peak_indices[1]:]
    # fitting damped harmonic
    damp_har, damp_har_params = fit_me([model_x,model_y ], damped_harmonic, [model_y[0],-0.1,f_,model_x[0]])
        #* d_S_gamma_morse(d_i, sep, a)
    #d_S_gamma(d_i, sep, fitted_exponent)
    modelled_err = fft.irfft(damp_har)

    fig, ax = plt.subplots(4, 1, figsize=(10, 20), sharex=False)

    # plotting
    ax[0].scatter(sep, data, label = 'dft calcs')
    ax[0].plot(sep, fit_, label=f'fit = A * d^(-{fitted_exponent}) + C, C = free atom elec energy')
    ax[0].plot(sep, prediction, label='pred')
    # ax[0].plot(sep, new_prediction, label='pred + C')
    ax[0].plot(sep, morse_potential_elec_fit, label='Morse - V_NN')

    ax[1].scatter(frequencies, real_fft_data, label='real fft')
    ax[1].scatter(frequencies, np.imag(fft_data), label='imag fft')
    ax[1].scatter(peak_freqs, peaks, label='peaks')
    ax[1].plot(model_x, damp_har, label='damped harmonic')
    ax[1].set_title(f'Fourier transform of error for {get_mol_symbol((Z1+Z2)/2,(Z1-Z2)/2,(Z2-Z1)/2)}')


    ax[2].plot(sep, data - prediction, label=f'{get_mol_symbol((Z1+Z2)/2,(Z1-Z2)/2,(Z2-Z1)/2)}')
    ax[2].plot(sep, data - morse_potential_elec_fit, label=f'Morse error')
    # ax[2].plot(modelled_err, label=f'Modelled error')

    ax[3].plot(sep, data_total, label=f'Actual Total')
    ax[3].plot(sep, morse, label=f'Morse ')



    ax[0].set_title(f'Energy vs sep {get_mol_symbol((Z1+Z2)/2,(Z1-Z2)/2,(Z2-Z1)/2)}')
    ax[0].set_xlabel("Separation (Bohr); delta = 0.1 Bohr")
    ax[0].set_ylabel("Energy (Hartree)")

    ax[2].set_title(f'Error for {get_mol_symbol((Z1+Z2)/2,(Z1-Z2)/2,(Z2-Z1)/2)}. MAE = {round(mae(data, prediction),3)}')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()

    # except:
    #     pass
    return np.array(data), np.array(prediction)


def FT(actual, new_prediction, max_lam_err, comps, atomic_number):
    err = actual - new_prediction
    subplots = 1
    fig, ax = plt.subplots(subplots, 1, figsize=(2*5*subplots, 5*subplots), sharex=True)
    imag_peak_freqs = []
    imag_peak_heights = []

    # put 1 to only plot error from equilibrium diatomic.
    # for i in range(1):
    i = 0
    data = err[i][:-1]
    N = len(data)  # Number of data points
    normalization = 1 / (np.sqrt(2*np.pi))   # Normalization factor
    fft_data = np.real(normalization * fft.rfft(data))
    fft_sine_data = fft.fftshift(normalization * scipy.fft.dst(data))[N//2:]
    fft_cos_data = fft.fftshift(normalization * scipy.fft.dct(data))[N//2:]

    # Compute the frequencies
    frequencies = (fft.rfftfreq(N, d = 1))
    if np.sum(np.imag(normalization * fft.rfft(data))) < 0.0001:
        print('Imaginary part is negligible')

    # plot only half of the frequencies
    # frequencies = frequencies[N//2:]
    # fft_data = fft_data[N//2:]
    ax.scatter(frequencies[:50], fft_data[:50],label=comps[i]) #,  label=f'{frequencies[peaks_r[0]]} {peaks_r[1]}'
     # # finding peaks for feeding curve-fit parameters
    # positive peaks
    # def fitting():
    peaks_r = find_peaks(fft_data, height = 0.2)
    real_peak_freq = frequencies[peaks_r[0][0]]
    real_peak_height = peaks_r[1]['peak_heights'][0]
    # negative peaks

    peaks_r_n = find_peaks(-fft_data[:10], height = 2e-3)
    real_peak_freq_n = frequencies[peaks_r_n[0][0]]
    real_peak_height_n = -peaks_r_n[1]['peak_heights'][0]

    fit_func = exp

    p0r = (real_peak_height, real_peak_freq,-1)
    popt_r, pcov_r = curve_fit(fit_func, frequencies[peaks_r[0][0]:], fft_data[peaks_r[0][0]:],\
        p0 = p0r, bounds = ((real_peak_height,real_peak_freq,-np.inf),(real_peak_height*1.001,real_peak_freq*1.001,0)))

    real_fit = fit_func(frequencies[peaks_r[0][0]:], *popt_r)

    ax.plot(frequencies[peaks_r[0][0]:], real_fit)
    ax.plot(real_peak_freq_n, real_peak_height_n, 'ro', label=f'neg peak {real_peak_freq_n}')

    # fitting()

    ax.set_title(f'Real comp. of Fourier Transform of first order correction  \n till lambda = {max_lam_err} @ n={exponent} ')


    fig.text(0.04, 0.5, 'amplitude', va='center', rotation='vertical',size=20)


    # preparing a new fft for checking
    array = np.zeros(len(fft_data))
    array[peaks_r_n[0][0]] = real_peak_height_n
    array[peaks_r[0][0]:] = real_fit # fft_data[peaks_r[0][0]:]
    # print(array)
    # ax.scatter(frequencies, array,label='Array fft')
    ax.legend()
    plt.show()
    return real_peak_height_n,real_peak_freq_n, *popt_r, fft.irfft(array), fft.irfft(fft_data)

'''Fitting quarttic error'''
def perform_quart_err_correction():
    quart_err = dft_restricted - quad_adjusted_pre_res
    fits = []
    for i in range(3):
        popt_, pcov_ = curve_fit(quart_fit, x_axis_err, quart_err[i],absolute_sigma=True)
        fitted_err_ = np.array(quart_fit(x_axis, *popt_))
        fits.append(fitted_err_)

    quart_adjusted_prediction = quad_adjusted_pre +np.array(fits)
    quart_adjusted_pre_restriced = np.array([quart_adjusted_prediction[i][:max_lam_err*int(1/step) + 1] for i in range(3)])

    plot_pol_errors(quad_adjusted_pre, quart_adjusted_prediction,'quartic')

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
    err_to_fit = dft_restricted - quad_adjusted_pre_res

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
        return quad_adjusted_pre + np.array(fits), np.array(popts)

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
        plt.scatter(x_axis, frac_energies - quad_adjusted_pre[i], label = f'quad err for {comps[i]}')
        plt.plot(x_axis, (get_fourier_pred(1)[0]-quad_adjusted_pre)[i], label = '1 fourier term')
        plt.plot(x_axis, (get_fourier_pred(2)[0]-quad_adjusted_pre)[i], label = '2 fourier term')
        plt.plot(x_axis, (get_fourier_pred(3)[0]-quad_adjusted_pre)[i], label = '3 fourier term')
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
    models = [quad_adjusted_pre, \
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
    x_axis = np.linspace(0,max_d_lam,steps)
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


def calc_energy(perturbation_tuple):
    atomic_number = 7
    delta_lambda, delta_gamma = perturbation_tuple

    mol = gto.M(atom= f"{atomic_number} 0 0 0; {atomic_number} 0 0 {delta_gamma}",unit="Bohr",basis='unc-ccpvdz')

    mol = FcM_like(mol,fcs=[-delta_lambda,delta_lambda])

    mf_mol=scf.RKS(mol)
    mf_mol.xc="PBE0"
    mf_mol.verbose = 0
    mf_mol.scf(dm0=mf_mol.init_guess_by_1e())
    print(f"{perturbation_tuple} DONE ",flush=True)
    sys.stdout.flush()
    return perturbation_tuple, round(mf_mol.energy_elec()[0],3)

def atomic_bomb(atomic_number):
    '''Returns you all errors for all exponents for all 3 delta lambdas'''
    print(f'starting {atomic_number}')
    step = 0.1
    max_d_lam = atomic_number
    steps =  int(atomic_number / step  + 1)
    d = 2.1
    args = [max_d_lam, steps, step, d ]

    exp_dic = {n: [] for n in np.round(np.arange(1.5, 2.5, 0.001),3)}

    dft = get_symmetric_change_data(0,atomic_number,atomic_number, args)[0] # dft electronic energues for the correspoing electronic series
    mol_objs = tri_party_mol_props(atomic_number, d) # mol_objs for predictions for the corresponding electronic series

    for exponent in list(exp_dic.keys()):
        a = args.copy()
        a.insert(-1, exponent)
        prediction, x_axis = init_alc(atomic_number, exponent, \
            dft, mol_objs, a)

        all_errs = []
        for p in prediction:
            # all errors for prediction at specific d_lambda
            all_errs_p = np.round(np.array(all_error(dft, p, x_axis)),4)
            all_errs.append(list(all_errs_p))

        exp_dic[exponent] = all_errs

    return exp_dic
