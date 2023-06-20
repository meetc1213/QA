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
from m_modules.energy import *
from m_modules.config import *
import json
# uncontracted cc-pvdz
d = 2.1
NN = gto.M(atom= f"N 0 0 0; N 0 0 {d}",unit="Bohr",basis='unc-ccpvdz')
# with open('objects.json', 'w') as f:
#     f.write(json.dumps([new_mol(NN, 7/3, 0,0),new_mol(NN, 7/3, -1,-1),new_mol(NN, 7/3, -2,-2), \
#         new_mol(NN, 7/3, -3,-3),new_mol(NN, 7/3, -4,-4),new_mol(NN, 7/3, -5,-5),\
#             new_mol(NN, 7/3, -6,-6),new_mol(NN, 7/3, -7,-7)]))
# with open('objects.pkl', 'rb') as f:
#     mol_NN, mol_CO, mol_BF, mol_BeNE, mol_LiNa, mol_HeMg, mol_HAl, mol_Si = pickle.load(f)

mf_NN, e_NN = new_mol(NN, 7/3, 0,0)[1],new_mol(NN,7/3, 0,0)[2]
CO, mf_CO, e_CO = new_mol(NN, 7/3, -1,1)[:3]
BF, mf_BF, e_BF = new_mol(NN, 7/3, -2,2) [:3]

e_BeNe, e_LiNa,e_HeMg, e_HAl, e_Si = new_mol(NN,7/3,-3,3)[2],new_mol(NN,7/3,-4,4)[2], new_mol(NN,7/3,-5,5)[2],\
                                    new_mol(NN,7/3,-6,6)[2],new_mol(NN,7/3,-7,7)[2]
# write above data to the pickle file objects.pkl



'''Data for max_lambda = 5 and = 7 is stored in data folder'''
## Calculating actual fractional charge energies
max_lam, steps, step = return_max_lam_steps() # ensures a 0.1 step size
gen = False
data = get_symmetric_change_data(0,max_lam,gen)
frac_energies,free_energies =  data[0], data[1]

'''getting / generating data for protonation'''
s = 0.94 # 0.5 angstroms or 0.94 Bohr.
gen = False
data = get_asymmetric_change_data(0,max_lam, gen)
R_P,L_P = data[0],data[1]

# Evaluating the linearized energy gradient at lambda = 0
AG_NN = AG(mf_NN) # the alchemical gradient i.e. d_E / d_Z_I for where Z_I are N atoms
AG_CO = AG(mf_CO)
AG_BF = AG(mf_BF)


pre_NN_l, pre_NN_nl = gen_data(NN,AG_NN,7/3,e_NN,0, max_lam)
pre_CO_l, pre_CO_nl = gen_data(CO,AG_CO,7/3,e_CO,-1, max_lam -1)
pre_BF_l, pre_BF_nl = gen_data(BF,AG_BF,7/3,e_BF,-2, max_lam -2)
prediction_7_3 = np.array([pre_NN_nl, pre_CO_nl, pre_BF_nl])

'''
Preparing prediction data for plotting for symmetrical alchemical changes for n in (0.5, 3) in steps of 0.1'''
'''
Structure of predictions_n:
    predictions_n = [...,[pre_NN_nl_n, pre_CO_nl_n, pre_BF_nl_n],...]
    pre_NN_nl_n = non-linear Z prediction over all lambda at n from NN
'''
predictions_n = []
for n in np.linspace(0.5,5,46):
    pre_NN_nl_n = gen_data(NN,AG_NN,n,e_NN,0, max_lam)[1]
    pre_CO_nl_n = gen_data(CO,AG_CO,n,e_CO,-1, max_lam -1)[1]
    pre_BF_nl_n = gen_data(BF,AG_BF,n,e_BF,-2, max_lam -2)[1]
    predictions_n.append(np.array([pre_NN_nl_n, pre_CO_nl_n, pre_BF_nl_n]))
predictions_n = np.array(predictions_n)
prediction_2 = predictions_n[15]

# fitting errors only upto a certain lambda
max_lam_err = 7
# store difference between numpy arrays of size (51,) and (3,71) in a numpy array of size (3,51) in one line
prediction = prediction_2
pre_2_restricted = np.array([ prediction[i][:max_lam_err*10 + 1] for i in range(3)])
dft_pred_restricted = np.array([ frac_energies[:max_lam_err*10 + 1] for i in range(3)])
'''Defining the total x_axis'''
x_axis = np.linspace(0,max_lam,steps + 1)

'''Defining the new x axis for fitting errors'''
x_axis_err = x_axis[:max_lam_err*10 + 1]

'''Fitting away 2nd order errors'''
quad_err = dft_pred_restricted - pre_2_restricted
# quad_err = dft_pred_restricted - prediction_7_3[:,:max_lam_err*10 + 1]
err_fits = []
for i in range(3):
    popt_, pcov_ = curve_fit(quad_fit, x_axis_err, quad_err[i],absolute_sigma=True,maxfev=100000)
    fitted_err_ = np.array(quad_fit(x_axis, *popt_))
    err_fits.append(fitted_err_)

quad_adjusted_prediction = prediction + np.array(err_fits)
quad_adjusted_pre_restriced = np.array([quad_adjusted_prediction[i][:max_lam_err*10 + 1] for i in range(3)])

# plotting error between actual and prediction
comps = ['NN','CO','BF']
def plot_quad_errors():
    # SHOULD PLOT BOTH INITIAL ERROR AND ERROR AFTER FITTING AWAY QUADRATIC ERRORS
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for i in range(3):
        # plt.scatter(x_axis, quart_err[i], label = 'actual err')
        # plt.scatter(x_axis_err,(dft_pred_restricted-quad_adjusted_pre_restriced)[i], label = 'quad adjusted err')
        ax[0].scatter(x_axis, frac_energies - prediction[i], label = f'err for {comps[i]} pre')
        ax[0].plot(x_axis, err_fits[i],label = 'quad err fit')
        # ax[1].scatter(x_axis, quart_err[i], label = 'actual err')
        # ax[1].scatter(x_axis_err,(dft_pred_restricted-quad_adjusted_pre_restriced)[i], label = i)
        ax[1].scatter(x_axis, frac_energies - quad_adjusted_prediction[i], label = f'{comps[i]} quad corrected pre.')
    ax[0].legend()
    # ax[0].set_ylabel('dft - prediction in Ha.')
    ax[0].set_title('Error between actual and prediction')
    ax[1].set_title('Error after adjusting for quadratic error')
    ax[1].set_xlabel('lambda')
    ax[1].legend()
    fig.text(0.04, 0.5, 'Error in Ha.', va='center', rotation='vertical',size=20)
    plt.show()

'''Fititng quarttic error'''
quart_err = dft_pred_restricted - quad_adjusted_pre_restriced
fits = []
for i in range(3):
    popt_, pcov_ = curve_fit(quart_fit, x_axis_err, quart_err[i],absolute_sigma=True)
    fitted_err_ = np.array(quart_fit(x_axis, *popt_))
    fits.append(fitted_err_)

quart_adjusted_prediction = quad_adjusted_prediction +np.array(fits)
quart_adjusted_pre_restriced = np.array([quart_adjusted_prediction[i][:max_lam_err*10 + 1] for i in range(3)])

def plot_quartic_errors():
    # plotting error between actual and quad adjusted
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    for i in range(3):
        # plt.scatter(x_axis, quart_err[i], label = 'actual err')
        # plt.scatter(x_axis_err,(dft_pred_restricted-quart_adjusted_pre_restriced)[i], label = 'quart adjusted err')
        ax[0].scatter(x_axis, frac_energies - quad_adjusted_prediction[i], label = f'err for {comps[i]} pre')
        ax[0].plot(x_axis, quart_adjusted_prediction[i] - quad_adjusted_prediction[i], label = f'fits for {comps[i]} err')
        ax[1].scatter(x_axis, frac_energies - quart_adjusted_prediction[i], label = f'for {comps[i]}')
    ax[0].set_title('Error before adjusting for quartic error')
    ax[1].set_title('Error after adjusting for quartic error')
    ax[0].legend()
    ax[1].legend()
    ax[1].set_xlabel(r'$\lambda$')
    fig.text(0.04, 0.5, 'Error in Ha.', va='center', rotation='vertical',size=20)
    plt.show()

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

def FT():
    err = dft_pred_restricted - quad_adjusted_pre_restriced
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for i in range(3):
        data = err[i]
        N = len(data)  # Number of data points
        normalization = 2 / N
        fft_data = normalization * np.fft.fft(data)
        # Compute the frequencies
        frequencies = np.fft.fftfreq(N, d = step)

        # Shift the frequencies to have 0 at the center
        frequencies = np.fft.fftshift(frequencies)[N//2:]
        fft_data = np.fft.fftshift(fft_data)[N//2:]

        # finding peaks for feeding curve-fit parameters
        peaks_r = find_peaks(np.real(fft_data), height = 0.05)
        peaks_i = find_peaks(np.imag(fft_data), height = (0.10,0.15))

        # fitting the fourier transforms
        # popt_r, pcov_r = curve_fit(x_n, frequencies, np.real(fft_data)[:], p0 = [0.1, 0.1, 0.1, 0.1])
        # Plot the Fourier transform
        ax[0].plot(frequencies, np.real(fft_data),  label=f'{frequencies[peaks_r[0]]} {peaks_r[1]}')
        ax[1].plot(frequencies, np.imag(fft_data), label=f'{frequencies[peaks_i[0]]} {peaks_i[1]}')

    ax[1].set_xlabel('Frequency')
    ax[1].set_title('Imaginary comp. of Fourier Transform of Actual - Quadratic global correction  \n till lambda = 7 @ n=2 with freqs 0.14 and 0.42')
    ax[1].legend()
    ax[0].set_title('Real comp. of Fourier Transform of Actual - Quadratic global correction  \n till lambda = 7 @ n=2 with freqs 0.14 and 0.42')
    ax[0].legend()
    fig.text(0.04, 0.5, 'amplitude', va='center', rotation='vertical',size=20)
