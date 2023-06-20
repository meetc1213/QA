'''Contains some plot code for errors with different exponents n and such'''
import numpy as np
import pyscf
import matplotlib.pyplot as plt

def n_err():
    # plotting error between frac_energies and predictions_n w.r.t lambda
    # Suggests that we should decrease the power to about 2.0 corresp to k = 15
    fig, (ax1, ax2,ax3) = plt.subplots(3,1, figsize = (10,10))
    x_axis = np.linspace(0,max_lam,steps + 1)
    for k in range(len(predictions_n)):
        if k == 15:
            ax1.plot(x_axis,frac_energies - predictions_n[k][0],'b--',label=f'n = {round(0.5 + k*0.1,2)}')
            ax2.plot(x_axis,frac_energies - predictions_n[k][1],'b--',label=f'n = {round(0.5 + k*0.1,2)}')
            ax3.plot(x_axis,frac_energies - predictions_n[k][2],'b--',label=f'n = {round(0.5 + k*0.1,2)}')
        else:
            ax1.plot(x_axis,np.abs(frac_energies - predictions_n[k][0]),label=f'n = {round(0.5 + k*0.1,2)}')
            ax2.plot(x_axis,np.abs(frac_energies - predictions_n[k][1]),label=f'n = {round(0.5 + k*0.1,2)}')
            ax3.plot(x_axis,np.abs(frac_energies - predictions_n[k][2]),label=f'n = {round(0.5 + k*0.1,2)}')

    # plotting the n = 7/3 quad errors

    ax1.plot(x_axis,np.abs(frac_energies - prediction_7_3[0]),label=f'n = {round(7/3,2)}', color = 'black', linewidth = 2)
    ax2.plot(x_axis,np.abs(frac_energies - prediction_7_3[1]),label=f'n = {round(7/3,2)}', color = 'black', linewidth = 2)
    ax3.plot(x_axis,np.abs(frac_energies - prediction_7_3[2]),label=f'n = {round(7/3,2)}', color = 'black', linewidth = 2)
    new_x_axis = x_axis[:61]
    errs = frac_energies - predictions_n[15]
    axes = [ax1, ax2, ax3]
    n_2_fit_err = []
    for i in range(3):
        popt_, pcov_ = curve_fit(quad_fit, new_x_axis, errs[i][:61],absolute_sigma=True)
        fitted_err_ = np.array(quad_fit(new_x_axis, *popt_))
        n_2_fit_err.append(fitted_err_)
        axes[i].plot(new_x_axis, fitted_err_, label = f'n = {round(0.5 + 15*0.1,2)} fit till l = 6', color = 'black', linewidth = 2)
    n_2_quad_adj = n_2_fit_err + predictions_n[15][:,:61]
    plt.xlabel("lambda")
    plt.ylabel("Absolute error")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

    '''plotting actual err between n_2_quad_adj and frac_energies'''
    plt.figure(figsize = (10,10))
    x_axis = np.linspace(0,max_lam,steps + 1)
    new_x_axis = x_axis[:61]
    for i in range(3):
        p0 = [0.00292706, -0.04284348,  0.23319628, -0.58398455,  0.63842687, -0.18881309]
        popt, pcov = curve_fit(quintic_fit, new_x_axis, frac_energies[:61] - n_2_quad_adj[i],absolute_sigma=True)
        plt.plot(new_x_axis, quintic_fit(new_x_axis, *p0))
        plt.plot(new_x_axis, quintic_fit(new_x_axis, *popt), label = f'n = {round(0.5 + 15*0.1,2)} fit till l = 6', color = 'black', linewidth = 1)
        plt.scatter(new_x_axis, frac_energies[:61] - n_2_quad_adj[i])
    plt.legend()

def diff_err_n():
    '''PLotting different kinds of errors w.r.t. to n'''
    error_funcs = [rmse, max_error, mae, std]
    all_errors = []
    for f in error_funcs:
        error = []
        for k in range(len(predictions_n)):
            # k accesses the predictions related to n = 0.5 + k*0.1
            # 0, 1, 2 accesses the NN, CO, BF predictions respectively
            # store errors in a list
            error.append(f(frac_energies,predictions_n[k][0]))
            error.append(f(frac_energies,predictions_n[k][1]))
            error.append(f(frac_energies,predictions_n[k][2]))
        all_errors.append(np.array(error))

    # plot errors
    fig, (ax1, ax2,ax3, ax4) = plt.subplots(4,1, figsize = (15,15))
    n_axis = np.linspace(0.5,5,46)
    axes = [ax1, ax2, ax3, ax4]
    for i in range(len(axes)):
        axes[i].scatter(n_axis,all_errors[i][0::3],label='from NN')
        axes[i].scatter(n_axis,all_errors[i][1::3],label='from CO')
        axes[i].scatter(n_axis,all_errors[i][2::3],label='from BF')
        axes[0].legend()
        axes[len(axes)-1].set_xlabel("n")
        axes[len(axes)-1].xaxis.label.set_size(20)
        axes[i].set_ylabel(error_funcs[i].__name__+' in Ha.')
        axes[i].yaxis.label.set_size(20)
    # set title for whole figure
    # reduce space between title and first subplot
    plt.subplots_adjust(top=0.94,hspace=0.1)
    # fig.tight_layout()  # Adjusts the spacing between subplots
    # fig.set_size_inches(8, 6)  # Set the width and height of the figure in inches
    fig.suptitle(r'Errors for different n where $Z(\lambda) = (Z_i^n + \lambda (Z_f^n-Z_i^n))^{\frac{1}{n}}$', fontsize=25)
    plt.savefig('data/figs/errors_n.png',bbox_inches='tight',dpi=300)