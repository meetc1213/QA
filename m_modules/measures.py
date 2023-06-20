'''Contains the various measures to compare different prediction models with plotting code'''

def delta_1():
    comps = ['NN', 'CO', 'BF']
    models = [quad_adjusted_prediction, quart_adjusted_prediction, \
        get_fourier_pred(1), get_fourier_pred(2), n_2_quad_adj,predictions_n[15]]
    labels = ['quad adj.', 'quart adj.', 'fourier 1', 'fourier 2',\
        'n = 2 adjusted quad. err','n = 2.0 for free E']
    for i in range(3):
        figure(figsize=(8, 8), dpi=180)
        x_axis = np.array(frac_energies - free_energies)

        plt.plot(x_axis, frac_energies - free_energies,'black',linewidth=2, label='Actual curve')

        for j in range(len(models)):
            if labels[j] in ['n = 2 adjusted quad. err','fourier 1', 'fourier 2']:
                xdata = x_axis[:61]
                ydata = models[j][i] - free_energies[:61]
            else:
                xdata = x_axis
                print(labels[j])
                ydata = models[j][i] - free_energies
            popt, pcov = curve_fit(lin_fit, xdata, ydata,absolute_sigma=True)

            fit = np.array(lin_fit(xdata, *popt))
            fit_corr_coeff = np.corrcoef(fit, ydata)[0,1]
            ac_corr_coeff = np.corrcoef(xdata, ydata)[0,1]
            print(fit_corr_coeff == ac_corr_coeff)
            plt.scatter(xdata, ydata, label=labels[j]+', corr. to. fit = '\
                +str(round(fit_corr_coeff,3))+'\n corr. to actual = '+str(round(ac_corr_coeff,3))+f'\n MAE = {mae(ydata,fit):.3f}')

        # plt.scatter(x_axis, predictions_n[15][i] - free_energies, label='n = 2.0 for E. free')
        plt.axis('square')
        plt.xlim(np.min(x_axis),np.max(x_axis))
        plt.ylim(np.min(x_axis),np.max(x_axis))
        # plt.ylim(-52,-38)
                # shift legend outside plots
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # set xscale equal to yscale

        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlabel(r'Actual $\Delta_1$ in Ha.')
        plt.ylabel(r'Predicted $\Delta_1$ in Ha.')
        plt.title(r'$\Delta_1 = E^{int}_{ele} = E^{Tot}_{ele} - \sum_{I} E^{free}_{I, ele}$ for '+f'{comps[i]}')
        plt.savefig(f'../images/delta1/1_{comps[i]}.png')

def delta_2():
    comps = ['NN', 'CO', 'BF']
    models = [quad_adjusted_prediction, quart_adjusted_prediction, \
        get_fourier_pred(1), get_fourier_pred(2), n_2_quad_adj,predictions_n[15]]
    labels = ['quad adj.', 'quart adj.', 'fourier 1', 'fourier 2',\
        'n = 2 adjusted quad. err','n = 2.0 for free E']
    for i in range(3):
        figure(figsize=(8, 8), dpi=180)
        x_axis = frac_energies / free_energies
        plt.plot(x_axis, frac_energies / free_energies,'black',linewidth=2, label='Actual curve')
        # plt.plot(x_axis, prediction_7_3[0] / free_energies, label='P. NN')

        for j in range(len(models)):
            if labels[j] in ['n = 2 adjusted quad. err','fourier 1', 'fourier 2']:
                xdata = x_axis[:61]
                ydata = models[j][i] / free_energies[:61]
            else:
                xdata = x_axis
                ydata = models[j][i] / free_energies
            popt, pcov = curve_fit(lin_fit, xdata, ydata,absolute_sigma=True)
            fit = np.array(lin_fit(xdata, *popt))
            fit_corr_coeff = np.corrcoef(fit, ydata)[0,1]
            ac_corr_coeff = np.corrcoef(xdata, ydata)[0,1]
            print(fit_corr_coeff == ac_corr_coeff)

            # plt.plot(xdata, fit)
            plt.scatter(xdata, ydata, label=labels[j]+', corr. to. fit = '\
                +str(round(fit_corr_coeff,3))+'\n corr. to actual = '+str(round(ac_corr_coeff,3))+f'\n MAE = {mae(ydata,fit):.3f}')
        # shift legend outside plots
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.axis('square')
        plt.xlim(np.min(x_axis),np.max(x_axis))
        plt.ylim(np.min(x_axis),np.max(x_axis))

        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlabel(r'Actual $\Delta_2$')
        plt.ylabel(r'Predicted $\Delta_2$')
        plt.title(r'$\Delta_2 = E^{Tot}_{ele} / \sum_{I} E^{free}_{I, ele}$ for '+f'{comps[i]}')
        plt.savefig(f'../images/delta2/2_{comps[i]}.png')

def delta_3_R():
    '''Right side'''
    comps = ['NN', 'CO', 'BF']
    models = [quad_adjusted_prediction, quart_adjusted_prediction, \
        get_fourier_pred(1), get_fourier_pred(2), n_2_quad_adj,predictions_n[15]]
    labels = ['quad adj.', 'quart adj.', 'fourier 1', 'fourier 2',\
        'n = 2 adjusted quad. err','n = 2.0 for free E']

    for i in range(3):
        figure(figsize=(8, 8), dpi=180)
        x_axis = R_P - frac_energies
        plt.plot(x_axis, R_P - frac_energies, 'black',linewidth=2,label='Actual curve')
        # plt.scatter(x_axis, prediction_7_3[0]  , label='P. NN')
        for j in range(len(models)):
            if labels[j] in ['n = 2 adjusted quad. err','fourier 1', 'fourier 2']:
                xdata = x_axis[:61]
                ydata = R_P[:61] - models[j][i]
            else:
                xdata = x_axis
                ydata = R_P - models[j][i]
            popt, pcov = curve_fit(lin_fit, xdata, ydata,absolute_sigma=True)
            fit = np.array(lin_fit(xdata, *popt))
            fit_corr_coeff = np.corrcoef(fit, ydata)[0,1]
            ac_corr_coeff = np.corrcoef(xdata, ydata)[0,1]

            # plt.plot(xdata, fit)
            plt.scatter(xdata, ydata, label=labels[j]+', corr. to. fit = '\
                +str(round(fit_corr_coeff,3))+'\n corr. to actual = '+str(round(ac_corr_coeff,3))+f'\n MAE = {mae(ydata,fit):.3f}')
        # shift legend outside plots
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


        plt.axis('square')
        plt.xlim(np.min(x_axis),np.max(x_axis))
        plt.ylim(np.min(x_axis),np.max(x_axis))
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlabel(r'Actual $\Delta_3$ in Ha.')
        plt.ylabel(r'Predicted $\Delta_3$ in Ha.')
        plt.title(r'$\Delta_3 = E^{Prot.}_{ele} - E^{Tot.}_{ele}$ for right side protonation for '+f'{comps[i]}')
        plt.savefig(f'../images/delta3/3_R_{comps[i]}.png')

def delta_3_L():
    '''left side'''
    for i in range(3):
        figure(figsize=(8, 8), dpi=180)
        x_axis = L_P - frac_energies
        plt.plot(x_axis, L_P - frac_energies,'black',linewidth=2, label='Actual curve')
        # plt.scatter(x_axis, prediction_7_3[0]  , label='P. NN')
        for j in range(len(models)):
            if labels[j] in ['n = 2 adjusted quad. err','fourier 1', 'fourier 2']:
                xdata = x_axis[:61]
                ydata = L_P[:61] - models[j][i]
            else:
                xdata = x_axis
                ydata = L_P - models[j][i]
            popt, pcov = curve_fit(lin_fit, xdata, ydata,absolute_sigma=True)
            fit = np.array(lin_fit(xdata, *popt))
            fit_corr_coeff = np.corrcoef(fit, ydata)[0,1]
            ac_corr_coeff = np.corrcoef(xdata, ydata)[0,1]

            # plt.plot(xdata, fit)
            plt.scatter(xdata, ydata, label=labels[j]+', corr. to. fit = '\
                +str(round(fit_corr_coeff,3))+'\n corr. to actual = '+str(round(ac_corr_coeff,3))+f'\n MAE = {mae(ydata,fit):.3f}')
        # shift legend outside plots
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # plt.legend()
        plt.axis('square')
        plt.xlim(np.min(x_axis),np.max(x_axis))
        plt.ylim(np.min(x_axis),np.max(x_axis))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel(r'Actual $\Delta_3$ in Ha.')
        plt.ylabel(r'Predicted $\Delta_3$ in Ha.')
        plt.title(r'$\Delta_3 = E^{Prot.}_{ele} - E^{Tot.}_{ele}$ for left side protonation for '+f'{comps[i]}')
        plt.savefig(f'../images/delta3/3_L_{comps[i]}.png')