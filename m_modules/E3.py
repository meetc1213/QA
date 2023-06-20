'''Extra code: some code for generating data for anatole'''

# via plotting different errors w.r.t n, found that n = 2 works best. Plot E2.diff_err_n()
prediction_2 = predictions_n[15]
n_2_fit_err = []
errs = frac_energies - prediction_2
new_x_axis = x_axis[:61]
for i in range(3):
    popt_, pcov_ = curve_fit(quad_fit, new_x_axis, errs[i][:61],absolute_sigma=True)
    fitted_err_ = np.array(quad_fit(new_x_axis, *popt_))
    n_2_fit_err.append(fitted_err_)
    plt.plot()
n_2_quad_adj = n_2_fit_err + predictions_n[15][:,:61]

# geneating data for anatole
nl_hf = []
for i in np.linspace(0,7,71):
    mol = new_mol(NN, 2, -i,i)[0]
    nl_hf.append(np.dot(d_Z_lambda(NN,mol,2,0),AG_NN ))

e,l = np.loadtxt('data/dft_hf_dvs.csv')[0], np.loadtxt('data/dft_hf_dvs.csv')[2]
np.savetxt('data/dft_hf_dvs.csv',[e, np.array(nl_hf),l])

'''For fitting further polynomial error '''
further_err = frac_energies - quart_adjusted_prediction
poo = [ 0.01, .03054518,  1.4933,  1.14250962]
popt_NN, pcov_NN = curve_fit(power_sine, x_axis, further_err[0],p0=poo,absolute_sigma=True)
fitted_err_NN = np.array(power_sine(x_axis, *popt_NN))

# popt_CO, pcov_CO = curve_fit(power_sine, x_axis,  further_err[1],absolute_sigma=True)
# q_fitted_err_CO = np.array(power_sine(x_axis, *popt_CO))

# popt_BF, pcov_BF = curve_fit(power_sine, x_axis,  further_err[2],absolute_sigma=True)
# q_fitted_err_BF = np.array(power_sine(x_axis, *popt_BF))
app = np.array(power_sine(x_axis, *popt_NN))
plt.plot(x_axis, further_err[0],label='NN further err')
plt.plot(x_axis, app,label='app')

plt.legend()