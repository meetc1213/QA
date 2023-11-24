
import matplotlib.pyplot as plt
from energy import *
from stat_funcs import *
# from main import *
import pandas as pd
import numpy as np
import pyperclip


'''Multi-processing'''
    # inputs = np.arange(5,17)
    # num_cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=num_cores)
    # results = pool.map(atomic_bomb, inputs)
    # pool.close()
    # pool.join()
    # exp_dic = {key: value for key, value in zip(inputs, results)}
    # csv_file = f'data/alc/step={step}/all_alc_errors_{d}_2.csv'
    # df = pd.DataFrame(exp_dic)
    # df.to_csv(csv_file, index=False)
def l_nl_plot(atomic_number, exponent, d, frac_energies, pre_NN_l, pre_CO_l, pre_BF_l, prediction, x_axis):
    '''Plotting'''
    mol_dic = {}
    NN = gto.M(atom= f"{atomic_number} 0 0 0; {atomic_number} 0 0 {d}",unit="Bohr",basis='unc-ccpvdz') # uncontracted cc-pvdz
    for i in range(0,atomic_number + 1):
        if i != atomic_number:
            mol_dic['e_'+get_mol_symbol(atomic_number, -i , +i)] = new_mol(NN,-i,i)[2]
        else:
            mol = gto.M(atom= f"{2*atomic_number} 0 0 0",unit="Bohr",basis='unc-ccpvdz') # uncontracted cc-pvdz
            mf_mol=scf.RKS(mol)
            mf_mol.xc="PBE0"
            mf_mol.verbose = 0
            mf_mol.scf(dm0=mf_mol.init_guess_by_1e())
            elec_energy = round(mf_mol.energy_elec()[0],3)
            mol_dic['e_'+get_element_symbol(2*atomic_number)] = elec_energy

    l_vals = np.array([pre_NN_l,pre_CO_l,pre_BF_l])
    l_keys = []
    nl_keys = []

    for i in range(3):
        l_keys.append(f'L. Z @ {get_mol_symbol(atomic_number, -i , +i)}')
        nl_keys.append(f'NL. Z @ {get_mol_symbol(atomic_number, -i , +i)}')

    l_lines = dict(zip(l_keys, l_vals))
    nl_lines = dict(zip(nl_keys,prediction))
    all_lines = {**l_lines, **nl_lines}

    format = ['r:','g:', 'b:','r--','g--', 'b--']
    fig, ax = plt.subplots()
    ax.scatter(np.arange(0,atomic_number + 1),mol_dic.values(),label='Comps.')
    ax.plot(x_axis,np.array(frac_energies),'black',label='Actual curve')


    '''extra lines'''
    # ax.plot(x_axis,np.array(free_energies),'orange',label='free energy curve')
    # ax.plot(x_axis,np.array(R_P),'brown',label='R. protonated curve')
    # ax.plot(x_axis,np.array(L_P),'grey',label='L. protonated curve')

    i = 0
    for line in all_lines:
        # plt.scatter(np.linspace(0,2,len(lines[line])),lines[line],label=line)
        plt.plot(x_axis,all_lines[line],format[i],label=line)
        i +=1

    ax.legend()
    ax.set_ylabel('Total Elec. enegy in Ha.')
    ax.set_xlabel(r'Lambda ($\lambda$)')
    ax.set_xlim(0,7)
    ax.set_title('Predictions with a quadratic error adjusted')

    # add another x axis in plot
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels(mol_dic.keys())
    plt.show()

def plot_errors(name, x_axis,actual, prev_pred, new_pred,\
                    comps, atomic_number,*arg):
    '''
    function to plot error before and after a correction
    '''
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    # put 1 to only plot error from equilibrium diatomic.
    for i in range(1):
        ax[0].scatter(x_axis, actual - prev_pred[i], label = f'err for {comps[i]} pre')
        ax[0].plot(x_axis, new_pred[i] - prev_pred[i], label = f'fits for {comps[i]} err')
        if arg:
            ax[1].plot(x_axis, arg[0][0], label = f'curve fit')
            ax[1].plot(x_axis, arg[1], label = f'manual fit')
        ax[1].scatter(x_axis, actual - new_pred[i], label = f'for {comps[i]}')

    ax[0].set_title(f'Error before adjusting for {name} error')
    ax[1].set_title(f'Error after adjusting for {name} error')
    ax[0].legend()
    ax[1].legend()
    ax[1].set_xlabel(r'$\lambda$')
    fig.text(0.04, 0.5, 'Error in Ha.', va='center', rotation='vertical',size=20)
    fig.suptitle(f'Error for total electrons from {get_element_symbol(atomic_number)}-{get_element_symbol(atomic_number)}',size=20)
    plt.show()
    return fig

def plot_n_2_errors():
    atomic_number = 7
    step = 0.1
    max_d_lam = atomic_number
    steps =  int(atomic_number / step  + 1)
    d = 2.1
    mol_objs = tri_party_mol_props(atomic_number, d)
    args = [max_d_lam, steps, step, d ]

    dft =  get_symmetric_change_data(0,atomic_number,atomic_number, args)[0]
    exponent = 2.000

    args = [max_d_lam, steps, step,exponent, d ]

    pre, x_axis = init_alc(atomic_number, exponent,dft, mol_objs, args)

    for i in range(len(pre)):
        plt.plot(x_axis, dft-pre[i], label = f'error from {i}, MAE = {round(mae(dft, pre[i]),3)}')

    plt.legend()

def table(temp):
    '''
    temp: the dictionary with columns as keys and the rows as values
    '''
    # temp={keys[i] : vals[i] for i in range(len(keys))}

    # Create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(temp)

    selected_columns = [key if isinstance(key, float) else str(key) for key in temp.keys()] #np.array(list(temp.keys()))

    column_spec = {('|c'* (len(selected_columns)+1) + '|')}
    latex_code = f"\\begin{{tabular}}{column_spec}\n"
    latex_code += "\\hline\n"
    latex_code += " & ".join(map(str, selected_columns)) + "\\\\\n"
    latex_code += "\\hline\n"

    for index, row in df.iterrows():
        row_values = [row[col] for col in selected_columns]
        latex_code += " & ".join(map(str, row_values)) + " \\\\\n"

    latex_code += "\\hline\n"
    latex_code += "\\end{tabular}"

    pyperclip.copy(latex_code)
    print(latex_code)

def DFT_PE_surface_plot():
    '''For plotting the DFT PE surface and related table'''
    Z = 7
    min_g = 0.5
    max_g = 6

    min_l = 0
    max_l = 7
    stepsize = 0.1

    lam = np.round(np.arange(min_l, max_l + stepsize, stepsize),3)
    gam = np.round(np.arange(min_g, max_g + stepsize, stepsize),3)
    gam = gam[gam <= 4.9]
    L, G = np.meshgrid(lam, gam)
    combinations = np.array(list(zip(L.ravel(), G.ravel())))

    dft_data = pd.read_csv("data/data.csv")

    k =  [x for x in combinations if x[1] <= 4.9]

    z = [dft_data.iloc[0, x] for x in range(1,len(k)+1)]
    x = L
    y = G

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('3D Scatter Plots')

    # Create a 3D subplot
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')

    # highlighting a path through the data


    # Plot the scatter plot
    ax1.scatter(x, y, z, s=5) # fragments[i].z

    # Set subplot title and labels
    ax1.set_title('DFT elec energy surface of N2 series')
    ax1.set_xlabel(r'$\Delta \lambda$ (Alchemical change)')
    ax1.set_ylabel(r'$\Delta \gamma$ (Separation change)')
    ax1.set_zlabel('Electronic Energy (in Ha.)')

    plt.tight_layout()

    # Show the figure
    plt.show()



    '''Table genratation data for the paper'''

    all_keys = np.array(dft_data.columns[1:])
    res_keys = []
    dic = {}
    for key in gam:
        dic[key] = []
        for v in range(len(all_keys)):
            if str(key)+')' in all_keys[v]:
                res_keys.append(all_keys[v])
                e = dft_data[all_keys[v]][0]
                # print(key, all_keys[v],e )
                dic[key].append(e)

    vals = list(dic.values())

    # the huge matrix table
    keys = ['lambda', *gam]
    vals = [lam, *vals]
    temp = {key: value for key,value in zip(keys, vals)}
    table(temp)

    '''Works'''
    # the perturbation tuple table
    # keys = ['(L, G)', 'DFT']
    # vals = [res_keys, np.array(dft_data[res_keys].values)[0]]
    # temp = {key: value for key,value in zip(keys, vals)}
    # table(temp)

def errors_atomic(atomic_number,df):

    '''Structure of delta_lam:
    delta_lam = [

    # delta_lam_1 (NN pred) : {chi_sq : [...values for all n...], ...}
    # delta_lam_2 ...
    ]
    '''
    err_funcs = [chi_sq, rmse, max_error, mae, std,\
            get_integral_error]
    delta_lam = [{f.__name__ : [] for f in err_funcs},
                 {f.__name__ : [] for f in err_funcs},
                 {f.__name__ : [] for f in err_funcs}]

    for errs in list(df[f'{atomic_number}']): # scavenging for particular n from here
        errs = np.array(eval(errs))
        errs = np.round(errs, 3)

        for j in range(len(errs)):
        # errs[j] is all errors for a given delta_lambda
        # for all n
            delta_lam_err = errs[j]

            for i in range(len(delta_lam_err)):
            # i is index of all errors for a given delta_lambda
            # at a given
                (delta_lam[j][err_funcs[i].__name__]).append(delta_lam_err[i])

    return delta_lam

def output_optimized_exps(atomic_number, df, n):
    # delta_lam should contain the optimized exponent and
    # the corresponding error for each type of error for all three predictions
    err_funcs = [chi_sq, rmse, max_error, mae, std,\
            get_integral_error]
    delta_lam = [{f.__name__ : [] for f in err_funcs}, \
                {f.__name__ : [] for f in err_funcs}, \
                {f.__name__ : [] for f in err_funcs}]

    all_err_data = errors_atomic(atomic_number, df)
    for delta_lambda in range(3):
        delta_lambda_data = all_err_data[delta_lambda]
        x = np.array(list(delta_lambda_data.values())) # contains all errors for a given delta_lambda  for all n for all types of errors
        keys = list(delta_lambda_data.keys()) # contains the name of errors ?

        delta_lambda_mol = get_mol_symbol(atomic_number, delta_lambda, - delta_lambda)
        # print(f'Prediction optimized exponents from {delta_lambda_mol} \n\n')

        for i in range(len(x)):
            current_error = np.array(x[i])
            min_err_index = np.argmin(current_error)# index of the n corresponding to the lowest error
            min_err_value = round(current_error[min_err_index],3 )
            min_exp = round(n[min_err_index],3)

            delta_lam[delta_lambda][keys[i]].append((min_exp, min_err_value))

            # print(f'Optimized exponent for {keys[i]} is {min_exp} with error {min_err_value}')
        # print('\n\n')

    return delta_lam

def get_plotting_data():
    '''
    fmt of plotting_data:
    plotting_data = [

        NN_pred_index : {chi_sq : [...optimized n for all atomic_numbers...], ...}
    ]
    '''
    inputs = np.arange(5,17)
    n = np.round(np.arange(1.5, 2.5, 0.001),3)
    step = 0.1
    d = 2.1
    csv_file = f'data/alc/step={step}/all_alc_errors_{d}_2.csv'
    df = pd.read_csv(csv_file)

    plotting_data = output_optimized_exps(5, df, n)
    for atomic_number in inputs[1:]:
        dic_new = output_optimized_exps(atomic_number, df, n)

        for i in range(3):
            plotting_data[i] = {key: np.concatenate((plotting_data[i][key],
                                                    dic_new[i][key])) for key in dic_new[i]}

    plotting_data = np.array(plotting_data)


    '''
    Generating data table:
    # want to make the table of the optimized exponents for a particular type of error
    cols = ['Z','delta_lam_1','delta_lam_2','delta_lam_3']
    values = [list(np.arange(5,17))]
    for i in range(3):
        values.insert(i+1, tuple_coverter(get_plotting_data()[0][i]['mae']))

    temp = {key: value for key,value in zip(cols, values)}
    table(temp)
    '''

    return plotting_data, inputs

def optimized_exps_3D_plot():
    plotting_data, inputs = get_plotting_data()
    for error_to_plot in plotting_data[0].keys():
        atomic_numbers = inputs
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('3D Scatter Plots')

        # Create a 3D subplot
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        f = 15
        for delta_lambda in range(3):
            exponents_errors = plotting_data[delta_lambda][error_to_plot]
            exponents_, errors_ = zip(*exponents_errors)

            # Plot the scatter plot
            ax1.scatter(atomic_numbers, exponents_, errors_, s=15, label=r'Prediction from $\Delta \lambda = $'+f'{delta_lambda}') # fragments[i].z

        # Set subplot title and labels
        ax1.set_title(f'{error_to_plot} error of optimized exponents against atomic numbers')
        ax1.set_xlabel(r'$Z$ series', fontsize=f)
        ax1.set_ylabel(r'n$_{opt}$', fontsize=f)
        ax1.set_zlabel(f'{error_to_plot.upper()} Error (Ha.)', fontsize=f)
        ax1.legend(loc='center right', fontsize=f)
        plt.tight_layout()
        plt.savefig(f'data/figs/alc/optimized/{error_to_plot}.png', dpi=300)
        plt.show()
