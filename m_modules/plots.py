
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from m_modules import energy
from m_modules.energy import *

def l_nl_plot():
    l_vals = np.array([np.array(x) for x in [pre_NN_l,pre_CO_l,pre_BF_l]])
    l_keys = ['L. Z @ NN', 'L. Z @ CO','L. Z @ BF']
    nl_keys = ['NL. Z @ NN','NL. Z @ CO','NL. Z @ BF']

    l_lines = dict(zip(l_keys, l_vals))
    nl_lines = dict(zip(nl_keys,prediction_7_3))
    all_lines = {**l_lines, **nl_lines}

    format = ['r:','g:', 'b:','r--','g--', 'b--']

    # figure(figsize=(8, 6), dpi=80)
    fig, ax = plt.subplots()
    ax.scatter([0,1,2,3,4,5,6,7],[e_NN, e_CO, e_BF, e_BeNe, e_LiNA, e_HeMg, e_HAl, e_Si],label='Comps.')
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
    ax.set_xlabel(r'Lambda ($\lambda$)')
    ax.set_xlim(0,7)
    ax.set_title('Predictions with a quadratic error adjusted')

    # add another x axis in plot
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels(['NN','CO','BF','BeNe','LiNa','HeMg','HAl','Si'])

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
            ax[0].plot(x_axis, arg[0], label = f'manual fit')
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
