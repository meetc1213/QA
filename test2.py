from main import *
from m_modules.energy import *
import pandas as pd

def dft_Surface():
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

    print(L)
    # Plot the scatter plot
    ax1.scatter(x, y, z, s=5) # fragments[i].z

    # Set subplot title and labels
    ax1.set_title('DFT elec energy surface of N2 series')
    ax1.set_xlabel(r'$\Delta \lambda$ (Alchemical change)')
    ax1.set_ylabel(r'$\Delta \gamma$ (Separation change)')
    ax1.set_zlabel('Electronic Energy (in Ha.)')

    plt.tight_layout()
    plt.savefig('3d_scatter.png')
    # Show the figure
    plt.show()

def optimized_exps_3D_plot():
    plotting_data, inputs = get_plotting_data()
    for error_to_plot in plotting_data[0].keys():
        atomic_numbers = inputs
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('3D Scatter Plots')

        # Create a 3D subplot
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')

        for delta_lambda in range(3):
            exponents_errors = plotting_data[delta_lambda][error_to_plot]
            exponents_, errors_ = zip(*exponents_errors)

            # Plot the scatter plot
            ax1.scatter(atomic_numbers, exponents_, errors_, s=15, label=r'Prediction from $\Delta \lambda = $'+f'{delta_lambda}') # fragments[i].z

        # Set subplot title and labels
        ax1.set_title(f'{error_to_plot} error of optimized exponents against atomic numbers')
        ax1.set_xlabel(r'$Z$ series')
        ax1.set_ylabel(r'Exponent $(n)$')
        ax1.set_zlabel('Error (Ha.)')
        ax1.legend()
        plt.tight_layout()
        plt.show()
