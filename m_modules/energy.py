# contains functions for predicting energy
import sys, os
sys.path.append('/Users/meet/Desktop/Courses/Research/Chem/Code/QA/g_modules')
# sys.path.append('/Users/meet/Desktop/Courses/Research/Chem/Code/QA/m_modules')
# # Add the parent directory to the system path
sys.path.append('/Users/meet/Desktop/Courses/Research/Chem/Code/QA')
from pyscf import gto,scf, cc, grad
import numpy as np
import basis_set_exchange as bse
from g_modules.FcMole import FcM, FcM_like
from g_modules.AP_class import APDFT_perturbator as AP
from g_modules.alch_deriv import first_deriv_elec,DeltaV
from m_modules.config import *
import os
import periodictable
from reloading import reloading
import sys

def is_file_in_folder(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    return os.path.isfile(file_path)

def get_element_symbol(atomic_number):
    element = periodictable.elements[atomic_number]
    return element.symbol

def get_mol_symbol(equi_Z, l_1, l_2):
    element1 = get_element_symbol(equi_Z+l_1)
    element2 = get_element_symbol(equi_Z+l_2)
    return element1 + element2

def tri_party_mol_props(atomic_number, d):

    '''Returns the molecular energies and the converged objects
    for the first three members of the series'''

    return_list = np.array([])
    for i in range(3):
        # print(f'trying to build ({atomic_number - i} ; {atomic_number + i})')
        NN = gto.M(atom= f"{atomic_number - i} 0 0 0; {atomic_number + i} 0 0 {d}",unit="Bohr",basis='unc-ccpvdz') # uncontracted cc-pvdz
        return_list = np.append(return_list, new_mol(NN, 0,0))

    return return_list

def get_free_energy(mol_i, n):
    '''takes a molecule and the power of the free energy term and returns the free energy'''
    if n == 2:
        return 0.5*np.sum(mol_i.atom_charges()**n)
    return np.sum(mol_i.atom_charges()**n)

def new_mol(mol_i,l_i, l_f):
    '''Returns a new molecule, the converged object,
    total electronic energy and free energy of the new molecule
    at the l_i, l_f perturbation with n as the non linearity parameter.'''

    mol = FcM_like(mol_i,fcs=[[0,1],[l_i,l_f]])

    mf_mol=scf.RKS(mol)
    mf_mol.xc="PBE0"
    mf_mol.verbose = 0
    mf_mol.scf(dm0=mf_mol.init_guess_by_1e())
    elec_energy = round(mf_mol.energy_elec()[0],3)

    return [mol, mf_mol, elec_energy]

def get_mol_energy_grad(Z1,Z2,sep):
    NN = gto.M(atom= f"{Z1} 0 0 0; {Z2} 0 0 {sep}",unit="Bohr",basis='unc-ccpvdz')
    mf_mol=scf.RKS(NN)
    mf_mol.xc="PBE0"
    mf_mol.verbose = 0
    Te = mf_mol.scf(dm0=mf_mol.init_guess_by_1e())
    elec_energy = round(mf_mol.energy_elec()[0],3)

    elec_grad_values = mf_mol.Gradients().grad_elec()
    total_grad_values = mf_mol.Gradients().grad()
    # elec_grad_values = grad_obj.kernel()
    return elec_energy, np.array([elec_grad_values[0][-1], elec_grad_values[1][-1]]), \
            np.array([total_grad_values[0][-1], total_grad_values[1][-1]])

def d_Z_lambda(mol_i, mol_f,n,lam):
    # non linear Z derviative
    Z_i = np.array(mol_i.atom_charges())
    Z_f = np.array(mol_f.atom_charges())
    A = 1

    num = A*(1/n)* (Z_f**n - Z_i**n)
    den = ((Z_i**n) + lam*(Z_f**n - Z_i**n))**(1 - 1/n)
    return num / den

def Z_diff(mol_i, mol_f):
    # linear Z derivative
    Z_i = np.array(mol_i.atom_charges())
    Z_f = np.array(mol_f.atom_charges())
    return Z_f - Z_i

def AG(mf,sites=[0,1]):
    # returns alchemical gradient vector of a molecule
    grads=[]
    for site in sites:
        grads.append(first_deriv_elec(mf,DeltaV(mf.mol,[[site],[1]])))
    return np.array(grads)

def get_pred(mol_i,AG_i, n,e_i,l_i,l_f):
    '''Returns the linear Z and non linear Z prediction
    from mol_i using its alchemical grad and energy
    at specific perturbation l_i, l_f at the individual atoms

    Caution: l_i = l_f to increase and decrease nuclear charge by same amount in 2 atoms.
    Observe the negative sign.
    '''

    mol = FcM_like(mol_i,fcs=[-l_i,l_f])
    return np.round([e_i + np.dot(Z_diff(mol_i,mol), AG_i),e_i + np.dot(d_Z_lambda(mol_i,mol,n,0), AG_i)],3)

def gen_data(mol,AG,e_mol,l_i, l_f, args):
    # generates linear and non linear prediction data
    max_d_lam, steps, step, exponent, d = args
    l_pre = []
    nl_pre = []
    for i in np.linspace(l_i,l_f, steps):
        pre = get_pred(mol, AG,exponent, e_mol, i,i)
        l_pre.append(pre[0])
        nl_pre.append(pre[1])
    return np.array(l_pre), np.array(nl_pre)

'''wrapper function for generating DFT data'''
def get_symmetric_change_data(min_lam,max_d_lam,atomic_number, args):
    max_d_lam, steps, step, d = args
    folder_path = f'data/alc/step={round(step,2)}/'
    file_name = f'{atomic_number}_{step}_dft_0_to_{max_d_lam}.csv'
    # print(f"{steps, step,d}",flush=True)
    sys.stdout.flush()
    if not is_file_in_folder(folder_path, file_name):

        frac_energies = []
        free_energies = []
        i = round(min_lam,3)
        for i in np.arange(min_lam, max_d_lam  + step, step):
            i = round(i,3)
            NN = gto.M(atom= f"{atomic_number} 0 0 0; {atomic_number} 0 0 {d}",unit="Bohr",basis='unc-ccpvdz')
            mol_props = new_mol(NN, -i,i)
            e_mol = mol_props[2]
            free_e = mol_props[3]
            frac_energies.append(e_mol)
            free_energies.append(free_e)

            if i%0.1 == 0:
                print(f"YOOOO lambda = {i} done for {atomic_number}",flush=True)
                sys.stdout.flush()
            i  += step
        frac_energies = np.array(frac_energies)
        free_energies = np.array(free_energies)
        np.savetxt(f"{folder_path}{file_name}",[frac_energies,free_energies])

    return np.loadtxt(f'{folder_path}{file_name}')

def get_asymmetric_change_data(min_lam,max_d_lam, args,gen=False):

    # gets the protnation data
    max_d_lam, steps, step, exponent, d = args
    s = 0.94 # 0.5 angstroms or 0.94 Bohr.
    if gen:
        R_NN = gto.M(atom= f"N 0 0 0; N 0 0 {d}; H 0 0 {d+s}",unit="Bohr",charge=1,basis='unc-ccpvdz')
        L_NN = gto.M(atom= f"H 0 0 {-s}; N 0 0 0; N 0 0 {d}",unit="Bohr",charge=1,basis='unc-ccpvdz')
        # can protonate in the nearby atom and the far atom
        R_atom_p = []
        L_atom_p = []
        i = min_lam

        while round(i,3) <= max_d_lam:
            mol_props_R = new_mol(R_NN, -i,i,left_right='R')
            e_mol_R = mol_props_R[2]
            R_atom_p.append(e_mol_R)

            mol_props_L = new_mol(L_NN, -i,i,left_right='L')
            e_mol_L = mol_props_L[2]
            L_atom_p.append(e_mol_L)
            i  += step

        np.savetxt(f"data/prot_0_to_{max_d_lam}.csv",[R_atom_p,L_atom_p])

    return np.loadtxt(f'data/prot_0_to_{max_d_lam}.csv')

# def get_morse_potential(): # need atomic charges, and separation distance
def gen_sep_data(Z1, Z2, min_sep,max_sep, sep_step):# need atomic charges, and separation range
    method = 'RKS'
    folder_path = f'data/sep/step={round(sep_step,2)}/'
    file_name = f'{Z1,Z2}_dft_{method}_{min_sep}_to_{max_sep}.csv'
    frac_energies = []
    i = round(min_sep,3)

    if not is_file_in_folder(folder_path, file_name):
        for i in np.arange(min_sep, max_sep  + sep_step, sep_step):
            i = round(i,3)
            NN = gto.M(atom= f"{Z1} 0 0 0; {Z2} 0 0 {i}",unit="Bohr",basis='unc-ccpvdz')
            mf_mol=scf.RKS(NN)
            mf_mol.xc="PBE0"
            mf_mol.verbose = 0
            TE = mf_mol.scf(dm0=mf_mol.init_guess_by_1e())
            elec_energy = round(mf_mol.energy_elec()[0],3)
            frac_energies.append(TE)

            if i%0.1 == 0:
                print(f"YOOOO lambda = {i} done for {Z1}",flush=True)
                sys.stdout.flush()

            i  += sep_step

        frac_energies = np.array(frac_energies)

        np.savetxt(f"{folder_path}{file_name}",[frac_energies])

    return np.loadtxt(f'{folder_path}{file_name}')

def d_S_gamma(d_i, d_f, n):

    return (-1/n) * (d_i**(1+n)) * ( 1 / (d_f**n) - 1 / (d_i**n ))

def d_S_gamma_morse(d_i, d_f, a):
    # evaluating the d_gamma Sep at delta gamma = 1.
    # Would return the negative of this for the the prediction

    return (1-np.exp(-a*(d_f-d_i))) / (2*a*np.exp(-a*(d_f-d_i)))


# (1/n) * d_i **(n(1-n)) * (d_i**n + U)**2 \
#             * (1 / (d_f**n + U) - 1 / (d_i**n + U)) # previous expression