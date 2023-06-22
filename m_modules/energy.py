# contains functions for predicting energy
# import sys, os
# sys.path.append('/Users/meet/Desktop/Courses/Research/Chem/Code/QA/g_modules')
# sys.path.append('/Users/meet/Desktop/Courses/Research/Chem/Code/QA/m_modules')
# # Add the parent directory to the system path
# sys.path.append('/Users/meet/Desktop/Courses/Research/Chem/Code/QA')
from pyscf import gto,scf, cc
import numpy as np
import pyscf
import basis_set_exchange as bse
from g_modules.FcMole import FcM, FcM_like
from g_modules.AP_class import APDFT_perturbator as AP
from g_modules.alch_deriv import first_deriv_elec,DeltaV
from m_modules.config import *

max_lam, steps, step, exponent = return_max_lam_steps()

def get_free_energy(mol_i, n):
    '''takes a molecule and the power of the free energy term and returns the free energy'''
    if n == 2:
        return -0.5*np.sum(mol_i.atom_charges()**n)
    return np.sum(mol_i.atom_charges()**n)

def new_mol(mol_i,n,l_i, l_f, left_right = None):
    '''Returns a new molecule, the converged object,
    total electronic energy and free energy of the new molecule
    at the l_i, l_f perturbation with n as the non linearity parameter.'''

    if left_right == 'L':
        '''if the H atom is on the left side of the molecule, only perturb the ones on the right'''
        mol = FcM_like(mol_i,fcs=[[1,2],[l_i,l_f]])

    else:
        mol = FcM_like(mol_i,fcs=[[0,1],[l_i,l_f]])

    mf_mol=scf.RKS(mol)
    mf_mol.xc="PBE0"
    mf_mol.verbose = 0
    mf_mol.scf(dm0=mf_mol.init_guess_by_1e())
    elec_energy = round(mf_mol.energy_elec()[0],3)

    return [mol, mf_mol, elec_energy, get_free_energy(mol,n)]

def d_Z_lambda(mol_i, mol_f,n,lam):
    # non linear Z derviative
    Z_i = np.array(mol_i.atom_charges())
    Z_f = np.array(mol_f.atom_charges())
    num = (1/n)* (Z_f**n - Z_i**n)
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

def gen_data(mol,AG,n,e_mol,l_i, l_f):
    # generates linear and non linear prediction data
    l_pre = []
    nl_pre = []
    for i in np.linspace(l_i,l_f, steps + 1):
        pre = get_pred(mol, AG,n, e_mol, i,i)
        l_pre.append(pre[0])
        nl_pre.append(pre[1])
    return np.array(l_pre), np.array(nl_pre)

'''wrapper function for generating DFT data'''
def get_symmetric_change_data(min_lam,max_lam, gen=False):
    # gets ANM1 data
    if gen:
        frac_energies = []
        free_energies = []
        i = min_lam
        while round(i,3) <= max_lam:
            mol_props = new_mol(NN,exponent, -i,i)
            e_mol = mol_props[2]
            free_e = mol_props[3]
            frac_energies.append(e_mol)
            free_energies.append(free_e)
            i  += step
        frac_energies = np.array(frac_energies)
        free_energies = np.array(free_energies)
        np.savetxt(f"data/dft_0_to_{max_lam}.csv",[frac_energies,free_energies])

    return np.loadtxt(f'data/dft_0_to_{max_lam}.csv')

def get_asymmetric_change_data(min_lam,max_lam, gen=False):
    # gets the protnation data
    if gen:
        R_NN = gto.M(atom= f"N 0 0 0; N 0 0 {d}; H 0 0 {d+s}",unit="Bohr",charge=1,basis='unc-ccpvdz')
        L_NN = gto.M(atom= f"H 0 0 {-s}; N 0 0 0; N 0 0 {d}",unit="Bohr",charge=1,basis='unc-ccpvdz')
        # can protonate in the nearby atom and the far atom
        R_atom_p = []
        L_atom_p = []
        i = min_lam

        while round(i,3) <= max_lam:
            mol_props_R = new_mol(R_NN,exponent, -i,i,left_right='R')
            e_mol_R = mol_props_R[2]
            R_atom_p.append(e_mol_R)

            mol_props_L = new_mol(L_NN,exponent, -i,i,left_right='L')
            e_mol_L = mol_props_L[2]
            L_atom_p.append(e_mol_L)
            i  += step

        np.savetxt(f"data/prot_0_to_{max_lam}.csv",[R_atom_p,L_atom_p])

    return np.loadtxt(f'data/prot_0_to_{max_lam}.csv')