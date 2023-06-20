from pyscf import gto,scf
import numpy as np
import pyscf
import matplotlib.pyplot as plt
import basis_set_exchange as bse
from g_modules.FcMole import FcM, FcM_like
from g_modules.AP_class import APDFT_perturbator as AP


# creating a NNH molecule
ex = gto.M(atom= f"N 0 0 0; N 0 0 2.1; H 0 0 {2.1+0.94}",unit="Bohr",charge=1,basis='unc-ccpvdz')
# the below should ideally create SiH molecule
new_ex = FcM_like(ex,fcs=[[0,1],[-7,7]])
mf_mol=scf.RKS(new_ex)
mf_mol.xc="PBE0"
Te_mol=mf_mol.scf(dm0=mf_mol.init_guess_by_1e())

# this should ideally be electron energy of SiH molecule
elec_energy = round(mf_mol.energy_elec()[0],3)

# calculating explicitly energy of SiH molecule
mol_ac = gto.M(atom= f"Si 0 0 2.1; H 0 0 {2.1+0.94}",unit="Bohr",charge=1,basis='unc-ccpvdz')
mf_mol_ac=scf.RKS(mol_ac)
mf_mol_ac.xc="PBE0"
Te_mol_ac=mf_mol_ac.scf(dm0=mf_mol_ac.init_guess_by_1e())

# this should ideally be electron energy of SiH molecule
actual_elec_energy = round(mf_mol_ac.energy_elec()[0],3)

print(elec_energy == actual_elec_energy)
