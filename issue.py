from pyscf import gto,scf
import numpy as np
import pyscf
import matplotlib.pyplot as plt
import basis_set_exchange as bse
from FcMole import FcM, FcM_like
from AP_class import APDFT_perturbator as AP

# # doesn't work:
NN = gto.M(atom= f"N 0 0 0; N 0 0 2.1",unit="Bohr",basis='unc-ccpvdz')

# works
# mol = gto.M(
#     atom = '''N 0 0 0; N 0 0 2''',
#     basis =
#     {'N': 'unc-ccpvdz', # prefix "unc-" will uncontract the ccpvdz basis.
#                                 # It is equivalent to assigning
#                                 #   'O': gto.uncontract(gto.load('ccpvdz', 'O')),
#              'H': 'unc-ccpvdz'  # H1 H2 will use the same basis ccpvdz
#             }
# )


