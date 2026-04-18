"""Physical constants used across the EBIC simulator (SI units)."""
import numpy as np

q      = 1.602176634e-19   # elementary charge (C)
kB     = 1.380649e-23      # Boltzmann (J/K)
h      = 6.62607015e-34    # Planck (J s)
hbar   = h / (2 * np.pi)
m0     = 9.1093837015e-31  # electron rest mass (kg)
eps0   = 8.8541878128e-12  # vacuum permittivity (F/m)

# Silicon defaults (used when a semiconductor field is missing)
NI_SI_300K = 1.0e10        # intrinsic carrier density (cm^-3) at 300 K
EG_SI_300K = 1.12          # bandgap (eV)
CHI_SI     = 4.05          # electron affinity (eV)
EPS_R_SI   = 11.7          # relative permittivity
ME_SI      = 1.08          # m*/m0 (DOS effective mass electron)
MH_SI      = 0.81          # m*/m0 (DOS effective mass hole)

T_DEFAULT  = 300.0         # K

# e-h pair generation energy in Si (empirical, ~3.6 eV)
EHP_ENERGY_SI_eV = 3.6
