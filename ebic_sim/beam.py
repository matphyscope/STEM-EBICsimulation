"""User-defined electron-beam condition.

The beam is characterised by its energy in keV and the number of
primary electrons delivered.  From these we derive:

* the generation rate of electron-hole pairs
  ``n_eh = n_electrons * (E_keV * 1000) / EHP_energy``
* the Kanaya-Okayama bulb radius, which sets the lateral extent of
  generation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from . import constants as C


@dataclass
class BeamCondition:
    energy_keV: float                    # accelerating voltage
    n_electrons: int                     # total primary electrons delivered
    ehp_energy_eV: float = C.EHP_ENERGY_SI_eV

    @property
    def total_eh_pairs_bulk(self) -> float:
        """Total e-h pairs if the whole beam energy were absorbed."""
        return self.n_electrons * self.energy_keV * 1000.0 / self.ehp_energy_eV

    @property
    def total_charge_C(self) -> float:
        """Total primary charge delivered (C)."""
        return self.n_electrons * C.q

    def energy_deposition_fraction(self, thickness_nm: float,
                                    A: float = 28.09, Z: float = 14.0,
                                    rho: float = 2.329) -> float:
        """Fraction of the primary energy deposited in a thin foil.

        In TEM geometry the beam usually goes *through* the sample; only
        a small part of the energy is dumped inside.  The cheapest
        reasonable approximation is a linear stopping-power scaling:
        ``deposit = min(1, thickness / R_KO)``.
        """
        R_KO = kanaya_okayama_range_nm(self.energy_keV, A, Z, rho)
        return float(min(1.0, thickness_nm / max(R_KO, 1.0)))

    def total_eh_pairs(self, thickness_nm: float | None = None,
                        A: float = 28.09, Z: float = 14.0,
                        rho: float = 2.329) -> float:
        """e-h pairs generated **in the foil**.  Bulk if thickness is None."""
        if thickness_nm is None:
            return self.total_eh_pairs_bulk
        frac = self.energy_deposition_fraction(thickness_nm, A, Z, rho)
        return self.total_eh_pairs_bulk * frac

    def effective_bulb_nm(self, thickness_nm: float | None = None,
                           A: float = 28.09, Z: float = 14.0,
                           rho: float = 2.329) -> float:
        """Lateral bulb radius used in the EBIC convolution.

        For bulk samples this is the Kanaya-Okayama range.  For thin
        foils (``thickness << R_KO``) the lateral spread is dominated
        by small-angle scattering inside the foil; we cap the bulb at
        ``3 * thickness`` in that regime.
        """
        R_KO = kanaya_okayama_range_nm(self.energy_keV, A, Z, rho)
        if thickness_nm is None:
            return R_KO
        return float(min(R_KO, max(3.0 * thickness_nm, 1.0)))


def kanaya_okayama_range_nm(E_keV: float, A: float = 28.09,
                             Z: float = 14.0, rho: float = 2.329) -> float:
    """K-O range in **nm**.  Defaults are for silicon."""
    return 0.0276 * A * E_keV ** 1.67 / (Z ** 0.889 * rho) * 1000.0
