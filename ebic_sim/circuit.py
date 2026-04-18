"""Electrical circuit: contact points and contact-type classification.

Three contact roles are supported:
    * ``"ammeter"`` - current-sensing, tied to ground (EBIC output)
    * ``"voltage"`` - applied bias source, with a voltage value (V)
    * ``"ground"`` - reference potential

Each contact is a pixel coordinate in the model image and is associated
with a specific region id.  When that region is a metal the contact-
vs-semiconductor work-function difference is used to decide whether the
interface is ohmic or Schottky, with the Schottky barrier height.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from . import constants as C
from . import materials as _mat


@dataclass
class Contact:
    role: str                 # 'ammeter' | 'voltage' | 'ground'
    pixel: tuple[int, int]    # (row, col)
    region_id: int
    voltage: float = 0.0      # V, used when role == 'voltage'
    label: str = ""


@dataclass
class Circuit:
    contacts: list[Contact] = field(default_factory=list)

    def add(self, **kw):
        self.contacts.append(Contact(**kw))

    def by_role(self, role):
        return [c for c in self.contacts if c.role == role]


# ---------------------------------------------------------------------------
# Contact type (ohmic / Schottky) from the work-function difference
# ---------------------------------------------------------------------------
def classify_contact(metal_entry: dict, semi_entry: dict,
                     N_local: float, dtype: str) -> dict:
    """Return contact-type description at a metal/semiconductor interface."""
    W_m = metal_entry.get("Work_function")
    if W_m is None:
        raise ValueError("Metal work function missing from table")

    chi = semi_entry.get("Electron_Affinity") or C.CHI_SI
    Eg0 = semi_entry.get("Bandgap") or C.EG_SI_300K
    Eg  = float(_mat.bandgap(N_local, Eg0))
    W_s = float(_mat.work_function_semi(N_local, dtype,
                                         chi=chi, base_Eg=Eg0))

    # Schottky barrier heights (ideal, no Fermi-level pinning)
    if dtype.lower().startswith("n"):
        phi_b = W_m - chi                   # barrier on electrons
        ohmic = W_m < W_s                   # metal Ef above semi CB edge
    else:
        phi_b = chi + Eg - W_m              # barrier on holes
        ohmic = W_m > W_s
    return dict(
        W_metal=W_m, W_semi=W_s, chi=chi, Eg=Eg,
        barrier_eV=float(phi_b), type="ohmic" if ohmic else "schottky",
        built_in_V=float(W_m - W_s),
    )
