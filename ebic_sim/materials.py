"""Material table + doping-dependent physics.

Loads the CSV material table where semiconductor-specific fields may
contain the placeholder ``Cal`` - these are computed from the local
doping via the Arora mobility model and the Slotboom BGN model.

Only the *doping-independent* parameters (chi, eps_r, effective mass,
metal work function) are stored in the table.  Everything else
(mobility, diffusion length, work function of doped Si, Eg with BGN)
is returned by the helper functions as a function of N.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import constants as C


# ---------------------------------------------------------------------------
# Table loading
# ---------------------------------------------------------------------------
def load_material_table(csv_path: str) -> dict:
    """Parse the transposed material CSV into a dict keyed by material name.

    File layout: first column = category name, second column = unit,
    remaining columns = one per material.  The first data row holds
    ``Material_name`` and its values are used as the dict keys.  Cells
    whose string value is ``Cal`` are stored as None so downstream code
    computes them from the local doping.
    """
    raw = pd.read_csv(csv_path, header=0)

    # Category labels live in the first column
    categories = raw.iloc[:, 0].astype(str).str.strip().tolist()

    # First data row gives the material names for each of the remaining columns
    name_row = raw.iloc[0, 2:].tolist()
    material_cols = list(raw.columns[2:])
    name_of = {col: str(name).strip() for col, name in zip(material_cols, name_row)}

    materials: dict[str, dict] = {}
    for col in material_cols:
        mat_name = name_of[col]
        entry: dict = {}
        # skip the Material_name row itself (i == 0)
        for i, cat in enumerate(categories):
            if i == 0:
                continue
            val = raw[col].iloc[i]
            key = cat
            if isinstance(val, str) and val.strip().lower() == "cal":
                entry[key] = None
            else:
                try:
                    entry[key] = float(val)
                except (TypeError, ValueError):
                    entry[key] = val if isinstance(val, str) and val.strip() else None
        # source file has a typo "Effective_h)mass"
        if "Effective_h)mass" in entry:
            entry["Effective_h_mass"] = entry.pop("Effective_h)mass")
        materials[mat_name] = entry
    return materials


# ---------------------------------------------------------------------------
# Arora mobility model (Si, 300 K)
# ---------------------------------------------------------------------------
# Arora, Hauser & Roulston, IEEE TED 29, 292 (1982).  T = 300 K defaults.
_ARORA = {
    "electron": dict(mu_min=88.0,  mu_max=1252.0, Nref=1.26e17, alpha=0.88),
    "hole":     dict(mu_min=54.3,  mu_max=407.0,  Nref=2.35e17, alpha=0.88),
}

def arora_mobility(N: np.ndarray | float, carrier: str = "electron",
                   T: float = C.T_DEFAULT) -> np.ndarray | float:
    """Arora majority-carrier mobility (cm^2/V/s) as a function of doping."""
    p = _ARORA[carrier]
    t = T / 300.0
    mu_min = p["mu_min"] * t ** -0.57
    mu_max = p["mu_max"] * t ** -2.33
    Nref   = p["Nref"]   * t ** 2.4
    alpha  = p["alpha"]  * t ** -0.146
    N = np.maximum(np.asarray(N, dtype=float), 1.0)   # avoid /0
    return mu_min + (mu_max - mu_min) / (1.0 + (N / Nref) ** alpha)


def diffusion_coefficient(N, carrier="electron", T=C.T_DEFAULT):
    """Einstein relation D = (kT/q) * mu, returned in cm^2/s."""
    Vt = C.kB * T / C.q     # thermal voltage (V)
    return Vt * arora_mobility(N, carrier, T)


def diffusion_length(N, tau_s: float = 1.0e-6, carrier="electron",
                     T=C.T_DEFAULT):
    """L = sqrt(D*tau)  [cm].  tau default is 1 us minority lifetime."""
    D = diffusion_coefficient(N, carrier, T)   # cm^2/s
    return np.sqrt(D * tau_s)                  # cm


# ---------------------------------------------------------------------------
# Bandgap narrowing - Slotboom-like, enabled above 1e18 cm^-3
# ---------------------------------------------------------------------------
def bgn_delta_eg(N, threshold: float = 1.0e18,
                 V0: float = 0.00692, Nref: float = 1.3e17):
    """Heavy-doping bandgap narrowing (eV).

    Zero below the threshold, Slotboom expression above it.  Works
    identically for P- and N-type since only |N| matters for the rigid
    shift used in device-level simulations.
    """
    N  = np.asarray(N, dtype=float)
    dE = np.zeros_like(N)
    mask = N >= threshold
    if np.any(mask):
        ratio = np.log(N[mask] / Nref)
        dE[mask] = V0 * (ratio + np.sqrt(ratio ** 2 + 0.5))
    return dE


def intrinsic_carrier_density(T: float = C.T_DEFAULT, Eg: float = C.EG_SI_300K,
                              me: float = C.ME_SI, mh: float = C.MH_SI):
    """ni (cm^-3) from effective DOS - used for BGN-corrected ni,eff."""
    Vt = C.kB * T / C.q
    # Nc, Nv effective DOS (cm^-3)
    pref = 2.0 * (2.0 * np.pi * C.m0 * C.kB * T / C.h ** 2) ** 1.5 * 1e-6
    Nc = pref * me ** 1.5
    Nv = pref * mh ** 1.5
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2.0 * Vt))


def ni_effective(N, T=C.T_DEFAULT):
    """ni,eff = ni0 * exp(dEg/2kT)   - Slotboom rigid-shift approximation."""
    Vt  = C.kB * T / C.q
    ni0 = intrinsic_carrier_density(T)
    return ni0 * np.exp(bgn_delta_eg(N) / (2.0 * Vt))


def bandgap(N, base_Eg: float = C.EG_SI_300K):
    """Eg with BGN applied."""
    return base_Eg - bgn_delta_eg(N)


# ---------------------------------------------------------------------------
# Work function for doped silicon (when material table says ``Cal``)
# ---------------------------------------------------------------------------
def work_function_semi(N, dtype: str, chi: float = C.CHI_SI,
                        base_Eg: float = C.EG_SI_300K, T: float = C.T_DEFAULT,
                        me: float = C.ME_SI, mh: float = C.MH_SI):
    """Work function (eV) for doped Si.

    For N-type:  W = chi + (Ec - Ef) = chi + Vt*ln(Nc/Nd)
    For P-type:  W = chi + Eg - (Ev side) = chi + Eg - Vt*ln(Nv/Na)

    Bandgap includes BGN via :func:`bandgap`.
    """
    Vt = C.kB * T / C.q
    pref = 2.0 * (2.0 * np.pi * C.m0 * C.kB * T / C.h ** 2) ** 1.5 * 1e-6
    Nc = pref * me ** 1.5
    Nv = pref * mh ** 1.5
    Eg = bandgap(N, base_Eg)
    N  = np.maximum(np.asarray(N, dtype=float), 1.0)
    if dtype.lower().startswith("n"):
        return chi + Vt * np.log(Nc / N)
    return chi + Eg - Vt * np.log(Nv / N)


# ---------------------------------------------------------------------------
# Convenience: look up properties respecting ``Cal`` placeholders
# ---------------------------------------------------------------------------
def resolve_semiconductor(entry: dict, N, dtype: str, T=C.T_DEFAULT):
    """Return a dict of resolved physical parameters for a semiconductor cell."""
    chi   = entry.get("Electron_Affinity") or C.CHI_SI
    eps_r = entry.get("Relative_permittivity") or C.EPS_R_SI
    me    = entry.get("Effective_e_mass") or C.ME_SI
    mh    = entry.get("Effective_h_mass") or C.MH_SI
    Eg0   = entry.get("Bandgap") or C.EG_SI_300K
    Eg    = bandgap(N, Eg0)
    W     = (entry.get("Work_function")
             or work_function_semi(N, dtype, chi=chi, base_Eg=Eg0, T=T,
                                   me=me, mh=mh))
    return dict(
        chi=chi, eps_r=eps_r, me=me, mh=mh,
        Eg=Eg, W=W, dtype=dtype,
        mu_n=arora_mobility(N, "electron", T),
        mu_p=arora_mobility(N, "hole", T),
        Ln=diffusion_length(N, carrier="electron", T=T),
        Lp=diffusion_length(N, carrier="hole", T=T),
        ni_eff=ni_effective(N, T),
    )
