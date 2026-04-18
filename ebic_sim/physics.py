"""Core device physics for the EBIC simulator.

This module solves - along a horizontal line through the sample - the
sequence

    doping(x)   ->   rho(x)   ->   E(x)   ->   V(x)   ->   band(x)

using *integration* (cumulative trapezoid) rather than the textbook
``V_bi / W`` short-cuts, plus the depletion edges found implicitly from
charge balance.  On top of that we compute:

    * Kanaya-Okayama generation range (bulb size)
    * Collection probability using drift + diffusion
    * EBIC and SEEBIC signals

The simulator operates on a 1-D slice because TEM cross-sections are
effectively planar along the electron-beam direction; the caller may
pick which row of the 2-D region mask to slice.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from . import constants as C
from . import materials as _mat


# ---------------------------------------------------------------------------
# Kanaya-Okayama range
# ---------------------------------------------------------------------------
def kanaya_okayama_range_nm(E_keV: float, A: float = 28.09,
                             Z: float = 14.0, rho: float = 2.329) -> float:
    """K-O range in **nm** for beam energy ``E_keV``.

    Defaults are for silicon (A=28.09 g/mol, Z=14, rho=2.329 g/cm^3).
    Formula returns micrometres, so we convert to nanometres.
    """
    R_um = 0.0276 * A * E_keV ** 1.67 / (Z ** 0.889 * rho)
    return R_um * 1000.0


# ---------------------------------------------------------------------------
# 1-D slice extraction
# ---------------------------------------------------------------------------
@dataclass
class Slice1D:
    x_nm:   np.ndarray        # distance along the slice (nm)
    Na:     np.ndarray
    Nd:     np.ndarray
    Nnet:   np.ndarray
    eps_r:  np.ndarray
    region: np.ndarray        # region_id per cell (0 outside sample)


def extract_slice(model, Na_map, Nd_map, eps_r_map,
                  row: int | None = None) -> Slice1D:
    """Pull a 1-D horizontal slice out of the 2-D maps."""
    if row is None:
        row = model.region_mask.shape[0] // 2
    rm  = model.region_mask[row]
    # keep only inside-sample pixels
    in_sample = rm > 0
    cols = np.where(in_sample)[0]
    if cols.size == 0:
        raise ValueError("Slice row is entirely outside the sample")
    x_nm = (cols - cols[0]) * model.nm_per_pixel
    Na = np.nan_to_num(Na_map[row, cols])
    Nd = np.nan_to_num(Nd_map[row, cols])
    return Slice1D(
        x_nm=x_nm, Na=Na, Nd=Nd, Nnet=Nd - Na,
        eps_r=eps_r_map[row, cols], region=rm[cols],
    )


# ---------------------------------------------------------------------------
# Depletion region: integration-based
# ---------------------------------------------------------------------------
def _peak_doping_near(abs_Nnet: np.ndarray, idx: int, direction: int,
                       search_cells: int = 64) -> float:
    """Peak |Nnet| within ``search_cells`` of ``idx`` in the given direction.

    ``direction`` is -1 for "look left", +1 for "look right".  Using a
    peak (rather than a mean) keeps the junction-facing heavy doping
    from being washed out by a distant lightly-doped substrate.
    """
    if direction < 0:
        lo = max(0, idx - search_cells)
        segment = abs_Nnet[lo:idx + 1]
    else:
        hi = min(len(abs_Nnet), idx + 1 + search_cells)
        segment = abs_Nnet[idx + 1:hi]
    if segment.size == 0:
        return 1e14
    return float(segment.max())


def depletion_region(sl: Slice1D, T: float = C.T_DEFAULT,
                      search_nm: float = 600.0) -> dict:
    """Find the depletion region around each junction.

    Every sign change of ``Nnet`` is treated as a metallurgical
    junction.  Two dopings characterising the junction are taken as the
    **peak** |Nnet| values on each side within ``search_nm`` of the
    boundary (peaks rather than means so distant substrate doesn't
    contaminate the estimate).  The total depletion width comes from
    the standard one-sided closed form, then split by charge balance.
    """
    Nnet = sl.Nnet
    absN = np.abs(Nnet)
    x    = sl.x_nm * 1e-7                    # -> cm
    dx_nm = float(np.mean(np.diff(sl.x_nm))) if len(sl.x_nm) > 1 else 1.0
    search_cells = max(4, int(round(search_nm / max(dx_nm, 1e-6))))

    junctions = []
    sign = np.sign(Nnet)
    idxs = np.where(np.diff(sign) != 0)[0]
    for j in idxs:
        if sign[j] == 0 or sign[j + 1] == 0:
            continue
        x_j = 0.5 * (x[j] + x[j + 1])
        eps = sl.eps_r[j] * C.eps0 * 1e-2    # F/cm

        N_left  = _peak_doping_near(absN, j, -1, search_cells)
        N_right = _peak_doping_near(absN, j, +1, search_cells)
        ni  = float(_mat.ni_effective(max(N_left, N_right), T))
        Vbi = (C.kB * T / C.q) * np.log(max(N_left * N_right /
                                             max(ni, 1.0) ** 2, 1.0))

        W_tot = np.sqrt(2.0 * eps * Vbi / C.q *
                        (N_left + N_right) / (N_left * N_right))
        # charge balance: N_left * w_left = N_right * w_right
        w_left  = W_tot * N_right / (N_left + N_right)
        w_right = W_tot - w_left

        # junction type classification
        left_sign  = np.sign(Nnet[max(j - 1, 0)])
        right_sign = np.sign(Nnet[min(j + 2, len(Nnet) - 1)])
        if left_sign < 0 and right_sign > 0:
            kind = "PN"
        elif left_sign > 0 and right_sign < 0:
            kind = "NP"
        else:
            kind = "HL"

        junctions.append(dict(
            index=j, x_nm=x_j * 1e7,
            type=kind, Vbi=Vbi, W_total_nm=W_tot * 1e7,
            # "xp" keeps the original naming: width on the left of junction
            xp_nm=w_left * 1e7, xn_nm=w_right * 1e7,
            N_left=N_left, N_right=N_right,
        ))
    return dict(junctions=junctions)


# ---------------------------------------------------------------------------
# Electric field via Poisson integration of the charge density
# ---------------------------------------------------------------------------
def electric_field(sl: Slice1D, depletion: dict) -> dict:
    """Compute E(x) and V(x) along the slice via Poisson integration.

    Uses the depletion approximation with the junction-facing peak
    dopings (so charge balance across each junction is exact and E
    returns to zero on both neutral sides).  ``rho`` on the N side is
    ``+q N_N`` and on the P side ``-q N_P``, where N_N, N_P come from
    :func:`depletion_region`.
    """
    x_cm = sl.x_nm * 1e-7
    eps  = sl.eps_r * C.eps0 * 1e-2     # F/cm
    rho  = np.zeros_like(sl.Nnet)

    for j in depletion["junctions"]:
        idx = j["index"]
        xj  = x_cm[idx]
        w_left  = j["xp_nm"] * 1e-7
        w_right = j["xn_nm"] * 1e-7
        # figure out which side is N and which is P from the junction kind
        if j["type"] == "NP":
            sign_left, sign_right = +1.0, -1.0
        elif j["type"] == "PN":
            sign_left, sign_right = -1.0, +1.0
        else:   # high-low: use Nnet signs of the first neighbour either side
            sign_left  = float(np.sign(sl.Nnet[max(idx - 1, 0)]))
            sign_right = float(np.sign(sl.Nnet[min(idx + 2, len(sl.Nnet) - 1)]))

        left  = (x_cm >= xj - w_left) & (x_cm <= xj)
        right = (x_cm >= xj) & (x_cm <= xj + w_right)
        rho[left]  += sign_left  * C.q * j["N_left"]
        rho[right] += sign_right * C.q * j["N_right"]

    E = cumulative_trapezoid(rho / eps, x_cm, initial=0.0)    # V/cm
    V = -cumulative_trapezoid(E, x_cm, initial=0.0)           # V
    return dict(E_Vcm=E, V_V=V, rho_Ccm3=rho)


# ---------------------------------------------------------------------------
# Band diagram
# ---------------------------------------------------------------------------
def band_diagram(sl: Slice1D, V: np.ndarray, Eg_base: float = C.EG_SI_300K):
    """Return Ec, Ev, Ef references along the slice.

    The reference is chosen so that the Fermi level is flat at 0 eV in
    the absence of applied bias (equilibrium).  BGN is applied per
    cell through :func:`materials.bandgap`.
    """
    N_abs = np.maximum(np.abs(sl.Nnet), 1.0)
    Eg    = _mat.bandgap(N_abs, Eg_base)
    chi   = C.CHI_SI
    # E_vac = chi + Ec   ;   Ec = -q*V + (reference)
    Ec = -V + chi * 0.0 - 0.0                 # shift is arbitrary
    Ev = Ec - Eg
    Ef = np.zeros_like(Ec)
    return dict(Ec=Ec, Ev=Ev, Ef=Ef, Eg=Eg)


# ---------------------------------------------------------------------------
# Collection probability
# ---------------------------------------------------------------------------
def collection_probability(sl: Slice1D, depletion: dict,
                            tau_s: float = 1e-6,
                            T: float = C.T_DEFAULT) -> np.ndarray:
    """Probability that a carrier generated at position x reaches a contact.

    * Inside any depletion region -> P = 1 (drift-dominated)
    * Outside -> P = exp(-distance_to_nearest_edge / L_local) where L
      comes from Arora diffusion coefficient + ``tau_s``.
    """
    x_nm = sl.x_nm
    P = np.zeros_like(x_nm, dtype=float)

    edges = []
    for j in depletion["junctions"]:
        edges.append((j["x_nm"] - j["xp_nm"], j["x_nm"] + j["xn_nm"]))
    if not edges:
        return P

    # L is minority-carrier diffusion length; use the smaller of
    # electron (in p) and hole (in n) depending on local doping sign
    Nabs = np.maximum(np.abs(sl.Nnet), 1.0)
    L_cm = np.where(sl.Nnet < 0,
                    _mat.diffusion_length(Nabs, tau_s, "electron", T),
                    _mat.diffusion_length(Nabs, tau_s, "hole", T))
    L_nm = L_cm * 1e7

    for i, xi in enumerate(x_nm):
        inside = False
        min_d  = np.inf
        for (a, b) in edges:
            if a <= xi <= b:
                inside = True
                break
            min_d = min(min_d, abs(xi - a), abs(xi - b))
        if inside:
            P[i] = 1.0
        else:
            P[i] = np.exp(-min_d / max(L_nm[i], 1e-6))
    return P


# ---------------------------------------------------------------------------
# EBIC and SEEBIC
# ---------------------------------------------------------------------------
def _gaussian_bulb(x_nm, x0_nm, R_nm):
    sigma = R_nm / 2.355      # FWHM == R
    g = np.exp(-0.5 * ((x_nm - x0_nm) / sigma) ** 2)
    norm = np.trapezoid(g, x_nm)
    return g / norm if norm > 0 else g


def ebic_scan(sl: Slice1D, depletion: dict, beam_kV: float,
              beam_current_A: float = 1e-9, tau_s: float = 1e-6,
              T: float = C.T_DEFAULT, A=28.09, Z=14.0, rho=2.329):
    """Simulate the EBIC line scan.

    At every beam position x0 we deposit ``beam_kV * beam_current`` of
    power, convert that to a generation rate using the material e-h pair
    energy, distribute the generation over a Gaussian bulb of K-O radius,
    and integrate ``g(x) * P(x)`` to get the collected current.
    """
    R_nm = kanaya_okayama_range_nm(beam_kV, A, Z, rho)
    P    = collection_probability(sl, depletion, tau_s=tau_s, T=T)
    G0   = (beam_kV * 1e3 * beam_current_A / C.q) / C.EHP_ENERGY_SI_eV
    #  e-h pairs / s generated by the beam

    I = np.zeros_like(sl.x_nm, dtype=float)
    for i, x0 in enumerate(sl.x_nm):
        g = _gaussian_bulb(sl.x_nm, x0, R_nm)
        I[i] = C.q * G0 * np.trapezoid(g * P, sl.x_nm)
    return dict(I_A=I, R_nm=R_nm, P=P)


def seebic_scan(sl: Slice1D, depletion: dict, ef: dict, beam_kV: float,
                beam_current_A: float = 1e-9, se_yield: float = 0.1):
    """Secondary-electron EBIC - current due to SE escape.

    Much smaller bulb because SEs escape only from within a couple of
    mean-free-paths of the surface.  We approximate this as a delta
    function at the beam position and weight it by (1 + alpha * E_local)
    to make the signal proportional to the local surface electric
    field, which is what drives SEEBIC contrast in practice.
    """
    E_abs = np.abs(ef["E_Vcm"])
    # Normalise the field weighting (unit-less)
    weight = 1.0 + E_abs / (E_abs.max() if E_abs.max() > 0 else 1.0)
    I_prim = beam_current_A          # primary beam current
    I_se   = se_yield * I_prim * weight
    # Sign follows the sign of the lateral field
    I_se *= np.sign(ef["E_Vcm"])
    return dict(I_A=I_se)
