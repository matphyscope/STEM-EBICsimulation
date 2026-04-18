"""Device-physics solver for 2-D TEM-EBIC simulation.

The simulator works on a Cartesian grid of the sample.  For each
placement of a SIMS profile the doping varies along the depth
direction; junctions are therefore lines perpendicular to that
direction.  We reuse the 1-D depletion / E-field solution along the
depth axis and extrude it across the placement band to build 2-D
fields.

Exposed quantities
------------------
* 1-D slice helpers :  :func:`extract_slice` + the classic
  :func:`depletion_region_1d`, :func:`electric_field_1d`
* 2-D maps         :  :func:`build_2d_fields`, :func:`collection_probability_2d`
* EBIC / SEEBIC    :  :func:`ebic_scan_2d`, :func:`seebic_scan_2d`
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import distance_transform_edt

from . import constants as C
from . import materials as _mat
from .beam import kanaya_okayama_range_nm, BeamCondition
from .sims import ProfilePlacement


# ---------------------------------------------------------------------------
# 1-D slice object
# ---------------------------------------------------------------------------
@dataclass
class Slice1D:
    x_nm:   np.ndarray
    Na:     np.ndarray
    Nd:     np.ndarray
    Nnet:   np.ndarray
    eps_r:  np.ndarray


def extract_slice_along_placement(model, Na_map, Nd_map, eps_r_map,
                                   placement: ProfilePlacement) -> Slice1D:
    """Pull a 1-D slice along the depth direction of a placement.

    The slice is taken at the midpoint of the transverse range, clipped
    to the actual image bounds so the row/column index is always valid.
    """
    X, Y = model.xy_grids_nm()
    H, W = model.shape
    px = model.nm_per_pixel
    lo, hi = placement.range_nm

    if placement.axis == "x":
        y_min, y_max = 0.0, H * px
        mid = 0.5 * (max(lo, y_min) + min(hi, y_max))
        row = int(np.clip(round(mid / px - 0.5), 0, H - 1))
        x_nm = X[row, :]
        Na = Na_map[row, :]; Nd = Nd_map[row, :]; eps = eps_r_map[row, :]
        if placement.direction == "+x":
            keep = x_nm >= placement.pos_nm
            coord_full = x_nm - placement.pos_nm
        else:
            keep = x_nm <= placement.pos_nm
            coord_full = placement.pos_nm - x_nm
        order = np.argsort(coord_full[keep])
        coord = coord_full[keep][order]
        Na = np.nan_to_num(Na[keep][order])
        Nd = np.nan_to_num(Nd[keep][order])
        eps = eps[keep][order]
    else:
        x_min, x_max = 0.0, W * px
        mid = 0.5 * (max(lo, x_min) + min(hi, x_max))
        col = int(np.clip(round(mid / px - 0.5), 0, W - 1))
        y_nm = Y[:, col]
        Na = Na_map[:, col]; Nd = Nd_map[:, col]; eps = eps_r_map[:, col]
        if placement.direction == "+y":
            keep = y_nm >= placement.pos_nm
            coord_full = y_nm - placement.pos_nm
        else:
            keep = y_nm <= placement.pos_nm
            coord_full = placement.pos_nm - y_nm
        order = np.argsort(coord_full[keep])
        coord = coord_full[keep][order]
        Na = np.nan_to_num(Na[keep][order])
        Nd = np.nan_to_num(Nd[keep][order])
        eps = eps[keep][order]

    # drop cells outside the sample (no permittivity assigned)
    good = np.isfinite(eps)
    if not np.any(good):
        return Slice1D(x_nm=np.array([0.0]), Na=np.array([0.0]),
                        Nd=np.array([0.0]), Nnet=np.array([0.0]),
                        eps_r=np.array([C.EPS_R_SI]))
    coord = coord[good]; Na = Na[good]; Nd = Nd[good]; eps = eps[good]
    coord = coord - coord.min()
    return Slice1D(x_nm=coord, Na=Na, Nd=Nd,
                    Nnet=Nd - Na, eps_r=eps)


# ---------------------------------------------------------------------------
# 1-D depletion region (junction-by-junction)
# ---------------------------------------------------------------------------
def _peak_near(absN, idx, direction, n_cells):
    if direction < 0:
        seg = absN[max(0, idx - n_cells):idx + 1]
    else:
        seg = absN[idx + 1:min(len(absN), idx + 1 + n_cells)]
    if seg.size == 0:
        return 1e14
    return float(seg.max())


def depletion_region_1d(sl: Slice1D, T: float = C.T_DEFAULT,
                         search_nm: float = 600.0) -> dict:
    Nnet = sl.Nnet
    absN = np.abs(Nnet)
    x = sl.x_nm * 1e-7
    dx = float(np.mean(np.diff(sl.x_nm))) if len(sl.x_nm) > 1 else 1.0
    n = max(4, int(round(search_nm / max(dx, 1e-6))))

    junctions = []
    sign = np.sign(Nnet)
    for j in np.where(np.diff(sign) != 0)[0]:
        if sign[j] == 0 or sign[j + 1] == 0:
            continue
        eps = sl.eps_r[j] * C.eps0 * 1e-2
        N_left  = _peak_near(absN, j, -1, n)
        N_right = _peak_near(absN, j, +1, n)
        ni  = float(_mat.ni_effective(max(N_left, N_right), T))
        Vbi = (C.kB * T / C.q) * np.log(max(N_left * N_right /
                                             max(ni, 1.0) ** 2, 1.0))
        W = np.sqrt(2.0 * eps * Vbi / C.q *
                    (N_left + N_right) / (N_left * N_right))
        w_left  = W * N_right / (N_left + N_right)
        w_right = W - w_left

        left_sign  = np.sign(Nnet[max(j - 1, 0)])
        right_sign = np.sign(Nnet[min(j + 2, len(Nnet) - 1)])
        if left_sign > 0 and right_sign < 0:
            kind = "NP"
        elif left_sign < 0 and right_sign > 0:
            kind = "PN"
        else:
            kind = "HL"
        junctions.append(dict(
            index=j, x_nm=0.5 * (sl.x_nm[j] + sl.x_nm[j + 1]),
            type=kind, Vbi=Vbi, W_total_nm=W * 1e7,
            w_left_nm=w_left * 1e7, w_right_nm=w_right * 1e7,
            N_left=N_left, N_right=N_right,
            left_sign=float(left_sign), right_sign=float(right_sign),
        ))
    return dict(junctions=junctions)


# ---------------------------------------------------------------------------
# 1-D field
# ---------------------------------------------------------------------------
def electric_field_1d(sl: Slice1D, depletion: dict) -> dict:
    x = sl.x_nm * 1e-7
    eps = sl.eps_r * C.eps0 * 1e-2
    rho = np.zeros_like(sl.Nnet)
    for j in depletion["junctions"]:
        idx = j["index"]
        xj = x[idx]
        wl = j["w_left_nm"] * 1e-7
        wr = j["w_right_nm"] * 1e-7
        left  = (x >= xj - wl) & (x <= xj)
        right = (x >= xj) & (x <= xj + wr)
        rho[left]  += j["left_sign"]  * C.q * j["N_left"]
        rho[right] += j["right_sign"] * C.q * j["N_right"]
    E = cumulative_trapezoid(rho / eps, x, initial=0.0)
    V = -cumulative_trapezoid(E, x, initial=0.0)
    return dict(E_Vcm=E, V_V=V, rho_Ccm3=rho)


def band_diagram_1d(sl: Slice1D, V: np.ndarray,
                     Eg_base: float = C.EG_SI_300K):
    Eg = _mat.bandgap(np.maximum(np.abs(sl.Nnet), 1.0), Eg_base)
    Ec = -V
    return dict(Ec=Ec, Ev=Ec - Eg, Ef=np.zeros_like(Ec), Eg=Eg)


# ---------------------------------------------------------------------------
# 2-D extrusion across a placement band
# ---------------------------------------------------------------------------
def _placement_key(pl: ProfilePlacement):
    return (pl.axis, float(pl.pos_nm),
             float(pl.range_nm[0]), float(pl.range_nm[1]),
             pl.direction)


def build_2d_fields(model, Na_map, Nd_map, eps_r_map,
                     placements: list[ProfilePlacement],
                     T: float = C.T_DEFAULT):
    """Compute 2-D E, V, depletion-mask and junction list.

    Placements that share (axis, pos, range, direction) are deduped so
    we don't double-count the same physical junction when both a P-
    and an N-type SIMS profile live on the same surface.  For each
    unique placement key we take a 1-D slice along the depth axis,
    run the 1-D Poisson solve, and extrude the result across the
    transverse range.
    """
    H, W = model.shape
    X, Y = model.xy_grids_nm()

    Ex = np.zeros((H, W))
    Ey = np.zeros((H, W))
    V2 = np.zeros((H, W))
    dep_mask = np.zeros((H, W), dtype=bool)
    junctions_2d = []      # list of (x_nm, y_nm, type, Vbi, W, ...)

    seen: dict = {}
    for pl in placements:
        seen.setdefault(_placement_key(pl), pl)

    for pl in seen.values():
        sl = extract_slice_along_placement(model, Na_map, Nd_map,
                                            eps_r_map, pl)
        dep = depletion_region_1d(sl, T=T)
        ef  = electric_field_1d(sl, dep)
        lo, hi = pl.range_nm

        # build a depth map (NaN outside the active band) and
        # interpolate 1-D E, V along depth coordinate
        depth = pl.depth_map(X, Y)
        mask = np.isfinite(depth)
        d_vals = depth[mask]

        # 1-D E along depth (V/cm -> V/m for the stored field? keep V/cm)
        E1 = np.interp(d_vals, sl.x_nm, ef["E_Vcm"], left=0.0, right=0.0)
        V1 = np.interp(d_vals, sl.x_nm, ef["V_V"],  left=0.0,
                        right=float(ef["V_V"][-1]))
        V2[mask] += V1
        # direction-aware assignment
        if pl.direction == "+x":
            Ex[mask] += E1
        elif pl.direction == "-x":
            Ex[mask] -= E1
        elif pl.direction == "+y":
            Ey[mask] += E1
        elif pl.direction == "-y":
            Ey[mask] -= E1

        # depletion mask per placement
        for j in dep["junctions"]:
            in_dep_1d = ((d_vals >= j["x_nm"] - j["w_left_nm"]) &
                          (d_vals <= j["x_nm"] + j["w_right_nm"]))
            tmp = np.zeros_like(mask)
            tmp[mask] = in_dep_1d
            dep_mask |= tmp

            # remember the junction location for overlay plotting
            # junction line passes through depth = j["x_nm"] along the
            # transverse axis
            tr_lo, tr_hi = pl.range_nm
            if pl.axis == "x":
                # surface at x=pos, depth along x ; junction at
                # x = pos +/- x_nm depending on direction
                xj = (pl.pos_nm + j["x_nm"]) if pl.direction == "+x" else (pl.pos_nm - j["x_nm"])
                junctions_2d.append(dict(kind=j["type"], axis="x", pos=xj,
                                          span=(tr_lo, tr_hi),
                                          Vbi=j["Vbi"], W_nm=j["W_total_nm"]))
            else:
                yj = (pl.pos_nm + j["x_nm"]) if pl.direction == "+y" else (pl.pos_nm - j["x_nm"])
                junctions_2d.append(dict(kind=j["type"], axis="y", pos=yj,
                                          span=(tr_lo, tr_hi),
                                          Vbi=j["Vbi"], W_nm=j["W_total_nm"]))

    E_mag = np.sqrt(Ex ** 2 + Ey ** 2)
    return dict(Ex=Ex, Ey=Ey, E_Vcm=E_mag, V_V=V2,
                 dep_mask=dep_mask, junctions=junctions_2d)


# ---------------------------------------------------------------------------
# 2-D collection probability
# ---------------------------------------------------------------------------
def collection_probability_2d(model, Na_map, Nd_map, dep_mask,
                               tau_s: float = 1.0e-6,
                               T: float = C.T_DEFAULT) -> np.ndarray:
    """P(r) = 1 inside depletion, ``exp(-d / L(r))`` outside.

    Distance ``d`` is a 2-D Euclidean distance (in nm) to the nearest
    depletion pixel.  Diffusion length is taken per pixel from the
    minority carrier in the locally dominant dopant (Arora + Einstein).
    """
    px = model.nm_per_pixel
    d_nm = distance_transform_edt(~dep_mask) * px     # nm

    Nabs = np.nan_to_num(np.abs(Nd_map - Na_map))
    Nabs = np.maximum(Nabs, 1.0)
    # Nnet sign map (vectorised)
    Nnet = Nd_map - Na_map
    L_cm = np.where(np.nan_to_num(Nnet) < 0,
                     _mat.diffusion_length(Nabs, tau_s, "electron", T),
                     _mat.diffusion_length(Nabs, tau_s, "hole", T))
    L_nm = L_cm * 1e7
    L_nm = np.where(np.isfinite(L_nm) & (L_nm > 0), L_nm, np.inf)

    P = np.where(dep_mask, 1.0, np.exp(-d_nm / np.maximum(L_nm, 1e-6)))
    # outside sample -> 0
    P[model.region_mask == 0] = 0.0
    return P


# ---------------------------------------------------------------------------
# 2-D EBIC scan with Kanaya-Okayama generation
# ---------------------------------------------------------------------------
def _gaussian_kernel_2d(sigma_px: float, cutoff: float = 3.0):
    half = int(np.ceil(cutoff * sigma_px))
    if half < 1:
        half = 1
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    g = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma_px ** 2)
    return g / g.sum()


def ebic_scan_2d(model, P_map: np.ndarray, beam: BeamCondition,
                  ko_A: float = 28.09, ko_Z: float = 14.0,
                  ko_rho: float = 2.329,
                  downsample: int = 4,
                  use_thin_foil: bool = True):
    """2-D EBIC map.

    The collection probability is convolved with a Gaussian whose
    lateral size is the effective bulb radius (K-O for bulk, capped at
    ~3 x thickness for thin TEM foils).  The result at every pixel is
    the probability-weighted overlap of a generation bulb centred
    there.  The collected charge is ``q * N_eh * overlap`` where
    ``N_eh`` is the number of e-h pairs actually produced in the foil
    (not in a bulk sample).

    Returns a dict with the 2-D **collected charge per beam position**
    in C, the bulb radius, and the (downsampled) x/y grids in nm.
    """
    from scipy.signal import fftconvolve

    R_nm_bulk = kanaya_okayama_range_nm(beam.energy_keV, ko_A, ko_Z, ko_rho)
    if use_thin_foil:
        R_nm = beam.effective_bulb_nm(model.thickness_nm,
                                        A=ko_A, Z=ko_Z, rho=ko_rho)
        N_eh = beam.total_eh_pairs(model.thickness_nm,
                                     A=ko_A, Z=ko_Z, rho=ko_rho)
    else:
        R_nm = R_nm_bulk
        N_eh = beam.total_eh_pairs_bulk

    sigma_nm = R_nm / 2.355
    sigma_px = sigma_nm / model.nm_per_pixel

    step = max(1, int(downsample))
    Pc = P_map[::step, ::step]
    sigma_c = sigma_px / step

    kernel = _gaussian_kernel_2d(max(sigma_c, 0.5))
    overlap = fftconvolve(Pc, kernel, mode="same")
    overlap = np.clip(overlap, 0.0, 1.0)

    Q = C.q * N_eh * overlap

    px = model.nm_per_pixel
    x_c = (np.arange(Pc.shape[1]) + 0.5) * px * step
    y_c = (np.arange(Pc.shape[0]) + 0.5) * px * step

    return dict(Q_C=Q, R_nm=R_nm, R_KO_bulk_nm=R_nm_bulk,
                 sigma_nm=sigma_nm, N_eh=N_eh,
                 x_nm=x_c, y_nm=y_c, P_coarse=Pc)


# ---------------------------------------------------------------------------
# 2-D SEEBIC
# ---------------------------------------------------------------------------
def seebic_scan_2d(model, fields: dict, beam: BeamCondition,
                    se_yield: float = 0.1, downsample: int = 4):
    """Secondary-electron EBIC map (bipolar contrast).

    In SEEBIC the escaping SE current is modulated by the local
    surface potential: above a positively-biased region the SEs face
    an additional attractive (or repulsive) surface voltage and the
    collected current differs from its equilibrium value.  The
    contrast is therefore proportional to the local electrostatic
    potential, taken relative to its mean so the trace is bipolar
    across the junction (classic S-shape along a line scan).
    """
    step = max(1, int(downsample))
    V = fields["V_V"][::step, ::step]
    # reference = mean potential inside the sample
    in_sample = model.region_mask[::step, ::step] > 0
    if in_sample.any():
        V_ref = V[in_sample].mean()
    else:
        V_ref = 0.0
    dV = V - V_ref
    Vmax = np.max(np.abs(dV))
    if Vmax == 0:
        Vmax = 1.0
    contrast = dV / Vmax

    Q = se_yield * beam.total_charge_C * contrast

    px = model.nm_per_pixel
    x_c = (np.arange(V.shape[1]) + 0.5) * px * step
    y_c = (np.arange(V.shape[0]) + 0.5) * px * step
    return dict(Q_C=Q, x_nm=x_c, y_nm=y_c, contrast=contrast, dV=dV)
