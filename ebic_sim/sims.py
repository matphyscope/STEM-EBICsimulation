"""Load SIMS doping profiles and map them onto the sample grid.

SIMS CSV layout::

    X, Y
    depth[nm], concentration[cm^-3]

Two profiles are expected - P- and N-type - each giving a 1-D
concentration vs depth trace anchored at the chosen "surface" in the
model.  The sign convention:

* Positive values in P-type -> acceptors Na
* Positive values in N-type -> donors Nd
* net doping N_net = Nd - Na  (positive = n-type, negative = p-type)

The substrate flag forces the deep part of the profile to the supplied
substrate concentration beyond the point where the SIMS trace drops
below ``substrate_transition`` of its peak.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


@dataclass
class SIMSProfile:
    """A single SIMS trace after reading and cleanup."""
    depth_nm: np.ndarray       # 1-D, monotonically increasing
    conc:     np.ndarray       # 1-D, cm^-3
    kind:     str              # 'P' or 'N'

    def sample(self, d_nm: np.ndarray) -> np.ndarray:
        """Linear interpolation with flat extrapolation outside range."""
        f = interp1d(self.depth_nm, self.conc, kind="linear",
                     bounds_error=False,
                     fill_value=(self.conc[0], self.conc[-1]))
        return np.clip(f(d_nm), 0.0, None)


def load_profile(csv_path: str, kind: str) -> SIMSProfile:
    df = pd.read_csv(csv_path)
    df = df.sort_values(df.columns[0])
    return SIMSProfile(
        depth_nm=df.iloc[:, 0].to_numpy(dtype=float),
        conc    =df.iloc[:, 1].to_numpy(dtype=float),
        kind    =kind.upper(),
    )


# ---------------------------------------------------------------------------
# Distance-from-surface map
# ---------------------------------------------------------------------------
def distance_from_surface(region_mask: np.ndarray, surface_pixels,
                          nm_per_pixel: float) -> np.ndarray:
    """Euclidean distance (nm) from each sample pixel to the nearest
    listed ``surface_pixels`` entry.  Pixels outside the sample get NaN.
    """
    from scipy.ndimage import distance_transform_edt

    seed = np.ones_like(region_mask, dtype=bool)
    for (r, c) in surface_pixels:
        seed[r, c] = False
    dist_px = distance_transform_edt(seed)
    dist_nm = dist_px * nm_per_pixel
    dist_nm[region_mask == 0] = np.nan
    return dist_nm


def _substrate_boundary_nm(profile: SIMSProfile,
                            substrate_transition: float = 0.1) -> float:
    """Depth at which the profile has decayed past the peak below the
    transition fraction.  We search **after** the peak so that a leading
    zero region (no doping near the surface) does not trigger the
    substrate prematurely."""
    peak_idx = int(profile.conc.argmax())
    peak = profile.conc[peak_idx]
    if peak <= 0:
        return float(profile.depth_nm[-1])
    thresh = peak * substrate_transition
    tail = profile.conc[peak_idx:]
    idx = np.where(tail < thresh)[0]
    if idx.size == 0:
        return float(profile.depth_nm[-1])
    return float(profile.depth_nm[peak_idx + idx[0]])


# ---------------------------------------------------------------------------
# Net doping map
# ---------------------------------------------------------------------------
def apply_sims_to_region(distance_nm: np.ndarray,
                         region_mask: np.ndarray,
                         region_id: int,
                         p_profile: SIMSProfile | None,
                         n_profile: SIMSProfile | None,
                         substrate_type: str | None = None,
                         substrate_conc: float = 0.0,
                         substrate_transition: float = 0.1
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Na, Nd, net_doping) arrays for all pixels in the region.

    Outside the region the values are NaN so downstream code can mask
    metallic/insulator areas.  Substrate handling: past the point where
    the shallowest SIMS trace has decayed below the transition
    fraction, the specified substrate concentration is enforced.
    """
    shape = region_mask.shape
    Na = np.full(shape, np.nan)
    Nd = np.full(shape, np.nan)

    in_region = region_mask == region_id
    d = distance_nm[in_region]

    na = p_profile.sample(d) if p_profile is not None else np.zeros_like(d)
    nd = n_profile.sample(d) if n_profile is not None else np.zeros_like(d)

    if substrate_type is not None and substrate_conc > 0:
        # where substrate dominates
        ref = p_profile or n_profile
        transition = _substrate_boundary_nm(ref, substrate_transition)
        sub_mask = d >= transition
        if substrate_type.lower().startswith("n"):
            nd = np.where(sub_mask, substrate_conc, nd)
            na = np.where(sub_mask, 0.0, na)
        else:
            na = np.where(sub_mask, substrate_conc, na)
            nd = np.where(sub_mask, 0.0, nd)

    Na[in_region] = na
    Nd[in_region] = nd
    return Na, Nd, Nd - Na
