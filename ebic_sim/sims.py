"""Load SIMS doping profiles and apply them to the 2-D sample grid.

Each SIMS file is a 1-D trace of concentration vs depth.  The user
decides where that surface (depth = 0) lies in the sample coordinate
system by specifying

* ``axis``   : "x" or "y"  - the axis on which the surface line lies
* ``pos_nm`` : position on ``axis`` where depth = 0
* ``range_nm``: ``(lo, hi)`` range on the **other** axis where the
                profile is active
* ``direction``: "+x"/"-x"/"+y"/"-y" direction of increasing depth

With ``axis="x"`` the surface is a vertical line at ``x = pos_nm``.
Depth extends along ``+x`` or ``-x`` from that line; the profile is
active for pixels whose ``y`` coordinate falls inside ``range_nm``.

Substrate: the user specifies a substrate type (P or N) + concentration.
Past the point where the reference profile has decayed below a chosen
fraction of its peak, the substrate value is forced.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------
@dataclass
class SIMSProfile:
    depth_nm: np.ndarray                 # 1-D, monotonically increasing
    conc:     np.ndarray                 # cm^-3
    kind:     str                        # 'P' or 'N'
    unit_x:   str = "nm"
    unit_y:   str = "cm^-3"

    def sample(self, d_nm: np.ndarray) -> np.ndarray:
        f = interp1d(self.depth_nm, self.conc, kind="linear",
                     bounds_error=False,
                     fill_value=(self.conc[0], self.conc[-1]))
        return np.clip(f(np.asarray(d_nm)), 0.0, None)


def load_profile(csv_path: str, kind: str) -> SIMSProfile:
    """Read a 2-column CSV; a second row of units is tolerated."""
    df = pd.read_csv(csv_path)
    unit_x, unit_y = "nm", "cm^-3"

    # Check whether the first data row is a units row (non-numeric).
    try:
        float(df.iloc[0, 0])
        float(df.iloc[0, 1])
    except (TypeError, ValueError):
        unit_x = str(df.iloc[0, 0])
        unit_y = str(df.iloc[0, 1])
        df = df.iloc[1:].reset_index(drop=True)

    df = df.astype(float).sort_values(df.columns[0])
    return SIMSProfile(
        depth_nm=df.iloc[:, 0].to_numpy(),
        conc    =df.iloc[:, 1].to_numpy(),
        kind    =kind.upper(),
        unit_x=unit_x, unit_y=unit_y,
    )


# ---------------------------------------------------------------------------
# Placement description
# ---------------------------------------------------------------------------
@dataclass
class ProfilePlacement:
    """Describe how a SIMS profile is laid out on the sample grid.

    Two construction styles are supported:

    1. **Manual**: caller sets ``axis``, ``pos_nm``, ``range_nm``,
       ``direction`` explicitly.  ``range_nm`` is the transverse
       extent on the OTHER axis and is optional for region-scoped
       placements (then the range is taken from the region bounding
       box).

    2. **Region-scoped**: the caller uses :func:`ProfilePlacement.for_region`
       which looks up the region bounding box and picks the surface as
       the edge facing ``direction``.  This is the natural way to say
       "SIMS applies inside region #N, with depth growing toward +y".

    Pixels outside the region index listed in ``build_doping_maps``
    are untouched even if they fall inside ``range_nm``; region
    filtering has precedence.
    """
    profile: SIMSProfile
    axis: str                            # 'x' or 'y'
    pos_nm: float                        # surface position on the axis
    direction: str                       # '+x' / '-x' / '+y' / '-y'
    range_nm: tuple[float, float] | None = None   # transverse extent

    # ------------------------------------------------------------------
    @classmethod
    def for_region(cls, profile: "SIMSProfile", model,
                    region_id: int, direction: str = "+y") -> "ProfilePlacement":
        """Build a placement whose surface sits at the region edge
        facing ``direction``, with the transverse range taken from the
        region's bounding box."""
        bb = model.region_bbox_nm(region_id)
        if bb is None:
            raise ValueError(f"region {region_id} is empty")
        x_lo, x_hi, y_lo, y_hi = bb
        if direction == "+y":
            return cls(profile, axis="y", pos_nm=y_lo, direction="+y",
                        range_nm=(x_lo, x_hi))
        if direction == "-y":
            return cls(profile, axis="y", pos_nm=y_hi, direction="-y",
                        range_nm=(x_lo, x_hi))
        if direction == "+x":
            return cls(profile, axis="x", pos_nm=x_lo, direction="+x",
                        range_nm=(y_lo, y_hi))
        if direction == "-x":
            return cls(profile, axis="x", pos_nm=x_hi, direction="-x",
                        range_nm=(y_lo, y_hi))
        raise ValueError(f"bad direction {direction!r}")

    # ------------------------------------------------------------------
    def depth_map(self, X_nm: np.ndarray, Y_nm: np.ndarray) -> np.ndarray:
        """Return a depth (nm) at every grid point, NaN outside the band
        where the profile is active.  If ``range_nm`` is None the
        transverse dimension is unrestricted - useful for region-scoped
        placements that will be clipped later by ``build_doping_maps``."""
        depth = np.full_like(X_nm, np.nan)
        if self.range_nm is not None:
            lo, hi = self.range_nm
        else:
            lo, hi = -np.inf, np.inf
        if self.direction == "+x":
            d = X_nm - self.pos_nm
            active = (d >= 0) & (Y_nm >= lo) & (Y_nm <= hi)
        elif self.direction == "-x":
            d = self.pos_nm - X_nm
            active = (d >= 0) & (Y_nm >= lo) & (Y_nm <= hi)
        elif self.direction == "+y":
            d = Y_nm - self.pos_nm
            active = (d >= 0) & (X_nm >= lo) & (X_nm <= hi)
        elif self.direction == "-y":
            d = self.pos_nm - Y_nm
            active = (d >= 0) & (X_nm >= lo) & (X_nm <= hi)
        else:
            raise ValueError(f"bad direction {self.direction!r}")
        depth[active] = d[active]
        return depth


# ---------------------------------------------------------------------------
# Apply placements to a sample grid
# ---------------------------------------------------------------------------
def _post_peak_decay_point(profile: SIMSProfile, frac: float) -> float:
    peak_idx = int(profile.conc.argmax())
    peak = profile.conc[peak_idx]
    if peak <= 0:
        return float(profile.depth_nm[-1])
    tail = profile.conc[peak_idx:]
    dropped = np.where(tail < peak * frac)[0]
    if dropped.size == 0:
        return float(profile.depth_nm[-1])
    return float(profile.depth_nm[peak_idx + dropped[0]])


def build_doping_maps(model, placements: list[ProfilePlacement],
                       sims_region_ids: list[int],
                       substrate_type: str | None = None,
                       substrate_conc: float = 0.0,
                       substrate_transition: float = 0.1):
    """Return ``(Na, Nd, Nnet)`` 2-D maps over the whole image.

    Parameters
    ----------
    model : SampleModel
    placements : list of :class:`ProfilePlacement`
    sims_region_ids : list of region ids that receive SIMS doping.
        Pixels outside these regions stay at NaN so later code can
        mask metal/insulator regions.
    substrate_type, substrate_conc : as per the spec - past the decay
        point of *each* placement the substrate doping is forced.

    **Sign convention** (requested by the user): the returned
    ``Nnet`` equals ``Nd - Na``, so N-type appears positive and
    P-type negative.
    """
    H, W = model.shape
    X, Y = model.xy_grids_nm()
    inside = np.isin(model.region_mask, list(sims_region_ids))

    Na = np.zeros((H, W))
    Nd = np.zeros((H, W))

    for pl in placements:
        depth = pl.depth_map(X, Y)
        active = np.isfinite(depth) & inside
        if not active.any():
            continue
        d_vals = depth[active]
        c_vals = pl.profile.sample(d_vals)

        # apply substrate past the decay of this specific profile
        if substrate_type is not None and substrate_conc > 0:
            cutoff = _post_peak_decay_point(pl.profile, substrate_transition)
            deep = d_vals >= cutoff
            if substrate_type.upper().startswith("N"):
                sub_nd = np.where(deep, substrate_conc, 0.0)
                sub_na = np.zeros_like(sub_nd)
            else:
                sub_na = np.where(deep, substrate_conc, 0.0)
                sub_nd = np.zeros_like(sub_na)

            if pl.profile.kind == "P":
                target_na = np.where(deep, sub_na, c_vals)
                target_nd = np.where(deep, sub_nd, 0.0)
            else:
                target_nd = np.where(deep, sub_nd, c_vals)
                target_na = np.where(deep, sub_na, 0.0)
        else:
            if pl.profile.kind == "P":
                target_na, target_nd = c_vals, np.zeros_like(c_vals)
            else:
                target_na, target_nd = np.zeros_like(c_vals), c_vals

        Na[active] += target_na
        Nd[active] += target_nd

    # NaN outside any SIMS region so downstream physics can mask them
    mask = ~inside
    Na[mask] = np.nan
    Nd[mask] = np.nan

    Nnet = Nd - Na           # N positive, P negative
    return Na, Nd, Nnet
