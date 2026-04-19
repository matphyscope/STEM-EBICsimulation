"""Plotting and numerical-dump helpers.

Every ``save_*`` helper returns a ``matplotlib.figure.Figure`` so the
caller can further style or save it; the ``dump_numerics`` helper
writes all the 2-D maps to a single ``.npz`` archive plus individual
CSV files for easy inspection.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm


# ---------------------------------------------------------------------------
# Geometry / segmentation
# ---------------------------------------------------------------------------
def plot_image(model):
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.imshow(model.image, extent=model.extent_nm)
    ax.set_title("Input image")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    fig.tight_layout()
    return fig


def plot_regions(model):
    fig, ax = plt.subplots(figsize=(4, 8))
    cmap = plt.get_cmap("tab10")
    im = ax.imshow(model.region_mask, extent=model.extent_nm, cmap=cmap,
                    vmin=0, vmax=9)
    ax.set_title("User-assigned regions")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    # annotate each region with its id + material at its largest-contour
    # centroid (robust against stray dust pixels)
    p = model.nm_per_pixel
    best = getattr(model, "largest_contour_per_cluster", lambda: {})()
    # map cluster -> region via current region_mask value
    for cid, (_cnt, _area, cx, cy) in best.items():
        # find rid by looking up the region_mask value at that centroid
        rid = int(model.region_mask[int(cy), int(cx)])
        if rid == 0:
            continue
        mat = model.materials.get(rid, "?")
        ax.text(cx * p, cy * p, f"#{rid}  {mat}", ha="center", va="center",
                 color="white", fontweight="bold")
    # fallback for regions added via bbox/color (no contour record)
    for rid in model.region_ids():
        if any(int(model.region_mask[int(cy), int(cx)]) == rid
                for _cid, (_c, _a, cx, cy) in best.items()):
            continue
        ys, xs = np.where(model.region_mask == rid)
        if not xs.size:
            continue
        cy = 0.5 * (ys.min() + ys.max()) * p
        cx = 0.5 * (xs.min() + xs.max()) * p
        mat = model.materials.get(rid, "?")
        ax.text(cx, cy, f"#{rid}  {mat}", ha="center", va="center",
                 color="white", fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Doping
# ---------------------------------------------------------------------------
def plot_doping_map(model, Nnet):
    """N-type = positive (red), P-type = negative (blue)."""
    fig, ax = plt.subplots(figsize=(5, 8))
    vmax = np.nanmax(np.abs(Nnet))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    lin_thresh = max(vmax * 1e-3, 1.0)
    im = ax.imshow(Nnet, extent=model.extent_nm, cmap="RdBu_r",
                    norm=SymLogNorm(linthresh=lin_thresh,
                                     vmin=-vmax, vmax=vmax))
    plt.colorbar(im, ax=ax,
                  label="Net doping  N_d - N_a  (cm^-3)\n(+ = N-type, - = P-type)")
    ax.set_title("Doping map")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    fig.tight_layout()
    return fig


def plot_slice_doping(sl, title="1-D doping profile"):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(sl.x_nm, np.maximum(sl.Na, 1), label="Na (P)", color="C3")
    ax.semilogy(sl.x_nm, np.maximum(sl.Nd, 1), label="Nd (N)", color="C0")
    ax.set_xlabel("depth (nm)"); ax.set_ylabel("Concentration (cm^-3)")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# E-field / depletion / bands (per-slice)
# ---------------------------------------------------------------------------
def plot_efield_1d(sl, ef, dep):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    axes[0].plot(sl.x_nm, ef["E_Vcm"] / 1e3)
    axes[0].set_ylabel("E_bi (kV/cm)")
    axes[0].set_title("Built-in electric field (1-D slice)")
    axes[1].plot(sl.x_nm, ef["V_V"])
    axes[1].set_ylabel("V_bi (V)"); axes[1].set_xlabel("depth (nm)")
    axes[1].set_title("Built-in potential")
    for ax in axes:
        for j in dep["junctions"]:
            ax.axvline(j["x_nm"], color="k", lw=0.6)
            ax.axvspan(j["x_nm"] - j["w_left_nm"], j["x_nm"] + j["w_right_nm"],
                        color="k", alpha=0.08)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_efield_2d(model, fields, n_arrows_across_dep: int = 6,
                    n_arrows_along: int = 8):
    """|E| colour map overlaid with direction arrows drawn densely
    inside the depletion region(s).

    Arrow spacing auto-adapts to the depletion width so the field
    direction is visible even when the depletion is only a few tens
    of nanometres wide.
    """
    fig, ax = plt.subplots(figsize=(5, 8))
    im = ax.imshow(fields["E_Vcm"] / 1e3, extent=model.extent_nm,
                    cmap="magma")
    plt.colorbar(im, ax=ax, label="|E_bi| (kV/cm)")

    Ex = fields["Ex"]; Ey = fields["Ey"]
    dep = fields["dep_mask"]
    H, W = Ex.shape
    px = model.nm_per_pixel

    # build a short list of arrows by scanning columns / rows
    # covering the depletion mask densely.  Arrows are drawn as unit
    # vectors (direction only) so they stay legible regardless of the
    # absolute field magnitude.
    if dep.any():
        ys, xs = np.where(dep)
        y_lo, y_hi = ys.min(), ys.max()
        x_lo, x_hi = xs.min(), xs.max()
        sy = max(1, (y_hi - y_lo) // max(n_arrows_across_dep, 1))
        sx = max(1, (x_hi - x_lo) // max(n_arrows_along, 1))
        YY, XX = np.mgrid[y_lo:y_hi + 1:sy, x_lo:x_hi + 1:sx]
        U = Ex[YY, XX].astype(float)
        V = Ey[YY, XX].astype(float)
        mag = np.hypot(U, V)
        m = mag > 0
        if m.any():
            Un = U[m] / mag[m]
            Vn = V[m] / mag[m]
            ax.quiver(XX[m] * px, YY[m] * px, Un, Vn,
                       color="cyan", pivot="mid",
                       angles="xy", scale=25, scale_units="width",
                       width=0.005, headwidth=4, headlength=5)

    for j in fields["junctions"]:
        if j["axis"] == "x":
            ax.axvline(j["pos"], color="white", lw=0.6, alpha=0.6)
        else:
            ax.axhline(j["pos"], color="white", lw=0.6, alpha=0.6)
    ax.set_title("Built-in electric field  |E_bi| + direction")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    fig.tight_layout()
    return fig


def plot_efield_streamlines(model, fields):
    """Streamline plot of the E vector field (direction only, coloured
    by |E|).  Complements :func:`plot_efield_2d` when arrow density
    makes quiver plots cluttered."""
    fig, ax = plt.subplots(figsize=(5, 8))
    H, W = fields["Ex"].shape
    px = model.nm_per_pixel
    x = (np.arange(W) + 0.5) * px
    y = (np.arange(H) + 0.5) * px
    strm = ax.streamplot(x, y, fields["Ex"], fields["Ey"],
                          color=fields["E_Vcm"] / 1e3,
                          cmap="magma", density=1.2, linewidth=1.0)
    plt.colorbar(strm.lines, ax=ax, label="|E_bi| (kV/cm)")
    ax.set_xlim(0, W * px); ax.set_ylim(H * px, 0)
    ax.set_title("Electric field streamlines")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    fig.tight_layout()
    return fig


def plot_depletion_2d(model, fields):
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.imshow(fields["dep_mask"].astype(float), extent=model.extent_nm,
               cmap="YlOrRd", vmin=0, vmax=1)
    for j in fields["junctions"]:
        if j["axis"] == "x":
            ax.axvline(j["pos"], color="k", lw=0.6)
            ax.text(j["pos"], 0.95 * (j["span"][1]), f"{j['kind']}",
                     color="k", ha="left", va="top")
        else:
            ax.axhline(j["pos"], color="k", lw=0.6)
            ax.text(j["span"][1], j["pos"], f"{j['kind']}",
                     color="k", ha="right", va="top")
    ax.set_title("Depletion region(s)")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    fig.tight_layout()
    return fig


def plot_band_diagram_1d(sl, bands, dep):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sl.x_nm, bands["Ec"], label="Ec", color="C0")
    ax.plot(sl.x_nm, bands["Ev"], label="Ev", color="C3")
    ax.plot(sl.x_nm, bands["Ef"], "--", label="Ef", color="k")
    for j in dep["junctions"]:
        ax.axvspan(j["x_nm"] - j["w_left_nm"], j["x_nm"] + j["w_right_nm"],
                    color="k", alpha=0.08)
    ax.set_xlabel("depth (nm)"); ax.set_ylabel("Energy (eV)")
    ax.set_title("Band diagram")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# EBIC / SEEBIC (2-D maps)
# ---------------------------------------------------------------------------
def plot_ebic_2d(model, ebic, title="EBIC"):
    fig, ax = plt.subplots(figsize=(5, 8))
    im = ax.imshow(ebic["Q_C"] * 1e15,
                    extent=[ebic["x_nm"][0], ebic["x_nm"][-1],
                            ebic["y_nm"][-1], ebic["y_nm"][0]],
                    cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Collected charge per beam pos. (fC)")
    ax.set_title(f"{title} (R_KO={ebic['R_nm']:.0f} nm)")
    ax.set_xlabel("beam x (nm)"); ax.set_ylabel("beam y (nm)")
    fig.tight_layout()
    return fig


def plot_seebic_2d(model, seebic):
    fig, ax = plt.subplots(figsize=(5, 8))
    vmax = np.max(np.abs(seebic["Q_C"])) * 1e15
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(seebic["Q_C"] * 1e15,
                    extent=[seebic["x_nm"][0], seebic["x_nm"][-1],
                            seebic["y_nm"][-1], seebic["y_nm"][0]],
                    cmap="RdBu_r", aspect="auto",
                    vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="SE charge per beam pos. (fC, signed)")
    ax.set_title("SEEBIC (bipolar)")
    ax.set_xlabel("beam x (nm)"); ax.set_ylabel("beam y (nm)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Numerical dump
# ---------------------------------------------------------------------------
def dump_numerics(out_dir, *, csv_max_points: int = 200_000, **arrays):
    """Save every array to ``all.npz`` (compressed) plus individual CSVs.

    Large 2-D arrays are downsampled for CSV so the files stay
    manageable (``<csv_max_points`` cells).  The full-resolution data is
    always available in the npz archive.
    """
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "all.npz"),
                         **{k: np.asarray(v) for k, v in arrays.items()})
    for name, arr in arrays.items():
        arr = np.asarray(arr)
        path = os.path.join(out_dir, f"{name}.csv")
        if arr.ndim == 2 and arr.size > csv_max_points:
            step = int(np.ceil(np.sqrt(arr.size / csv_max_points)))
            np.savetxt(path, arr[::step, ::step], delimiter=",",
                        header=f"downsampled by {step}x{step}")
        elif arr.ndim <= 2:
            np.savetxt(path, arr, delimiter=",")
        else:
            np.savetxt(path, arr.reshape(arr.shape[0], -1), delimiter=",")
