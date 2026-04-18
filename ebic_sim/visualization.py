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
    # annotate each region with its id + material at the centre of its bbox
    for rid in model.region_ids():
        ys, xs = np.where(model.region_mask == rid)
        p = model.nm_per_pixel
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
    axes[0].set_ylabel("E (kV/cm)")
    axes[0].set_title("Electric field (1-D slice)")
    axes[1].plot(sl.x_nm, ef["V_V"])
    axes[1].set_ylabel("V (V)"); axes[1].set_xlabel("depth (nm)")
    axes[1].set_title("Potential")
    for ax in axes:
        for j in dep["junctions"]:
            ax.axvline(j["x_nm"], color="k", lw=0.6)
            ax.axvspan(j["x_nm"] - j["w_left_nm"], j["x_nm"] + j["w_right_nm"],
                        color="k", alpha=0.08)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_efield_2d(model, fields):
    fig, ax = plt.subplots(figsize=(5, 8))
    im = ax.imshow(fields["E_Vcm"] / 1e3, extent=model.extent_nm,
                    cmap="magma")
    plt.colorbar(im, ax=ax, label="|E| (kV/cm)")
    for j in fields["junctions"]:
        if j["axis"] == "x":
            ax.axvline(j["pos"], color="cyan", lw=0.8, alpha=0.6)
        else:
            ax.axhline(j["pos"], color="cyan", lw=0.8, alpha=0.6)
    ax.set_title("Electric field (2-D)")
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
