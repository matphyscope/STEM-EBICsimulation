"""Plotting helpers.  All functions return a ``matplotlib.figure.Figure``."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_model(model, ax=None):
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    ax.imshow(model.image, extent=model.extent_nm)
    ax.set_title("Input image + scale")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    return ax.figure


def plot_regions(model, ax=None):
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    ax.imshow(model.region_mask, extent=model.extent_nm, cmap="tab10")
    ax.set_title(f"Segmented regions ({len(model.region_ids)})")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    return ax.figure


def plot_doping_map(model, Nnet_map, ax=None):
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    # symmetric log: negative (P) blue, positive (N) red
    abs_max = np.nanmax(np.abs(Nnet_map))
    if not np.isfinite(abs_max) or abs_max == 0:
        abs_max = 1.0
    im = ax.imshow(Nnet_map, extent=model.extent_nm, cmap="RdBu_r",
                    vmin=-abs_max, vmax=abs_max)
    plt.colorbar(im, ax=ax, label="Net doping  Nd-Na  (cm⁻³)")
    ax.set_title("Doping map")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    return ax.figure


def plot_slice_doping(sl):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(sl.x_nm, np.maximum(sl.Na, 1), label="Na  (P)", color="C3")
    ax.semilogy(sl.x_nm, np.maximum(sl.Nd, 1), label="Nd  (N)", color="C0")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("Concentration  (cm⁻³)")
    ax.set_title("1-D doping profile")
    ax.legend(); ax.grid(alpha=0.3)
    return fig


def plot_efield(sl, ef, depletion):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    axes[0].plot(sl.x_nm, ef["E_Vcm"] / 1e3)
    axes[0].set_ylabel("E  (kV/cm)")
    axes[0].set_title("Electric field")
    axes[1].plot(sl.x_nm, ef["V_V"])
    axes[1].set_ylabel("V  (V)"); axes[1].set_xlabel("x (nm)")
    axes[1].set_title("Potential")
    for ax in axes:
        for j in depletion["junctions"]:
            ax.axvline(j["x_nm"], color="k", lw=0.7)
            ax.axvspan(j["x_nm"] - j["xp_nm"], j["x_nm"] + j["xn_nm"],
                        color="k", alpha=0.07)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_depletion(sl, depletion):
    fig, ax = plt.subplots(figsize=(7, 2.0))
    in_dep = np.zeros_like(sl.x_nm, dtype=bool)
    for j in depletion["junctions"]:
        in_dep |= ((sl.x_nm >= j["x_nm"] - j["xp_nm"]) &
                    (sl.x_nm <= j["x_nm"] + j["xn_nm"]))
    ax.fill_between(sl.x_nm, 0, in_dep.astype(float), step="mid",
                     color="orange", alpha=0.6)
    for j in depletion["junctions"]:
        ax.axvline(j["x_nm"], color="k", lw=0.6)
        ax.text(j["x_nm"], 1.02, j["type"], ha="center")
    ax.set_title("Depletion region(s)")
    ax.set_xlabel("x (nm)")
    ax.set_yticks([])
    return fig


def plot_band_diagram(sl, bands, depletion):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sl.x_nm, bands["Ec"], label="Ec", color="C0")
    ax.plot(sl.x_nm, bands["Ev"], label="Ev", color="C3")
    ax.plot(sl.x_nm, bands["Ef"], "--", label="Ef", color="k")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("Energy (eV)")
    ax.set_title("Band diagram")
    for j in depletion["junctions"]:
        ax.axvspan(j["x_nm"] - j["xp_nm"], j["x_nm"] + j["xn_nm"],
                    color="k", alpha=0.07)
    ax.legend(); ax.grid(alpha=0.3)
    return fig


def plot_ebic(sl, ebic, seebic=None):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(sl.x_nm, ebic["I_A"] * 1e9, label="EBIC", color="C0")
    if seebic is not None:
        ax2 = ax.twinx()
        ax2.plot(sl.x_nm, seebic["I_A"] * 1e9, label="SEEBIC",
                 color="C1", alpha=0.7)
        ax2.set_ylabel("SEEBIC current (nA)", color="C1")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("EBIC current (nA)", color="C0")
    ax.set_title(
        f"EBIC line scan (bulb R≈{ebic['R_nm']:.0f} nm)"
    )
    ax.grid(alpha=0.3)
    return fig
