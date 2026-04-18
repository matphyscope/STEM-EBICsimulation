"""End-to-end EBIC / SEEBIC simulation for the example TEM sample.

Example parameters (all user-specified)
---------------------------------------
* Image              : image.png (3 stacked regions)
* Scale-bar image    : scalebar.png  (black bar = 10 um)
* Thickness (Z)      : 100 nm
* Region assignment  : top (yellow) = #1 Pt,
                       middle (blue) = #2 SIMS  (Si doping from CSV),
                       bottom (orange) = #3 Al
* SIMS placement     : surface at x = 10 um, covering y = 0 .. 100 um,
                       depth direction +x.  Both the P- and N-type
                       profiles share this placement.
* Substrate          : P-type, 1e15 cm^-3
* Electron beam      : 200 keV, 1000 electrons
"""
from __future__ import annotations
import os
import numpy as np

from ebic_sim import (materials, image_model, sims, beam, circuit,
                       physics, visualization as viz)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "outputs")
NUM_OUT = os.path.join(OUT, "numerical")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Image + scale  -> model
# ---------------------------------------------------------------------------
model = image_model.build_model(
    image_path=os.path.join(HERE, "image.png"),
    scalebar_path=os.path.join(HERE, "scalebar.png"),
    n_clusters=1,                     # we'll set regions manually below
    thickness_nm=100.0,
    bar_length_um=10.0,
)
print(f"nm / pixel = {model.nm_per_pixel:.3f}")
print(f"image size : {model.shape} pixels  ->  "
      f"{model.shape[1]*model.nm_per_pixel/1000:.1f} x "
      f"{model.shape[0]*model.nm_per_pixel/1000:.1f} um")


# ---------------------------------------------------------------------------
# 2.  Region numbering  (user-specified).  The image has three stacked
# stripes - yellow (top), blue (middle), orange (bottom).  We pick
# them by colour so minor image-edge variations don't matter.
# ---------------------------------------------------------------------------
model.clear_regions()
model.add_region_from_color(1, rgb=(250, 190,  30))   # yellow top
model.add_region_from_color(2, rgb=( 70, 115, 200))   # blue middle
model.add_region_from_color(3, rgb=(230, 130,  40))   # orange bottom
print(f"regions found: {model.region_ids()}")


# ---------------------------------------------------------------------------
# 3. Material assignment (user-specified)
# ---------------------------------------------------------------------------
model.set_material(1, "Pt")     # top contact
model.set_material(2, "SIMS")   # semiconductor - doping from SIMS
model.set_material(3, "Al")     # bottom contact

mat_table = materials.load_material_table(
    os.path.join(HERE, "Material_table_for_ebic_cal.csv"))
print("Materials loaded:", list(mat_table.keys()))


# ---------------------------------------------------------------------------
# 4. SIMS profiles + placement
# ---------------------------------------------------------------------------
p_prof = sims.load_profile(os.path.join(HERE, "SIMSPdata.csv"), kind="P")
n_prof = sims.load_profile(os.path.join(HERE, "SIMSNdata.csv"), kind="N")

# Place both profiles: surface at x = 10 um, y in [0, 100] um,
# depth increases in the +x direction.
placement_P = sims.ProfilePlacement(p_prof, axis="x",
                                     pos_nm=10_000.0,
                                     range_nm=(0.0, 100_000.0),
                                     direction="+x")
placement_N = sims.ProfilePlacement(n_prof, axis="x",
                                     pos_nm=10_000.0,
                                     range_nm=(0.0, 100_000.0),
                                     direction="+x")

# Build the 2-D doping map: only region #2 receives SIMS doping.
Na, Nd, Nnet = sims.build_doping_maps(
    model, [placement_P, placement_N],
    sims_region_ids=[2],
    substrate_type="P", substrate_conc=1e15,
)

# Permittivity map (Si for region #2, NaN elsewhere so fields are only
# computed inside the semiconductor).
eps_r_map = np.full(model.shape, np.nan)
eps_r_map[model.region_mask == 2] = mat_table["Si"]["Relative_permittivity"]


# ---------------------------------------------------------------------------
# 5. Electron-beam condition (user input)
# ---------------------------------------------------------------------------
beam_cfg = beam.BeamCondition(energy_keV=200.0, n_electrons=1000)
R_KO_bulk = beam.kanaya_okayama_range_nm(beam_cfg.energy_keV)
eh_foil = beam_cfg.total_eh_pairs(model.thickness_nm)
eh_bulk = beam_cfg.total_eh_pairs_bulk
dep_frac = beam_cfg.energy_deposition_fraction(model.thickness_nm)
bulb_eff = beam_cfg.effective_bulb_nm(model.thickness_nm)

print(f"\nBeam: {beam_cfg.energy_keV} keV, N_e={beam_cfg.n_electrons}")
print(f"  total charge / shot       = {beam_cfg.total_charge_C*1e18:.2f} aC")
print(f"  K-O range (bulk Si)       = {R_KO_bulk:.0f} nm")
print(f"  foil thickness            = {model.thickness_nm:.0f} nm")
print(f"  energy deposition fraction = {dep_frac*100:.3f} %")
print(f"  e-h pairs in foil         = {eh_foil:.2e} (bulk would be {eh_bulk:.2e})")
print(f"  effective lateral bulb    = {bulb_eff:.0f} nm")


# ---------------------------------------------------------------------------
# 6. Sanity-check mobility / diffusion length (Arora)
# ---------------------------------------------------------------------------
for N in [1e15, 1e17, 1e19]:
    mu_n = materials.arora_mobility(N, "electron")
    mu_p = materials.arora_mobility(N, "hole")
    Ln   = materials.diffusion_length(N, 1e-6, "electron") * 1e4  # um
    Lp   = materials.diffusion_length(N, 1e-6, "hole") * 1e4
    print(f"  N={N:.0e} cm^-3  mu_n={mu_n:6.1f}  mu_p={mu_p:6.1f}"
          f"   Ln={Ln:5.1f} um  Lp={Lp:5.1f} um")


# ---------------------------------------------------------------------------
# 7. 2-D physics: fields, depletion, collection probability, EBIC, SEEBIC
# ---------------------------------------------------------------------------
fields = physics.build_2d_fields(model, Na, Nd, eps_r_map,
                                  placements=[placement_P, placement_N])
print(f"\nJunctions detected: {len(fields['junctions'])}")
for j in fields["junctions"]:
    axis = j["axis"]; pos = j["pos"]
    print(f"  {j['kind']} at {axis}={pos:.0f} nm   "
          f"Vbi={j['Vbi']:.2f} V  W={j['W_nm']:.1f} nm")

P_map = physics.collection_probability_2d(model, Na, Nd, fields["dep_mask"])
ebic  = physics.ebic_scan_2d(model, P_map, beam_cfg, downsample=8)
seebic = physics.seebic_scan_2d(model, fields, beam_cfg, downsample=8)

print(f"\nPeak EBIC:   {ebic['Q_C'].max()*1e18:.2f} aC per beam position"
      f"   (total for all {beam_cfg.n_electrons} electrons)")
print(f"Peak SEEBIC: {seebic['Q_C'].max()*1e18:.2f} aC per beam position")


# ---------------------------------------------------------------------------
# 8. Circuit + contact classification
# ---------------------------------------------------------------------------
circ = circuit.Circuit()
H, W = model.shape
# place ammeter in the middle of Pt (region 1), ground in the middle of Al,
# and a voltage source at the top-left corner (just to exercise the API).
for rid, role, label in [(1, "ammeter", "Pt top"),
                          (3, "ground",  "Al bottom")]:
    ys, xs = np.where(model.region_mask == rid)
    if xs.size:
        circ.add(role=role, region_id=rid,
                 pixel=(int(ys.mean()), int(xs.mean())),
                 label=label)
circ.add(role="voltage", region_id=1, pixel=(0, 0), voltage=0.0, label="V=0")

print("\nContacts:")
for c in circ.contacts:
    if c.region_id in (1, 3):       # metal region
        metal = mat_table["Pt"] if c.region_id == 1 else mat_table["Al"]
        # use local semi doping at the metal/semi boundary
        # (first Si pixel nearest the contact).
        ys, xs = np.where(model.region_mask == 2)
        if xs.size:
            i0 = int(np.argmin(
                (ys - c.pixel[0]) ** 2 + (xs - c.pixel[1]) ** 2))
            N_loc = abs(np.nan_to_num(Nnet[ys[i0], xs[i0]])) or 1e15
            dtype = "N" if np.nan_to_num(Nnet[ys[i0], xs[i0]]) > 0 else "P"
            info = circuit.classify_contact(metal, mat_table["Si"],
                                             N_loc, dtype)
            print(f"  {c.label:12s} role={c.role:8s}  {info['type']}  "
                   f"barrier={info['barrier_eV']:.2f} eV  "
                   f"Vbi={info['built_in_V']:+.2f} V")


# ---------------------------------------------------------------------------
# 9. Save figures + numerical data
# ---------------------------------------------------------------------------
viz.plot_image(model).savefig(os.path.join(OUT, "01_image.png"), dpi=150)
viz.plot_regions(model).savefig(os.path.join(OUT, "02_regions.png"), dpi=150)
viz.plot_doping_map(model, Nnet).savefig(os.path.join(OUT, "03_doping_map.png"), dpi=150)
viz.plot_efield_2d(model, fields).savefig(os.path.join(OUT, "04_efield_2d.png"), dpi=150)
viz.plot_depletion_2d(model, fields).savefig(os.path.join(OUT, "05_depletion_2d.png"), dpi=150)
viz.plot_ebic_2d(model, ebic).savefig(os.path.join(OUT, "06_ebic_2d.png"), dpi=150)
viz.plot_seebic_2d(model, seebic).savefig(os.path.join(OUT, "07_seebic_2d.png"), dpi=150)

# 1-D slice through the SIMS band (centre of the range)
sl = physics.extract_slice_along_placement(model, Na, Nd, eps_r_map,
                                            placement_P)
dep = physics.depletion_region_1d(sl)
ef  = physics.electric_field_1d(sl, dep)
bands = physics.band_diagram_1d(sl, ef["V_V"])
viz.plot_slice_doping(sl, title="1-D doping along depth @ y=50 um"
                         ).savefig(os.path.join(OUT, "08_slice_doping.png"), dpi=150)
viz.plot_efield_1d(sl, ef, dep).savefig(os.path.join(OUT, "09_efield_1d.png"), dpi=150)
viz.plot_band_diagram_1d(sl, bands, dep).savefig(os.path.join(OUT, "10_bands.png"), dpi=150)

# numerical outputs
X_nm, Y_nm = model.xy_grids_nm()
viz.dump_numerics(
    NUM_OUT,
    X_nm=X_nm, Y_nm=Y_nm,
    region_mask=model.region_mask,
    Na=Na, Nd=Nd, Nnet=Nnet,
    Ex=fields["Ex"], Ey=fields["Ey"], E_mag_Vcm=fields["E_Vcm"],
    V_V=fields["V_V"], depletion_mask=fields["dep_mask"].astype(np.int8),
    collection_probability=P_map,
    ebic_x_nm=ebic["x_nm"], ebic_y_nm=ebic["y_nm"], ebic_Q_C=ebic["Q_C"],
    seebic_x_nm=seebic["x_nm"], seebic_y_nm=seebic["y_nm"],
    seebic_Q_C=seebic["Q_C"],
)
print(f"\nFigures   -> {OUT}")
print(f"Numerics  -> {NUM_OUT}   (all.npz + individual .csv)")
