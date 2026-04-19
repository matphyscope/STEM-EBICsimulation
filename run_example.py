"""End-to-end EBIC / SEEBIC simulation demo.

Edit the CONFIG section at the top to re-run on a different sample.
Any PNG / TIF / TIFF / JPG / BMP file works for ``IMAGE_PATH`` - the
scale bar in the bottom-left corner of the image sets the physical
size and ``SCALE_BAR_UM`` tells the simulator the bar's physical
length (change this if your scale bar is not 10 um).
"""
from __future__ import annotations
import os
import numpy as np

from ebic_sim import (materials, image_model, sims, beam, circuit,
                       physics, visualization as viz)

HERE = os.path.dirname(os.path.abspath(__file__))


# =========================================================================
# CONFIG  -- edit these numbers to match your experiment
# =========================================================================
IMAGE_PATH          = os.path.join(HERE, "image.png")    # .png / .tif ok
MATERIAL_TABLE_CSV  = os.path.join(HERE, "Material_table_for_ebic_cal.csv")
SIMS_P_CSV          = os.path.join(HERE, "SIMSPdata.csv")
SIMS_N_CSV          = os.path.join(HERE, "SIMSNdata.csv")

SCALE_BAR_UM        = 10.0        # physical length of the scale bar
THICKNESS_NM        = 100.0       # TEM foil thickness (Z)

# region numbering is automatic (top -> bottom = 1, 2, 3).  Attach a
# material to each.  Use "SIMS" to tell the SIMS module "this region
# is doped by the SIMS profiles".
REGION_MATERIALS    = {1: "Pt", 2: "SIMS", 3: "W"}

# Profile depth direction inside the SIMS region.  The surface (depth=0)
# is taken as the region edge facing the opposite direction, so
# ``"+y"`` means "depth grows downward; surface at the top of the SIMS
# region".
SIMS_REGION_ID      = 2
SIMS_DIRECTION      = "+y"

# Substrate type + concentration, applied past the SIMS profile's
# post-peak decay point.
SUBSTRATE_TYPE      = "P"
SUBSTRATE_CONC      = 1e15        # cm^-3

# Electron beam condition
BEAM_KV             = 200.0
BEAM_N_ELECTRONS    = 1000

# Applied external bias on the 'voltage' contact (V).  Zero => pure
# built-in field; positive forward-biases the first junction.
APPLIED_BIAS_V      = 0.0

OUT_DIR             = os.path.join(HERE, "outputs")
NUM_OUT             = os.path.join(OUT_DIR, "numerical")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================================
# 1. Load image + auto-detect scale
# =========================================================================
model = image_model.build_model(
    image_path   = IMAGE_PATH,
    n_clusters   = len(REGION_MATERIALS),
    thickness_nm = THICKNESS_NM,
    bar_length_um= SCALE_BAR_UM,
    auto_number_by="y",
)
for rid, name in REGION_MATERIALS.items():
    model.set_material(rid, name)

print(f"nm / pixel = {model.nm_per_pixel:.3f}   "
      f"(scale bar {model.scalebar['length_px']} px = {SCALE_BAR_UM} um)")
print(f"image size : {model.shape} pixels  ->  "
      f"{model.shape[1]*model.nm_per_pixel/1000:.1f} x "
      f"{model.shape[0]*model.nm_per_pixel/1000:.1f} um")
print(f"regions    : {model.region_ids()}  ({REGION_MATERIALS})")


# =========================================================================
# 2. Material table
# =========================================================================
mat_table = materials.load_material_table(MATERIAL_TABLE_CSV)
print(f"materials  : {list(mat_table.keys())}")


# =========================================================================
# 3. SIMS profiles placed INSIDE the SIMS region
# =========================================================================
p_prof = sims.load_profile(SIMS_P_CSV, kind="P")
n_prof = sims.load_profile(SIMS_N_CSV, kind="N")

# Region-scoped placement: surface = top of region #2, range = region bbox,
# depth grows in +y into the semiconductor.
placement_P = sims.ProfilePlacement.for_region(p_prof, model,
                                                 region_id=SIMS_REGION_ID,
                                                 direction=SIMS_DIRECTION)
placement_N = sims.ProfilePlacement.for_region(n_prof, model,
                                                 region_id=SIMS_REGION_ID,
                                                 direction=SIMS_DIRECTION)

# Doping map - only pixels inside region_id=SIMS_REGION_ID are touched.
sims_rid = [rid for rid, m in REGION_MATERIALS.items() if m == "SIMS"]
Na, Nd, Nnet = sims.build_doping_maps(
    model, [placement_P, placement_N],
    sims_region_ids=sims_rid,
    substrate_type=SUBSTRATE_TYPE, substrate_conc=SUBSTRATE_CONC,
)

# Permittivity map.  Use Si inside the SIMS region; NaN everywhere else
# so the Poisson solver knows where the semiconductor ends.
eps_r_map = np.full(model.shape, np.nan)
for rid in sims_rid:
    eps_r_map[model.region_mask == rid] = mat_table["Si"]["Relative_permittivity"]


# =========================================================================
# 4. Electron beam
# =========================================================================
beam_cfg = beam.BeamCondition(energy_keV=BEAM_KV, n_electrons=BEAM_N_ELECTRONS)
R_KO_bulk = beam.kanaya_okayama_range_nm(beam_cfg.energy_keV)
eh_foil   = beam_cfg.total_eh_pairs(model.thickness_nm)
dep_frac  = beam_cfg.energy_deposition_fraction(model.thickness_nm)
bulb_eff  = beam_cfg.effective_bulb_nm(model.thickness_nm)
print(f"\nBeam: {beam_cfg.energy_keV} keV, N_e={beam_cfg.n_electrons}")
print(f"  total charge / shot        = {beam_cfg.total_charge_C*1e18:.2f} aC")
print(f"  K-O range (bulk Si)        = {R_KO_bulk:.0f} nm")
print(f"  foil thickness             = {model.thickness_nm:.0f} nm")
print(f"  energy deposition fraction = {dep_frac*100:.3f} %")
print(f"  e-h pairs in foil          = {eh_foil:.2e}")
print(f"  effective lateral bulb     = {bulb_eff:.0f} nm")


# =========================================================================
# 5. Built-in fields (plus optional applied bias)
# =========================================================================
fields = physics.build_2d_fields(model, Na, Nd, eps_r_map,
                                  placements=[placement_P, placement_N],
                                  applied_bias_V=APPLIED_BIAS_V)
print(f"\nJunctions detected: {len(fields['junctions'])}  "
      f"(V_bias = {APPLIED_BIAS_V} V)")
for j in fields["junctions"]:
    print(f"  {j['kind']} at {j['axis']}={j['pos']:.0f} nm   "
          f"Vbi={j['Vbi']:.2f} V  W={j['W_nm']:.1f} nm")

P_map  = physics.collection_probability_2d(model, Na, Nd, fields["dep_mask"])
ebic   = physics.ebic_scan_2d(model, P_map, beam_cfg, downsample=8)
seebic = physics.seebic_scan_2d(model, fields, beam_cfg, downsample=8)

print(f"\nPeak EBIC  : {ebic['Q_C'].max()*1e18:.2f} aC / beam position")
print(f"Peak SEEBIC: {np.max(np.abs(seebic['Q_C']))*1e18:.2f} aC / beam position"
      f"  (signed)")


# =========================================================================
# 6. Circuit / contact classification
# =========================================================================
circ = circuit.Circuit()
metal_roles = {1: ("ammeter", "Pt top"), 3: ("ground", "W bottom")}
for rid, (role, label) in metal_roles.items():
    ys, xs = np.where(model.region_mask == rid)
    if xs.size:
        circ.add(role=role, region_id=rid,
                 pixel=(int(ys.mean()), int(xs.mean())), label=label)

print("\nContacts:")
for c in circ.contacts:
    metal_name  = model.materials.get(c.region_id, "?")
    metal_entry = mat_table.get(metal_name)
    if metal_entry is None or metal_entry.get("Type") != "Metal":
        continue
    ys, xs = np.where(np.isin(model.region_mask, sims_rid))
    if not xs.size:
        continue
    i0 = int(np.argmin((ys - c.pixel[0])**2 + (xs - c.pixel[1])**2))
    N_loc = abs(np.nan_to_num(Nnet[ys[i0], xs[i0]])) or 1e15
    dtype = "N" if np.nan_to_num(Nnet[ys[i0], xs[i0]]) > 0 else "P"
    info = circuit.classify_contact(metal_entry, mat_table["Si"], N_loc, dtype)
    print(f"  {c.label:12s} role={c.role:8s}  {info['type']}  "
          f"barrier={info['barrier_eV']:.2f} eV  "
          f"Vbi={info['built_in_V']:+.2f} V")


# =========================================================================
# 7. Figures + numerical dumps
# =========================================================================
viz.plot_image(model).savefig(os.path.join(OUT_DIR, "01_image.png"), dpi=150)
viz.plot_regions(model).savefig(os.path.join(OUT_DIR, "02_regions.png"), dpi=150)
viz.plot_doping_map(model, Nnet).savefig(os.path.join(OUT_DIR, "03_doping_map.png"), dpi=150)
viz.plot_efield_2d(model, fields).savefig(os.path.join(OUT_DIR, "04_Ebi_2d.png"), dpi=150)
viz.plot_depletion_2d(model, fields).savefig(os.path.join(OUT_DIR, "05_depletion.png"), dpi=150)
viz.plot_ebic_2d(model, ebic).savefig(os.path.join(OUT_DIR, "06_ebic_2d.png"), dpi=150)
viz.plot_seebic_2d(model, seebic).savefig(os.path.join(OUT_DIR, "07_seebic_2d.png"), dpi=150)

sl = physics.extract_slice_along_placement(model, Na, Nd, eps_r_map, placement_P)
dep = physics.depletion_region_1d(sl)
if APPLIED_BIAS_V:
    for j in dep["junctions"]:
        f = np.sqrt(max(j["Vbi"] - APPLIED_BIAS_V, 1e-3) / j["Vbi"])
        j["w_left_nm"] *= f; j["w_right_nm"] *= f; j["W_total_nm"] *= f
ef = physics.electric_field_1d(sl, dep)
bands = physics.band_diagram_1d(sl, ef["V_V"])
viz.plot_slice_doping(sl, title="1-D doping along depth"
                         ).savefig(os.path.join(OUT_DIR, "08_slice_doping.png"), dpi=150)
viz.plot_efield_1d(sl, ef, dep).savefig(os.path.join(OUT_DIR, "09_Ebi_1d.png"), dpi=150)
viz.plot_band_diagram_1d(sl, bands, dep).savefig(os.path.join(OUT_DIR, "10_bands.png"), dpi=150)

X_nm, Y_nm = model.xy_grids_nm()
viz.dump_numerics(
    NUM_OUT,
    X_nm=X_nm, Y_nm=Y_nm,
    region_mask=model.region_mask,
    Na=Na, Nd=Nd, Nnet=Nnet,
    Ex_bi=fields["Ex"], Ey_bi=fields["Ey"], E_bi_Vcm=fields["E_Vcm"],
    V_bi_V=fields["V_V"], depletion_mask=fields["dep_mask"].astype(np.int8),
    collection_probability=P_map,
    ebic_x_nm=ebic["x_nm"], ebic_y_nm=ebic["y_nm"], ebic_Q_C=ebic["Q_C"],
    seebic_x_nm=seebic["x_nm"], seebic_y_nm=seebic["y_nm"],
    seebic_Q_C=seebic["Q_C"],
)
print(f"\nFigures   -> {OUT_DIR}")
print(f"Numerics  -> {NUM_OUT}  (all.npz + individual .csv)")
