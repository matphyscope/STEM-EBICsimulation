"""End-to-end EBIC simulation using the sample assets in this repo.

Workflow follows the user specification
    1. image  -> modelling (scale from bar, Z=100 nm fixed)
    2. circuit connections (ammeter/voltage/ground)
    3. material assignment (table lookup; 'SIMS' means use SIMS data)
    4. SIMS data                (N-type and P-type profiles)
    5. beam condition           (keV, pA)
    6. compute + visualise      (doping, E-field, depletion, bands,
                                 EBIC, SEEBIC)
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from ebic_sim import materials, image_model, sims, circuit, physics, visualization as viz


HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Build the model from the image
# ---------------------------------------------------------------------------
# The demo image has a single blue rectangle + "10 um" scale bar.
# We mark that region as 'SIMS' so the SIMS files drive its doping.
material_map = {1: "SIMS"}

model = image_model.build_model(
    image_path=os.path.join(HERE, "image.png"),
    material_map=material_map,
    scalebar_um=10.0,
    thickness_nm=100.0,
    n_regions=1,
)
print(f"nm/pixel = {model.nm_per_pixel:.2f}")
print(f"regions  = {model.region_ids}")
for rid in model.region_ids:
    w = model.region_width_nm(rid)
    print(f"  region {rid}  width = {w:.0f} nm   material = {model.material_map[rid]}")


# ---------------------------------------------------------------------------
# 2. Circuit
# ---------------------------------------------------------------------------
H, W = model.region_mask.shape
rid  = model.region_ids[0]

# For the demo put the ammeter at the left edge and ground at the right edge,
# with a 0-V "voltage" contact on top (useful to exercise the book-keeping).
ys, xs = np.where(model.region_mask == rid)
left_pixel  = (int(np.median(ys)), int(xs.min()))
right_pixel = (int(np.median(ys)), int(xs.max()))
top_pixel   = (int(ys.min()), int(np.median(xs)))

circ = circuit.Circuit()
circ.add(role="ammeter", pixel=left_pixel,  region_id=rid, label="Pt / left")
circ.add(role="voltage", pixel=top_pixel,   region_id=rid, voltage=0.0, label="top")
circ.add(role="ground",  pixel=right_pixel, region_id=rid, label="right")


# ---------------------------------------------------------------------------
# 3. Materials table
# ---------------------------------------------------------------------------
mat_table = materials.load_material_table(
    os.path.join(HERE, "Material_table_for_ebic_cal.csv"))


# ---------------------------------------------------------------------------
# 4. SIMS data + substrate handling
# ---------------------------------------------------------------------------
p_prof = sims.load_profile(os.path.join(HERE, "SIMSPdata.csv"), kind="P")
n_prof = sims.load_profile(os.path.join(HERE, "SIMSNdata.csv"), kind="N")

# The SIMS "surface" is taken as the left edge of the region (x = xmin).
surface_pixels = [(int(y), int(xs.min())) for y in np.unique(ys)]
dist_nm = sims.distance_from_surface(model.region_mask,
                                     surface_pixels, model.nm_per_pixel)

# Substrate: user chose a P-type substrate at 1e15 cm^-3 that starts
# deep after the SIMS traces have decayed.
Na, Nd, Nnet = sims.apply_sims_to_region(
    dist_nm, model.region_mask, rid,
    p_profile=p_prof, n_profile=n_prof,
    substrate_type="P", substrate_conc=1e15,
    substrate_transition=0.1,
)

# Permittivity map: use the Si value from the table everywhere in the sample.
eps_r_map = np.full_like(dist_nm, np.nan)
eps_r_map[model.region_mask == rid] = mat_table["Si"]["Relative_permittivity"]


# ---------------------------------------------------------------------------
# 5. Beam conditions
# ---------------------------------------------------------------------------
BEAM_KV   = 30.0        # keV
BEAM_I_A  = 1.0e-9      # A (1 nA probe)


# ---------------------------------------------------------------------------
# 6. Solve along a central horizontal slice
# ---------------------------------------------------------------------------
sl = physics.extract_slice(model, Na, Nd, eps_r_map)
dep = physics.depletion_region(sl)
ef  = physics.electric_field(sl, dep)
bands = physics.band_diagram(sl, ef["V_V"])
ebic  = physics.ebic_scan(sl, dep, beam_kV=BEAM_KV, beam_current_A=BEAM_I_A)
seebic = physics.seebic_scan(sl, dep, ef, beam_kV=BEAM_KV, beam_current_A=BEAM_I_A)

print(f"\nJunctions found: {len(dep['junctions'])}")
for j in dep["junctions"]:
    print(f"  {j['type']} at {j['x_nm']:.0f} nm"
           f"   Vbi={j['Vbi']:.2f} V   W={j['W_total_nm']:.1f} nm"
           f"   xp={j['xp_nm']:.1f}  xn={j['xn_nm']:.1f}")

print(f"Kanaya-Okayama bulb radius: {ebic['R_nm']:.0f} nm")
print(f"Peak EBIC:  {ebic['I_A'].max()*1e9:.3f} nA"
      f"   Peak SEEBIC: {np.abs(seebic['I_A']).max()*1e9:.3f} nA")


# ---------------------------------------------------------------------------
# 7. Report contact types on a per-contact basis
# ---------------------------------------------------------------------------
print("\nContacts:")
for c in circ.contacts:
    local_N = Nnet[c.pixel[0], c.pixel[1]]
    dtype   = "N" if local_N > 0 else "P"
    # For the demo all three contacts sit inside the semiconductor; if
    # they were metal regions we'd use a different material entry.
    info = circuit.classify_contact(mat_table["Pt"], mat_table["Si"],
                                    abs(local_N) or 1e15, dtype)
    print(f"  {c.label:12s} role={c.role:8s} -> {info['type']}  "
          f"barrier={info['barrier_eV']:.2f} eV  Vbi={info['built_in_V']:+.2f} V")


# ---------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------
viz.plot_model(model).savefig(os.path.join(OUT, "01_model.png"), dpi=150)
viz.plot_regions(model).savefig(os.path.join(OUT, "02_regions.png"), dpi=150)
viz.plot_doping_map(model, Nnet).savefig(os.path.join(OUT, "03_doping_map.png"), dpi=150)
viz.plot_slice_doping(sl).savefig(os.path.join(OUT, "04_slice_doping.png"), dpi=150)
viz.plot_efield(sl, ef, dep).savefig(os.path.join(OUT, "05_efield.png"), dpi=150)
viz.plot_depletion(sl, dep).savefig(os.path.join(OUT, "06_depletion.png"), dpi=150)
viz.plot_band_diagram(sl, bands, dep).savefig(os.path.join(OUT, "07_bands.png"), dpi=150)
viz.plot_ebic(sl, ebic, seebic).savefig(os.path.join(OUT, "08_ebic.png"), dpi=150)

print(f"\nFigures written to {OUT}/")
