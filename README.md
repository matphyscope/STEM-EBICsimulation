# STEM-EBICsimulation

Python package that simulates **2-D EBIC** and **SEEBIC** maps on a
thin (TEM) cross-section starting from

* a **SEM/TEM image** of the sample,
* a **separate scale-bar image** (used to calibrate nm / pixel),
* one or more **SIMS** depth profiles (P- and/or N-type),
* a **material table** (CSV) with permittivity, work function, etc.,
* a user-specified **circuit** (ammeter, voltage, ground) and
* an **electron-beam** condition (keV + number of electrons).

## Workflow

```
image + scalebar  ->  model (nm/px, Z=100nm fixed)
   |
   v
user numbers each region (1, 2, 3, ...) and assigns a material.
"SIMS" is a special marker: that region gets its doping from the
SIMS files.
   |
   v
SIMS profiles placed with
    axis  = "x" or "y"            (axis on which surface line lies)
    pos   = coordinate of surface
    range = (lo, hi) on the other axis
    direction = "+x" / "-x" / "+y" / "-y"   (depth direction)
   |
   v
substrate type + concentration  -->  2-D doping map
        (N-type positive, P-type negative)
   |
   v
2-D physics:  Poisson integration -> E-field, depletion, bands
              Arora mobility -> diffusion length map
              collection probability map
   |
   v
beam (keV, n_electrons)
              |                              |
              v                              v
         Kanaya-Okayama range +         SE yield x primary charge
         thin-foil energy-deposition    modulated by surface V (bipolar)
         correction
              |                              |
              v                              v
          2-D EBIC                        2-D SEEBIC
```

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python run_example.py
```

Outputs (both images and numerical arrays):

```
outputs/
  01_image.png               input image with axes in nm
  02_regions.png             user-numbered regions with material labels
  03_doping_map.png          Nd-Na 2-D map (red=N, blue=P)
  04_efield_2d.png           2-D |E| with junction lines
  05_depletion_2d.png        depletion region(s)
  06_ebic_2d.png             2-D EBIC map (charge per beam position)
  07_seebic_2d.png           2-D SEEBIC (bipolar)
  08_slice_doping.png        1-D doping along the SIMS depth direction
  09_efield_1d.png           1-D E(x) and V(x) for the slice
  10_bands.png               band diagram along the slice
  numerical/
    all.npz                  every 2-D / 1-D array at full resolution
    *.csv                    individual CSV dumps (large maps downsampled)
```

## API overview

```python
from ebic_sim import materials, image_model, sims, beam, circuit, physics
from ebic_sim import visualization as viz

# 1) model + scale
model = image_model.build_model(
    image_path="image.png", scalebar_path="scalebar.png",
    n_clusters=1, thickness_nm=100.0, bar_length_um=10.0)

# 2) user-numbered regions
model.clear_regions()
model.add_region_from_color(1, rgb=(250, 190, 30))   # Pt top
model.add_region_from_color(2, rgb=( 70, 115, 200))  # SIMS middle
model.add_region_from_color(3, rgb=(230, 130, 40))   # Al bottom
# (alternatively use model.add_region_bbox(rid, x_nm=..., y_nm=...) )

# 3) materials
model.set_material(1, "Pt")
model.set_material(2, "SIMS")
model.set_material(3, "Al")
mat = materials.load_material_table("Material_table_for_ebic_cal.csv")

# 4) SIMS profiles + placement
p = sims.load_profile("SIMSPdata.csv", kind="P")
n = sims.load_profile("SIMSNdata.csv", kind="N")
place_P = sims.ProfilePlacement(p, axis="x", pos_nm=10_000,
                                 range_nm=(0, 100_000), direction="+x")
place_N = sims.ProfilePlacement(n, axis="x", pos_nm=10_000,
                                 range_nm=(0, 100_000), direction="+x")
Na, Nd, Nnet = sims.build_doping_maps(
    model, [place_P, place_N], sims_region_ids=[2],
    substrate_type="P", substrate_conc=1e15)

# 5) beam
b = beam.BeamCondition(energy_keV=200.0, n_electrons=1000)

# 6) 2-D physics
eps = ... # build a 2-D permittivity map from the table
fields = physics.build_2d_fields(model, Na, Nd, eps, [place_P, place_N])
P_map  = physics.collection_probability_2d(model, Na, Nd, fields["dep_mask"])
ebic   = physics.ebic_scan_2d(model, P_map, b, downsample=8)
seebic = physics.seebic_scan_2d(model, fields, b, downsample=8)
```

## Physics (with reference)

| Quantity | Model | Where |
|----------|-------|-------|
| mobility | Arora-Hauser-Roulston (Si, 300 K) | `materials.arora_mobility` |
| diffusion length | `L = sqrt((kT/q) * mu * tau)` | `materials.diffusion_length` |
| BGN | Slotboom; active at N >= 1e18 cm^-3, P or N | `materials.bgn_delta_eg` |
| work function (semi) | `W = chi + (Ec-Ef)` or `chi + Eg - (Ev-Ef)` | `materials.work_function_semi` |
| contact type | ΔW criterion, ohmic / Schottky + barrier | `circuit.classify_contact` |
| depletion | charge balance + Vbi with peak doping on each side | `physics.depletion_region_1d` |
| E, V | Poisson integration (`cumtrapz`, not averaging) | `physics.electric_field_1d` |
| K-O range | `0.0276 * A * E^1.67 / (Z^0.889 * rho)` um | `beam.kanaya_okayama_range_nm` |
| thin-foil energy | linear stopping-power scaling `t/R_KO` | `beam.BeamCondition` |
| collection prob. | `P = 1` inside depletion, else `exp(-d / L(r))` | `physics.collection_probability_2d` |
| EBIC | Gaussian bulb * P, integrated, times `q*N_eh` | `physics.ebic_scan_2d` |
| SEEBIC | `se_yield * Q_beam * (V - <V>)/max`  (bipolar) | `physics.seebic_scan_2d` |

## Notes

* Sign convention for the net doping map: ``N_d - N_a`` so N-type
  appears positive and P-type negative (user request).
* EBIC shows a broad peak at the junction whose width is set by the
  diffusion length; SEEBIC shows a sharp bipolar S-shape across the
  junction set by the surface-potential change.
* At high beam energies (e.g. 200 keV) in a thin TEM foil only a
  small fraction of the primary energy is deposited.  The simulator
  reports the deposition fraction together with the total e-h pairs
  actually created in the sample.
