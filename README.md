# STEM-EBICsimulation

Python package that simulates **EBIC** and **SEEBIC** line scans on a
thin (TEM) cross-section, starting from

* a **SEM/TEM image** of the sample (with a scale bar),
* a **SIMS** P-type and N-type depth profile,
* a **material table** (CSV) with permittivity / work function / etc.,
* a user-specified **circuit** (ammeter, voltage source, ground) and
  **electron-beam condition** (keV, pA).

The workflow follows exactly the spec in the issue description:

```
image ─▶ region segmentation + scale  (Z = 100 nm fixed)
        │
        ▼
 circuit contacts       material table (Cal values resolved per-cell)
        │                         │
        ▼                         ▼
 SIMS profiles ─────▶ 2-D doping map (with substrate, interpolated)
        │
        ▼
 E-field (Poisson integration) ─▶ depletion ─▶ band diagram
        │                           │
        ▼                           ▼
 Kanaya-Okayama generation  +  collection probability (L from Arora)
        │
        ▼
 EBIC, SEEBIC line scans
```

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python run_example.py
```

Outputs are written to `outputs/`:

| file | content |
|------|---------|
| `01_model.png`        | source image with axes in nm |
| `02_regions.png`      | colour-segmented region mask |
| `03_doping_map.png`   | 2-D Nd-Na map (cm⁻³) |
| `04_slice_doping.png` | 1-D Na, Nd along a horizontal slice |
| `05_efield.png`       | E(x) and V(x), depletion bands shaded |
| `06_depletion.png`    | depletion regions + junction type (PN/NP/HL) |
| `07_bands.png`        | Ec, Ev, Ef (BGN applied for N > 10¹⁸ cm⁻³) |
| `08_ebic.png`         | EBIC + SEEBIC line scans |

## Physics implemented

* **Image modelling** — colour clustering + scale-bar auto-detection
  (`image_model.detect_scale_bar`, `image_model.segment_regions`).
* **Material table** — CSV with `Cal` placeholders that trigger
  doping-dependent evaluation of the work function and bandgap
  (`materials.load_material_table`, `materials.resolve_semiconductor`).
* **Arora mobility** — full temperature- and doping-dependent
  expression (`materials.arora_mobility`).  Diffusion length from
  `D = (kT/q) μ` and `L = √(Dτ)` (`materials.diffusion_length`).
* **Bandgap narrowing** — Slotboom-style; enabled above 10¹⁸ cm⁻³
  regardless of P- or N-type (`materials.bgn_delta_eg`,
  `materials.bandgap`).
* **Work function** — metals come from the table; doped Si uses
  `W = χ + (Ec − Ef)` or `W = χ + Eg − (Ev − Ef)` with BGN-corrected
  Eg (`materials.work_function_semi`).
* **SIMS → grid** — linear interpolation onto the sample grid; the
  substrate type + concentration is forced past the point where the
  measured profile has decayed past a (configurable) fraction of its
  peak (`sims.apply_sims_to_region`).
* **Contact classification** — from ΔW between metal and local
  semiconductor; returns `ohmic`/`schottky`, barrier height, and
  built-in voltage (`circuit.classify_contact`).
* **Electric field** — cumulative trapezoidal integration of
  Poisson's equation with the depletion approximation (peak doping on
  each side of the junction, exact charge balance)
  (`physics.electric_field`).
* **Depletion** — junctions found at every sign change of Nnet;
  widths from closed form + charge balance
  (`physics.depletion_region`).  Each junction is tagged **PN**, **NP**,
  or **HL** (high-low, same-type).
* **Kanaya–Okayama range** — `R = 0.0276·A·E^1.67 / (Z^0.889·ρ)` μm,
  implemented for arbitrary materials
  (`physics.kanaya_okayama_range_nm`).
* **Collection probability** — P = 1 inside the depletion region,
  `exp(−distance / L_local)` outside, with a minority-carrier L
  computed per-cell from the Arora mobility
  (`physics.collection_probability`).
* **EBIC scan** — at each beam position a Gaussian-bulb generation of
  total rate `Ibeam·V/EHP` is convolved with the collection probability
  along the slice (`physics.ebic_scan`).
* **SEEBIC scan** — secondary-electron current, modulated by the
  local surface field (`physics.seebic_scan`).

## Material table format (`Material_table_for_ebic_cal.csv`)

Transposed layout — one column per material; rows are

```
Material_name, Type (Semi/Metal/Insulator), Work_function, Electron_Affinity,
Bandgap, Relative_permittivity, Effective_e_mass, Effective_h_mass
```

Use the literal string `Cal` in a cell when the value depends on the
local doping (e.g. Eg or Work_function for doped Si) — the simulator
computes it through `materials.resolve_semiconductor`.

## Using your own image / SIMS data

```python
from ebic_sim import image_model, sims, materials, physics

model = image_model.build_model(
    "my_image.png",
    material_map={1: "SIMS", 2: "Pt", 3: "Al"},   # region_id -> material
    scalebar_um=10.0,
    thickness_nm=100.0,
)

p = sims.load_profile("my_Pprofile.csv", kind="P")
n = sims.load_profile("my_Nprofile.csv", kind="N")
```

then follow `run_example.py`.

## Notes

* The simulator does a 1-D Poisson analysis along a horizontal slice
  through the sample; this is the right approximation for planar
  cross-sections but if your geometry is genuinely 2-D (e.g.
  comb-like contacts) pick the slice row that cuts through the feature
  you care about.
* BGN is applied at any concentration above 10¹⁸ cm⁻³ for both dopant
  types.
* The K-O bulb is treated as a Gaussian with FWHM = R_KO; for ultra-thin
  TEM foils (Z ≪ R_KO) you should be aware that the actual lateral
  straggle is limited by the foil thickness.
