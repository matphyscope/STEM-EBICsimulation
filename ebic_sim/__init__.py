"""STEM-EBIC simulation package.

Public modules
--------------
materials     - CSV loader + Arora / BGN / work-function physics
image_model   - Build a 2-D sample model from an image + scale bar
sims          - SIMS profile loader + axis-based placement
beam          - User-defined electron-beam condition (keV, n electrons)
circuit       - Electrical contacts, ohmic/Schottky classification
physics       - Depletion, E-field, collection probability,
                2-D EBIC / SEEBIC scans
visualization - Plotting + numerical dump helpers
"""

from . import (materials, image_model, sims, beam, circuit, physics,
                visualization)

__all__ = ["materials", "image_model", "sims", "beam", "circuit",
            "physics", "visualization"]
