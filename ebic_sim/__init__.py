"""STEM-EBIC simulation package.

Modules
-------
materials   : Material table + Arora/BGN/work-function physics
image_model : Build 2D model from an image (regions + scale bar)
sims        : Load SIMS P/N profiles and map onto the grid
circuit     : Electrical contacts and contact-type classification
physics     : Device physics - integration-based E-field, depletion,
              Kanaya-Okayama generation volume, collection probability,
              EBIC and SEEBIC
visualization : All plotting helpers
"""

from . import materials, image_model, sims, circuit, physics, visualization

__all__ = [
    "materials",
    "image_model",
    "sims",
    "circuit",
    "physics",
    "visualization",
]
