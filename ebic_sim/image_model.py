"""Build a 2-D TEM-sample model from an image.

Responsibilities
----------------
* Read the image and optionally auto-detect the scale bar to obtain a
  pixel->nm conversion.  When detection fails the caller can pass the
  scale explicitly.
* Segment the sample into labelled regions via colour clustering.  The
  background (white/near-white) is dropped.
* Let the user map region labels to material names.  The special token
  ``"SIMS"`` means "this region is semiconductor whose doping comes from
  the SIMS file(s) supplied later."
* Produce a dense grid with thickness Z (default 100 nm) and a region
  index for every cell.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


@dataclass
class SampleModel:
    """Dense 2-D sample description produced by :func:`build_model`."""
    image: np.ndarray                        # (H, W, 3) uint8
    region_mask: np.ndarray                  # (H, W) int - 0 = background
    region_ids: list[int] = field(default_factory=list)
    nm_per_pixel: float = 1.0
    thickness_nm: float = 100.0
    # map region_id -> material name (or "SIMS")
    material_map: dict[int, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------
    @property
    def extent_nm(self):
        h, w = self.region_mask.shape
        return (0.0, w * self.nm_per_pixel, 0.0, h * self.nm_per_pixel)

    def region_bbox_nm(self, rid: int):
        ys, xs = np.where(self.region_mask == rid)
        if xs.size == 0:
            return None
        x0, x1 = xs.min() * self.nm_per_pixel, (xs.max() + 1) * self.nm_per_pixel
        y0, y1 = ys.min() * self.nm_per_pixel, (ys.max() + 1) * self.nm_per_pixel
        return x0, x1, y0, y1

    def region_width_nm(self, rid: int):
        bb = self.region_bbox_nm(rid)
        return None if bb is None else bb[1] - bb[0]


# ---------------------------------------------------------------------------
# Scale-bar detection
# ---------------------------------------------------------------------------
def detect_scale_bar(img: np.ndarray, known_um: float = 10.0,
                      darkness: int = 80) -> float | None:
    """Estimate nm / pixel by finding the longest dark horizontal run.

    The simple heuristic: threshold near-black pixels, then find the row
    containing the longest consecutive dark run.  That run is assumed to
    be the scale bar of length ``known_um``.
    """
    gray = np.asarray(Image.fromarray(img).convert("L"))
    mask = gray < darkness
    best_row_run = 0
    for row in mask:
        # longest run of True in this row
        if not row.any():
            continue
        idx = np.flatnonzero(row)
        gaps = np.diff(idx)
        runs = np.split(idx, np.where(gaps > 1)[0] + 1)
        longest = max(len(r) for r in runs)
        if longest > best_row_run:
            best_row_run = longest
    if best_row_run < 10:
        return None
    return known_um * 1000.0 / best_row_run        # nm per pixel


# ---------------------------------------------------------------------------
# Region segmentation by colour
# ---------------------------------------------------------------------------
def segment_regions(img: np.ndarray, n_regions: int = 1,
                    background_threshold: int = 235) -> np.ndarray:
    """Return a mask where 0 = background and 1..k = detected regions.

    ``n_regions`` is the number of non-background colour clusters to
    keep.  For the simple demo image it is just 1.
    """
    h, w, _ = img.shape
    fg_mask = np.any(img < background_threshold, axis=2)
    mask = np.zeros((h, w), dtype=np.int32)
    fg_pixels = img[fg_mask].reshape(-1, 3)
    if fg_pixels.size == 0:
        return mask

    k = max(1, n_regions)
    if k == 1:
        mask[fg_mask] = 1
        return mask

    km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(fg_pixels)
    labels = km.labels_ + 1        # so background stays at 0
    mask[fg_mask] = labels
    return mask


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def build_model(image_path: str,
                material_map: dict[int, str],
                nm_per_pixel: float | None = None,
                scalebar_um: float = 10.0,
                thickness_nm: float = 100.0,
                n_regions: int | None = None) -> SampleModel:
    """Load an image and produce a :class:`SampleModel`.

    Parameters
    ----------
    material_map : dict
        ``{region_id: material_name}``.  ``material_name`` can be any key
        from the material table, or the literal string ``"SIMS"``.
    nm_per_pixel : float, optional
        Override the auto-detected scale.
    """
    img = np.asarray(Image.open(image_path).convert("RGB"))
    if nm_per_pixel is None:
        nm_per_pixel = detect_scale_bar(img, known_um=scalebar_um) or 1.0

    n = n_regions if n_regions is not None else max(material_map.keys())
    mask = segment_regions(img, n_regions=n)
    rids = [i for i in np.unique(mask) if i != 0]

    return SampleModel(
        image=img,
        region_mask=mask,
        region_ids=rids,
        nm_per_pixel=float(nm_per_pixel),
        thickness_nm=float(thickness_nm),
        material_map=dict(material_map),
    )
