"""Build a 2-D sample model from an image + separate scale-bar image.

Workflow
--------
1. Auto-detect the pixel length of the black bar inside ``scalebar.png``
   and convert it into a global nm/pixel using the known bar length
   (default 10 um).
2. Segment ``image.png`` by colour and expose a K-mean-clustered label
   map.  The user then tells us explicitly which physical region
   number to attach to each cluster, e.g. ``{cluster: region_id}``.
3. The user assigns a material to each region_id.  The special token
   ``"SIMS"`` signals that this region's doping will be supplied via
   SIMS profile applications (see ``sims.py``).

All coordinates exposed by :class:`SampleModel` are in **nanometres**
with ``(x, y) = (0, 0)`` at the top-left corner of the image to match
the way the image is shown on screen (positive y goes downwards).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Scale-bar detection (separate image)
# ---------------------------------------------------------------------------
def detect_scale_from_image(scalebar_path: str, bar_length_um: float = 10.0,
                             darkness: int = 80) -> float:
    """Return nm / pixel using the length of the dark bar in the image.

    The heuristic looks for the longest consecutive run of dark
    pixels in any row of the image.  That run is assumed to be the
    scale bar of length ``bar_length_um`` (default 10 um).
    """
    gray = np.asarray(Image.open(scalebar_path).convert("L"))
    mask = gray < darkness
    best = 0
    for row in mask:
        if not row.any():
            continue
        idx = np.flatnonzero(row)
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        best = max(best, max(len(r) for r in runs))
    if best < 5:
        raise ValueError("Could not find a scale bar in the image")
    return bar_length_um * 1000.0 / best         # nm per pixel


# ---------------------------------------------------------------------------
# Colour segmentation
# ---------------------------------------------------------------------------
def cluster_image(img: np.ndarray, n_clusters: int,
                   background_threshold: int = 235) -> np.ndarray:
    """Return an integer label image; 0 = background, 1..k = clusters."""
    h, w, _ = img.shape
    fg_mask = np.any(img < background_threshold, axis=2)
    labels = np.zeros((h, w), dtype=np.int32)
    if not fg_mask.any():
        return labels
    fg_pixels = img[fg_mask].reshape(-1, 3)
    if n_clusters <= 1:
        labels[fg_mask] = 1
        return labels
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0).fit(fg_pixels)
    labels[fg_mask] = km.labels_ + 1
    return labels


# ---------------------------------------------------------------------------
# Sample model
# ---------------------------------------------------------------------------
@dataclass
class SampleModel:
    image: np.ndarray                    # (H, W, 3) uint8
    cluster_mask: np.ndarray             # (H, W) int raw KMeans labels
    region_mask: np.ndarray              # (H, W) int user-assigned region ids
    nm_per_pixel: float
    thickness_nm: float
    # region_id  ->  material name or "SIMS"
    materials: dict[int, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def shape(self):
        return self.region_mask.shape

    @property
    def extent_nm(self):
        """Extent for matplotlib ``imshow``.  Positive y goes **down** so
        ``origin='upper'`` (the default) keeps the sample visually
        upright."""
        h, w = self.region_mask.shape
        return (0.0, w * self.nm_per_pixel,
                h * self.nm_per_pixel, 0.0)

    def xy_grids_nm(self):
        """Return ``(X_nm, Y_nm)`` 2-D grids of coordinates in nm."""
        h, w = self.region_mask.shape
        px = self.nm_per_pixel
        x = (np.arange(w) + 0.5) * px
        y = (np.arange(h) + 0.5) * px
        return np.meshgrid(x, y)

    def region_bbox_nm(self, rid: int):
        ys, xs = np.where(self.region_mask == rid)
        if xs.size == 0:
            return None
        p = self.nm_per_pixel
        return (xs.min() * p, xs.max() * p,
                ys.min() * p, ys.max() * p)

    def region_ids(self):
        return sorted(int(r) for r in np.unique(self.region_mask) if r != 0)

    def set_material(self, rid: int, name: str):
        self.materials[int(rid)] = name
        return self

    # ------------------------------------------------------------------
    # Region editing helpers
    # ------------------------------------------------------------------
    def assign_clusters_to_regions(self, mapping: dict[int, int]):
        """``mapping`` = ``{cluster_id: region_id}``.

        Re-labels :attr:`region_mask` so that every pixel belonging to
        cluster ``c`` in :attr:`cluster_mask` becomes region ``mapping[c]``.
        Cluster ids not listed end up in region 0 (background).
        """
        new_mask = np.zeros_like(self.cluster_mask)
        for cid, rid in mapping.items():
            new_mask[self.cluster_mask == cid] = int(rid)
        self.region_mask = new_mask
        return self

    # ---------------- manual region specification ----------------
    def clear_regions(self):
        self.region_mask = np.zeros_like(self.cluster_mask)
        self.materials = {}
        return self

    def add_region_bbox(self, rid: int, *,
                        x_nm: tuple[float, float] | None = None,
                        y_nm: tuple[float, float] | None = None):
        """Paint the rectangle ``x_nm x y_nm`` with ``rid`` (integer).

        Bounds default to the full image if an axis is not specified.
        """
        H, W = self.cluster_mask.shape
        px = self.nm_per_pixel
        x_lo, x_hi = x_nm if x_nm is not None else (0.0, W * px)
        y_lo, y_hi = y_nm if y_nm is not None else (0.0, H * px)
        c0 = max(0, int(np.floor(x_lo / px)))
        c1 = min(W, int(np.ceil (x_hi / px)))
        r0 = max(0, int(np.floor(y_lo / px)))
        r1 = min(H, int(np.ceil (y_hi / px)))
        self.region_mask[r0:r1, c0:c1] = int(rid)
        return self

    def add_region_from_color(self, rid: int,
                               rgb: tuple[int, int, int],
                               tolerance: int = 40):
        """Paint every pixel matching ``rgb`` (+/- tolerance) with ``rid``."""
        r, g, b = rgb
        dr = self.image[..., 0].astype(int) - r
        dg = self.image[..., 1].astype(int) - g
        db = self.image[..., 2].astype(int) - b
        mask = (np.abs(dr) <= tolerance) & \
                (np.abs(dg) <= tolerance) & \
                (np.abs(db) <= tolerance)
        self.region_mask[mask] = int(rid)
        return self


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def build_model(image_path: str, scalebar_path: str,
                n_clusters: int,
                thickness_nm: float = 100.0,
                bar_length_um: float = 10.0) -> SampleModel:
    """Load an image + scale bar and return a :class:`SampleModel`.

    The caller completes the model by calling
    ``assign_clusters_to_regions`` and ``set_material`` (or passing
    ``SampleModel(...).materials = {...}``).
    """
    img = np.asarray(Image.open(image_path).convert("RGB"))
    nm_per_pixel = detect_scale_from_image(scalebar_path, bar_length_um)
    cluster_mask = cluster_image(img, n_clusters=n_clusters)
    return SampleModel(
        image=img,
        cluster_mask=cluster_mask,
        region_mask=cluster_mask.copy(),   # default 1:1 mapping
        nm_per_pixel=float(nm_per_pixel),
        thickness_nm=float(thickness_nm),
    )
