"""OpenCV-based image modelling.

The input image contains
  * a scale bar in the **bottom-left** corner (a short horizontal
    black line, with its length label above it), and
  * one or more coloured shapes that represent the sample regions.

Pipeline
--------
1. ``_flatten_alpha``  - composite any transparent image over white.
2. ``detect_scalebar`` - in the bottom-left quadrant, find the
   longest straight black horizontal segment and return its pixel
   length.  That pixel length equals ``bar_length_um``.
3. ``segment_regions``  - K-means the non-background pixels in RGB
   space, then find contours for every cluster with
   ``cv2.findContours``.  Each contour becomes a region whose id the
   user controls through :meth:`SampleModel.assign_clusters_to_regions`
   or direct painting via :meth:`SampleModel.add_region_bbox` /
   :meth:`SampleModel.add_region_from_color`.
4. ``build_model``     - glue everything together.

Only ``opencv-python`` (+ numpy + Pillow + scikit-learn for KMeans)
is needed.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def _flatten_alpha(path: str) -> np.ndarray:
    """Read an image and return a plain RGB ``uint8`` array.  Transparent
    areas are composited onto a white background so any "empty" region
    is bright, not black."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im.convert("RGBA"))
        im = bg.convert("RGB")
    else:
        im = im.convert("RGB")
    return np.asarray(im)


# ---------------------------------------------------------------------------
# Scale-bar detection (bottom-left quadrant)
# ---------------------------------------------------------------------------
def detect_scalebar(rgb: np.ndarray,
                     bar_length_um: float = 10.0,
                     quadrant: float = 0.5,
                     darkness: int = 60,
                     min_length_px: int = 15) -> dict:
    """Find a horizontal scale bar in the bottom-left corner.

    Parameters
    ----------
    rgb : ndarray, H x W x 3
    bar_length_um : float
        Physical length represented by the bar.
    quadrant : float
        Fractional size of the bottom-left ROI; ``0.5`` == bottom-left
        quarter.
    darkness : int
        Pixels with grayscale value below this threshold are considered
        "black".
    min_length_px : int
        Discard any run shorter than this (noise / text strokes).

    Returns
    -------
    dict with keys ``nm_per_pixel``, ``length_px``, ``y``, ``x0``, ``x1``.
    """
    H, W, _ = rgb.shape
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # ROI = bottom-left quadrant
    y0 = int(H * (1.0 - quadrant))
    x1 = int(W * quadrant)
    roi = gray[y0:, :x1]

    # Binary mask of dark pixels.  Open with a horizontal kernel to
    # keep only horizontal strokes (the bar) and kill text shapes.
    mask = (roi < darkness).astype(np.uint8)
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hker)

    # Connected components on the horizontal-only mask.  We pick the
    # component with the widest bounding box - that's the bar.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(horiz, 8)
    best = None
    for lbl in range(1, num):
        x, y, w, h, area = stats[lbl]
        if w < min_length_px:
            continue
        # strongly prefer wide, thin shapes (bars, not blobs)
        if h > 0 and w / h < 5:
            continue
        if best is None or w > best[2]:
            best = (x, y, w, h)
    if best is None:
        raise ValueError("No scale bar found in the bottom-left quadrant")
    bx, by, bw, bh = best
    # translate back to full-image coordinates
    return dict(
        nm_per_pixel = bar_length_um * 1000.0 / bw,
        length_px    = int(bw),
        y            = int(y0 + by + bh // 2),
        x0           = int(bx),
        x1           = int(bx + bw),
        roi_origin   = (y0, 0),
    )


# ---------------------------------------------------------------------------
# Shape segmentation
# ---------------------------------------------------------------------------
def _background_mask(rgb: np.ndarray, white_threshold: int = 235):
    """True where the pixel is "white / empty" (not part of a shape)."""
    return np.all(rgb > white_threshold, axis=2)


def _strip_scalebar_region(rgb: np.ndarray, scalebar: dict,
                            pad: int = 10) -> np.ndarray:
    """Blank-out the scale-bar + label area so it isn't picked up as a
    shape.  We zero out the full width up to x1 + pad, from the scale
    bar's y - a few rows up to the image bottom."""
    out = rgb.copy()
    H, W, _ = rgb.shape
    y_top = max(0, scalebar["y"] - int(H * 0.15))        # label sits above
    x_right = scalebar["x1"] + pad
    out[y_top:, :x_right] = 255
    return out


def segment_shapes(rgb: np.ndarray, n_clusters: int,
                    scalebar: dict | None = None,
                    background_threshold: int = 235,
                    min_area_px: int = 100) -> dict:
    """HSV-Hue K-means colour-segment and contour extraction.

    Clustering is done in the HSV space (using H and S only) so
    visually distinct hues - for example yellow vs orange - separate
    cleanly, which is a well-known weak point of RGB K-means.

    Returns
    -------
    dict with keys:
        ``label_mask`` : (H, W) int32 with cluster ids 1..k (0 = background)
        ``cluster_colors`` : ndarray (k, 3) of mean RGB per cluster
        ``contours``   : list of (cluster_id, contour) tuples from OpenCV
    """
    img = rgb if scalebar is None else _strip_scalebar_region(rgb, scalebar)
    H, W, _ = img.shape
    bg = _background_mask(img, background_threshold)
    fg = ~bg

    label_mask = np.zeros((H, W), dtype=np.int32)
    if not fg.any():
        return dict(label_mask=label_mask, cluster_colors=np.zeros((0, 3)),
                     contours=[])

    fg_pix = img[fg].reshape(-1, 3).astype(np.uint8)
    n_clusters = max(1, int(n_clusters))

    if n_clusters == 1 or len(fg_pix) < n_clusters:
        label_mask[fg] = 1
        centers = fg_pix.mean(axis=0, keepdims=True).astype(np.float32)
    else:
        # Cluster on Hue (scaled) + Saturation so similar hues like
        # yellow vs orange (both high-R in RGB) are separated properly.
        hsv = cv2.cvtColor(fg_pix.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV
                            ).reshape(-1, 3).astype(np.float32)
        feats = np.stack([hsv[:, 0] * 2.0,        # 0..180 -> 0..360 (cyclic)
                           hsv[:, 1] * 0.5], axis=1)
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0).fit(feats)
        label_mask[fg] = km.labels_ + 1
        # report cluster colors back in plain RGB for convenience
        centers = np.stack([fg_pix[km.labels_ == k].mean(axis=0)
                             for k in range(n_clusters)]).astype(np.float32)

    # Contours per cluster
    contours = []
    for cid in range(1, n_clusters + 1):
        cluster_mask = (label_mask == cid).astype(np.uint8)
        # cleanup
        cluster_mask = cv2.morphologyEx(
            cluster_mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        cnts, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < min_area_px:
                continue
            contours.append((cid, c))
    return dict(label_mask=label_mask,
                 cluster_colors=np.asarray(centers),
                 contours=contours)


# ---------------------------------------------------------------------------
# SampleModel
# ---------------------------------------------------------------------------
@dataclass
class SampleModel:
    image: np.ndarray                    # RGB uint8
    cluster_mask: np.ndarray             # (H, W) int (raw KMeans labels)
    region_mask: np.ndarray              # (H, W) int (user-assigned)
    nm_per_pixel: float
    thickness_nm: float
    scalebar: dict = field(default_factory=dict)
    contours: list = field(default_factory=list)
    cluster_colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    materials: dict[int, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def shape(self):
        return self.region_mask.shape

    @property
    def extent_nm(self):
        h, w = self.region_mask.shape
        return (0.0, w * self.nm_per_pixel,
                h * self.nm_per_pixel, 0.0)

    def xy_grids_nm(self):
        h, w = self.region_mask.shape
        p = self.nm_per_pixel
        x = (np.arange(w) + 0.5) * p
        y = (np.arange(h) + 0.5) * p
        return np.meshgrid(x, y)

    def region_ids(self):
        return sorted(int(r) for r in np.unique(self.region_mask) if r != 0)

    def region_bbox_nm(self, rid):
        ys, xs = np.where(self.region_mask == rid)
        if xs.size == 0:
            return None
        p = self.nm_per_pixel
        return (xs.min() * p, xs.max() * p,
                ys.min() * p, ys.max() * p)

    # ------------------------------------------------------------------
    # Material assignment
    # ------------------------------------------------------------------
    def set_material(self, rid: int, name: str):
        self.materials[int(rid)] = name
        return self

    # ------------------------------------------------------------------
    # Region id assignment
    # ------------------------------------------------------------------
    def clear_regions(self):
        self.region_mask = np.zeros_like(self.cluster_mask)
        self.materials = {}
        return self

    def assign_clusters_to_regions(self, mapping: dict[int, int],
                                     keep_only_largest: bool = True):
        """Re-label ``region_mask`` from ``{cluster_id: region_id}``.

        When ``keep_only_largest`` is True (default) only the biggest
        connected component per cluster survives, which gets rid of
        stray dust pixels from colour-segmentation noise.
        """
        new_mask = np.zeros_like(self.cluster_mask)
        for cid, rid in mapping.items():
            src = (self.cluster_mask == cid).astype(np.uint8)
            if keep_only_largest and src.any():
                num, lbls, stats, _ = cv2.connectedComponentsWithStats(src, 8)
                if num > 1:
                    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    src = (lbls == largest).astype(np.uint8)
            new_mask[src > 0] = int(rid)
        self.region_mask = new_mask
        return self

    def largest_contour_per_cluster(self):
        """Return ``{cluster_id: (contour, area, cx, cy)}`` with the
        single biggest contour for each cluster."""
        best: dict[int, tuple] = {}
        for cid, cnt in self.contours:
            area = cv2.contourArea(cnt)
            if cid in best and best[cid][1] >= area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            best[cid] = (cnt, area,
                          M["m10"] / M["m00"], M["m01"] / M["m00"])
        return best

    def auto_number_regions_by(self, axis: str = "y"):
        """Assign region ids 1..k by sorting the largest contour of
        each cluster along ``axis`` (``'y'`` top-down, ``'x'`` left-right).
        """
        best = self.largest_contour_per_cluster()
        if not best:
            return self
        key = 3 if axis == "y" else 2        # cy at index 3, cx at index 2
        order = sorted(best.keys(), key=lambda c: best[c][key])
        self.assign_clusters_to_regions({cid: i + 1
                                           for i, cid in enumerate(order)})
        return self

    def add_region_bbox(self, rid: int, *,
                         x_nm: tuple[float, float] | None = None,
                         y_nm: tuple[float, float] | None = None):
        H, W = self.shape
        p = self.nm_per_pixel
        x_lo, x_hi = x_nm if x_nm is not None else (0.0, W * p)
        y_lo, y_hi = y_nm if y_nm is not None else (0.0, H * p)
        c0 = max(0, int(np.floor(x_lo / p)))
        c1 = min(W, int(np.ceil (x_hi / p)))
        r0 = max(0, int(np.floor(y_lo / p)))
        r1 = min(H, int(np.ceil (y_hi / p)))
        self.region_mask[r0:r1, c0:c1] = int(rid)
        return self

    def add_region_from_color(self, rid: int,
                               rgb: tuple[int, int, int],
                               tolerance: int = 40):
        r, g, b = rgb
        dr = self.image[..., 0].astype(int) - r
        dg = self.image[..., 1].astype(int) - g
        db = self.image[..., 2].astype(int) - b
        mask = ((np.abs(dr) <= tolerance)
                & (np.abs(dg) <= tolerance)
                & (np.abs(db) <= tolerance))
        # also mask out the scalebar label region
        if self.scalebar:
            H, _W, _ = self.image.shape
            y_top = max(0, self.scalebar["y"] - int(H * 0.15))
            mask[y_top:, :self.scalebar["x1"] + 10] = False
        self.region_mask[mask] = int(rid)
        return self


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------
def build_model(image_path: str,
                n_clusters: int,
                *,
                bar_length_um: float = 10.0,
                thickness_nm: float = 100.0,
                auto_number_by: str | None = "y",
                min_area_px: int = 200,
                nm_per_pixel: float | None = None) -> SampleModel:
    """End-to-end model builder.

    Parameters
    ----------
    image_path : str
        Image containing both the sample shapes and a scale bar in the
        bottom-left corner.
    n_clusters : int
        Number of distinct colour clusters in the sample.
    bar_length_um : float
        Physical length represented by the scale bar.
    auto_number_by : 'x' | 'y' | None
        If given, the detected clusters are numbered along that axis.
        Pass ``None`` and call ``model.assign_clusters_to_regions``
        manually if you need full control.
    nm_per_pixel : float
        Override auto-detection.
    """
    rgb = _flatten_alpha(image_path)
    if nm_per_pixel is None:
        sb = detect_scalebar(rgb, bar_length_um=bar_length_um)
        nm_per_pixel = sb["nm_per_pixel"]
    else:
        sb = {}
    seg = segment_shapes(rgb, n_clusters=n_clusters,
                          scalebar=sb or None,
                          min_area_px=min_area_px)
    model = SampleModel(
        image=rgb,
        cluster_mask=seg["label_mask"],
        region_mask=seg["label_mask"].copy(),
        nm_per_pixel=float(nm_per_pixel),
        thickness_nm=float(thickness_nm),
        scalebar=sb,
        contours=seg["contours"],
        cluster_colors=seg["cluster_colors"],
    )
    if auto_number_by is not None:
        model.auto_number_regions_by(axis=auto_number_by)
    return model
