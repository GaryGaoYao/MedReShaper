#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explosive Sliding (ES) module:
- Rotate the anatomical slicing plane around a cone axis with stride α and range β
- For each rotated plane: make P-AM (green), run ASM-Net to reconstruct surface,
  back-project AM to 3D point cloud, and select the best via Symmetric Chamfer Distance
  computed on a 2× dilated boundary band around the defect.

Dependencies: numpy, opencv-python, scikit-learn, trimesh, matplotlib
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple, List

import cv2
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors
from matplotlib.figure import Figure


# ========== Geometry helpers ==========
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length vector.")
    return v / n


def rotate_vec_around_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = unit(axis)
    v_par = np.dot(v, axis) * axis
    v_perp = v - v_par
    return v_par + v_perp * math.cos(angle_rad) + np.cross(axis, v_perp) * math.sin(angle_rad)


def define_T(x_axis: np.ndarray, y_axis: np.ndarray, origin: np.ndarray) -> np.ndarray:
    x_n = unit(x_axis); y_n = unit(y_axis); z_n = unit(np.cross(x_n, y_n))
    T = np.column_stack([x_n, y_n, z_n, origin])
    T = np.vstack([T, [0, 0, 0, 1]])
    return T


def project_to_2d(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts_h = np.column_stack([points_xyz, np.ones(len(points_xyz))])
    pts_t = pts_h @ T
    return pts_t[:, :2]


# ========== AM generation (green only) ==========
def make_green_pam(
    pts: np.ndarray,                   # (N,3) sampled from mesh
    o: np.ndarray,                     # plane point
    n: np.ndarray,                     # plane normal (unit)
    image_size: Tuple[int, int] = (1000, 1000),
    plane_span_xy: Tuple[float, float] = (150.0, 150.0),
    d_max: float = 30.0,
    out_png: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      green_img [H,W,3] (uint8, only G used), T (4x4), (o, n)
    """
    # signed distance & projection
    v = pts - o[None, :]
    signed_d = v @ n
    d_abs = np.abs(signed_d)
    proj_pts = pts - signed_d[:, None] * n[None, :]

    # 2D frame: X <- world Y, Y <- horizontal dir through (o.x,o.z)
    origin = np.array([0.0, 0.0, 0.0])
    x_axis = np.array([0.0, 1.0, 0.0])           # world Y
    y_axis = np.array([o[0], 0.0, o[2]])         # horizontal through (o.x, o.z)
    T = define_T(x_axis, y_axis, origin)

    xy = project_to_2d(proj_pts, T)              # (N,2) in plane coords
    W, H = image_size
    Uspan, Vspan = plane_span_xy

    # map plane XY to image pixels
    u = np.clip((xy[:, 0] / Uspan) * (W - 1), 0, W - 1).astype(np.int32)
    v_pix = np.clip((xy[:, 1] / Vspan) * (H - 1), 0, H - 1).astype(np.int32)

    # green = distance magnitude normalized to [0,255]
    G = np.clip((d_abs / max(1e-6, d_max)) * 255.0, 0, 255).astype(np.uint8)
    img = np.full((H, W, 3), 255, dtype=np.uint8)  # white background
    img[v_pix, u, 1] = np.maximum(img[v_pix, u, 1].astype(np.int32), G.astype(np.int32)).astype(np.uint8)
    img[v_pix, u, 0] = 0  # optional: suppress B/R to keep "valid" pixels different from white
    img[v_pix, u, 2] = 0

    if out_png:
        # draw as a dense scatter using matplotlib to avoid alias holes (optional)
        fig = Figure(figsize=(W/100, H/100), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.scatter(u, v_pix, c=G/255.0, s=1, marker='o')
        ax.invert_yaxis(); ax.axis('off')
        fig.savefig(out_png, dpi=100)

    return img, T, (o, n)


# ========== AM -> 3D point cloud (green only) ==========
def green_am_to_points(
    green_img: np.ndarray,          # [H,W,3] BGR (OpenCV convention)
    o: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    n_axis: Optional[np.ndarray],
    plane_span_xy: Tuple[float, float],
    d_max: float,
    stride: int = 1,
    sign: float = +1.0,             # along +n by default
    g_min: int = 1,                 # ignore near-background
) -> np.ndarray:
    H, W, _ = green_img.shape
    G = green_img[:, :, 1].astype(np.float32)
    u_hat = unit(u_axis)
    v_hat = unit(v_axis)
    n_hat = unit(np.cross(u_hat, v_hat)) if n_axis is None else unit(n_axis)
    Uspan, Vspan = plane_span_xy

    pts = []
    for y in range(0, H, stride):
        Vv = (y / max(1, H - 1)) * Vspan
        for x in range(0, W, stride):
            g = G[y, x]
            if g <= g_min:
                continue
            d = (g / 255.0) * d_max
            Uu = (x / max(1, W - 1)) * Uspan
            p_plane = o + Uu * u_hat + Vv * v_hat
            p_3d = p_plane + sign * d * n_hat
            pts.append(p_3d)
    if not pts:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


# ========== Defect boundary band (2× dilated) ==========
def make_2x_dilated_boundary(mask_defect: np.ndarray, dilate_kernel: int = 3) -> np.ndarray:
    """
    mask_defect: HxW uint8/bool, 1 where defect (hole/invalid) is.
    Return a boolean mask for a 2×-dilated boundary band around the defect.
    """
    mask = (mask_defect.astype(np.uint8) > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    dil1 = cv2.dilate(mask, kernel, iterations=1)
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)  # ring of 1px around defect
    band = cv2.dilate(edge, kernel, iterations=2)              # 2× dilated boundary
    return band.astype(bool)


def sample_points_on_band(
    green_img: np.ndarray,
    band_mask: np.ndarray,
    o: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray, n_axis: Optional[np.ndarray],
    plane_span_xy: Tuple[float, float], d_max: float, stride: int = 1, sign: float = +1.0
) -> np.ndarray:
    H, W = band_mask.shape
    G = green_img[:, :, 1].astype(np.float32)
    u_hat = unit(u_axis); v_hat = unit(v_axis)
    n_hat = unit(np.cross(u_hat, v_hat)) if n_axis is None else unit(n_axis)
    Uspan, Vspan = plane_span_xy
    pts = []
    for y in range(0, H, stride):
        Vv = (y / max(1, H - 1)) * Vspan
        row = band_mask[y]
        xs = np.where(row)[0]
        for x in xs[::stride]:
            d = (G[y, x] / 255.0) * d_max
            Uu = (x / max(1, W - 1)) * Uspan
            p_plane = o + Uu * u_hat + Vv * v_hat
            pts.append(p_plane + sign * d * n_hat)
    if not pts:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


# ========== Symmetric Chamfer Distance ==========
def chamfer_symmetric(P: np.ndarray, Q: np.ndarray) -> float:
    """
    P: (N,3), Q: (M,3)
    D_cham(P,Q) = (1/N) sum_i min_j ||p_i - q_j||^2 + (1/M) sum_j min_i ||q_j - p_i||^2
    """
    if len(P) == 0 or len(Q) == 0:
        return float('inf')
    nn_p = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(Q)
    d_p, _ = nn_p.kneighbors(P, return_distance=True)
    nn_q = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(P)
    d_q, _ = nn_q.kneighbors(Q, return_distance=True)
    return float((d_p**2).mean() + (d_q**2).mean())


# ========== Core ES pipeline ==========
@dataclass
class ESConfig:
    alpha_deg: float                  # stride (deg)
    beta_deg: float                   # range (max |angle|) (deg)
    image_size: Tuple[int, int] = (1000, 1000)
    plane_span_xy: Tuple[float, float] = (150.0, 150.0)
    d_max: float = 30.0
    stride_backproj: int = 1          # pixel step for back-projection
    sign_backproj: float = +1.0       # along +n
    g_min: int = 1


class ExplosiveSliding:
    def __init__(self, stl_path: str, sample_points: int = 2_000_000):
        self.mesh = trimesh.load(stl_path)
        self.pts = self.mesh.sample(sample_points)  # use same sampling for all variants

    def _angles(self, alpha_deg: float, beta_deg: float) -> List[float]:
        # angles from -β to +β with step α, inclusive of 0
        npos = int(round(beta_deg / max(alpha_deg, 1e-6)))
        angles = [k * alpha_deg for k in range(-npos, npos + 1)]
        return angles

    def run(
        self,
        o: np.ndarray,                  # plane point (e.g., gravity point)
        axis_p1: np.ndarray,           # other point on axis (e.g., mid(Dacryon, Orbitale))
        es_cfg: ESConfig,
        asm_infer_fn: Callable[[np.ndarray], np.ndarray],
        # above callback: input green AM image (H,W,3 BGR uint8) -> output reconstructed green AM image (H,W,3 or just G)
        defect_mask_img: Optional[np.ndarray] = None,  # HxW uint8, 1 where defect
    ) -> Tuple[np.ndarray, Tuple[float, float], List[Tuple[float, float, float]]]:
        """
        Return:
          best_recon_green_img, (alpha*, beta*==angle*), candidates_metrics list of (angle_deg, SCD, #points)
        """
        o = np.asarray(o, dtype=float)
        axis = unit(np.asarray(axis_p1, dtype=float) - o)
        # base normal n0 from o,B,C as in your spec
        point_B = o + np.array([0.0, 10.0, 0.0])
        point_C = o + np.array([0.0, 50.0, 0.0])
        n0 = unit(np.cross(point_B - o, point_C - o))

        # fixed 2D frame for all variants (consistent coordinates)
        x_axis = np.array([0.0, 1.0, 0.0])    # world Y
        y_axis = np.array([o[0], 0.0, o[2]])  # horizontal via (o.x, o.z)

        angles = self._angles(es_cfg.alpha_deg, es_cfg.beta_deg)

        # Prepare P (defective neighborhood) on 2× dilated boundary band
        # If defect_mask_img is None: try to auto-make from "holes" => here we require user to pass it for robustness
        if defect_mask_img is None:
            raise ValueError("Please provide defect_mask_img (HxW uint8).")

        band = make_2x_dilated_boundary(defect_mask_img)
        # We also need a "reference" green (at 0°) just to back-project P (magnitude source)
        ref_green, T0, _ = make_green_pam(self.pts, o, n0, es_cfg.image_size, es_cfg.plane_span_xy, es_cfg.d_max)
        P = sample_points_on_band(
            green_img=ref_green, band_mask=band,
            o=o, u_axis=x_axis, v_axis=y_axis, n_axis=None,
            plane_span_xy=es_cfg.plane_span_xy, d_max=es_cfg.d_max,
            stride=es_cfg.stride_backproj, sign=es_cfg.sign_backproj
        )

        # Iterate candidates
        best_scd = float('inf')
        best_img = None
        best_angle = 0.0
        metrics = []

        for ang in angles:
            n = rotate_vec_around_axis(n0, axis, math.radians(ang))
            # 1) P-AM (green) for this angle
            green_img, T, _ = make_green_pam(
                pts=self.pts, o=o, n=n,
                image_size=es_cfg.image_size,
                plane_span_xy=es_cfg.plane_span_xy,
                d_max=es_cfg.d_max,
                out_png=None
            )

            # 2) ASM-Net inference (user-provided)
            recon_green = asm_infer_fn(green_img)  # expect same HxW, green distance encoded in G-channel (0..255)

            # 3) AM -> 3D for candidate (Q_{α,β})
            Q = green_am_to_points(
                green_img=recon_green,
                o=o, u_axis=x_axis, v_axis=y_axis, n_axis=None,
                plane_span_xy=es_cfg.plane_span_xy,
                d_max=es_cfg.d_max,
                stride=es_cfg.stride_backproj,
                sign=es_cfg.sign_backproj,
                g_min=es_cfg.g_min
            )

            # 4) Chamfer on boundary band
            scd = chamfer_symmetric(P, Q)
            metrics.append((ang, scd, float(len(Q))))
            if scd < best_scd:
                best_scd = scd
                best_img = recon_green
                best_angle = ang

        return best_img, (es_cfg.alpha_deg, best_angle), metrics

