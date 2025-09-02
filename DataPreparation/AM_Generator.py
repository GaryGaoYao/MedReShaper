#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Gary @ UZ Leuven
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Iterable, Set

import numpy as np
from matplotlib.figure import Figure
from sklearn.neighbors import NearestNeighbors
import trimesh

def parse_xyz(s: Optional[str]) -> Optional[np.ndarray]:
    if not s:
        return None
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 3:
        raise ValueError(f"Invalid xyz string: {s}")
    return np.array(vals, dtype=float)


def rotate_vec_around_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    v_par = np.dot(v, axis) * axis
    v_perp = v - v_par
    v_perp_rot = v_perp * np.cos(angle_rad) + np.cross(axis, v_perp) * np.sin(angle_rad)
    return v_par + v_perp_rot
  
def define_transformation_matrix(x_axis: np.ndarray, y_axis: np.ndarray, origin: np.ndarray) -> np.ndarray:
    x_axis_n = x_axis / (np.linalg.norm(x_axis) + 1e-12)
    y_axis_n = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    z_axis = np.cross(x_axis_n, y_axis_n)
    T = np.column_stack([x_axis_n, y_axis_n, z_axis, origin])
    T = np.vstack([T, [0, 0, 0, 1]])
    return T


def project_to_2d(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """points_xyz: (N,3). T: 4x4. returns (N,2)"""
    pts_h = np.column_stack([points_xyz, np.ones(len(points_xyz))])
    pts_t = pts_h @ T
    return pts_t[:, :2]

def linear_normalize(x: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    return np.clip((x - fmin) / (fmax - fmin + 1e-12), 0.0, 1.0)


def psi_from_raw(x: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    phi = linear_normalize(x, fmin, fmax)
    # ψ_f = φ_f · sigmoid(φ_f)
    return phi * (1.0 / (1.0 + np.exp(-phi)))

def compute_local_descriptors(
    points_xyz: np.ndarray,
    d_to_plane: np.ndarray,
    plane_normal: np.ndarray,
    k: int = 50,
    sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    points_xyz: (N,3) raw 3D points (not projected)
    d_to_plane: (N,)
    plane_normal: (3,) unit normal
    return: (kappa, rho, theta) length-N arrays
    """
    N = points_xyz.shape[0]
    if N < 5:
        z = np.zeros(N)
        return z, z.copy(), z.copy()

    k_eff = min(k, N)
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(points_xyz)
    dist_knn, idx_knn = nbrs.kneighbors(points_xyz, return_distance=True)

    h_i = np.median(dist_knn[:, 1:], axis=1)       # median neighbor spacing
    r_i = np.maximum(sigma * h_i, 1e-6)

    kappa = np.zeros(N, dtype=np.float64)
    rho   = np.zeros(N, dtype=np.float64)
    theta = np.zeros(N, dtype=np.float64)

    for i in range(N):
        neigh_idx = idx_knn[i]
        Pi   = points_xyz[i]
        Pnb  = points_xyz[neigh_idx]
        Dj   = d_to_plane[neigh_idx]

        dij2 = np.sum((Pnb - Pi)**2, axis=1)
        w = np.exp(-dij2 / (2.0 * (r_i[i]**2) + 1e-12)) + 1e-12

        X = (Pnb - Pi)
        C = (X.T @ (w[:, None] * X)) / np.sum(w)

        evals, evecs = np.linalg.eigh(C)          # ascending
        order = np.argsort(evals)[::-1]           # λ1≥λ2≥λ3
        lam = evals[order]
        vec = evecs[:, order]

        lam_sum = np.sum(lam) + 1e-12
        kappa[i] = lam[-1] / lam_sum              # λ3/(λ1+λ2+λ3)

        dj_bar   = np.sum(w * Dj) / np.sum(w)
        rho[i]   = np.sqrt(np.sum(w * (Dj - dj_bar)**2) / np.sum(w) + 1e-18)

        n_i      = vec[:, -1]
        cosang   = np.clip(np.abs(np.dot(n_i, plane_normal)), 0.0, 1.0)
        theta[i] = np.arccos(cosang)

    return kappa, rho, theta

def generate_green_and_reds(
    stl_path: Path,
    out_prefix: Path,
    o: Optional[np.ndarray] = None,
    axis_p1: Optional[np.ndarray] = None,
    angles_deg: Tuple[int, ...] = (0, 10, -10, 30, -30, 45, -45),
    red_enable_idx: Optional[Set[int]] = None,
    n_points: int = 2_000_000,
    shield: float = 0.0,
    alpha: float = 0.15,
    d_max: float = 30.0,
    kappa_max: float = 0.2,
    rho_max: float = 5.0,
    theta_max: float = np.pi/2,
) -> None:
    """
    Create 1 green map (above, angle=0°) and up to 7 red maps (below) for angles in angles_deg.
    angles index mapping (1..7): [0, +10, -10, +30, -30, +45, -45]
      => red index=1 => 0°
         red index=3 => -10°
         red index=4 => +30°
         red index=7 => -45°
    Use --enable-red 1,3,4,7 to only export those four.

    out files:
      out_prefix_green.png                             (above, 0°)
      out_prefix_red_ang_0.png, _ang_p10.png, ...      (below, each enabled angle)
    """
    mesh = trimesh.load(str(stl_path))
    pts  = mesh.sample(n_points)  # (N,3)

    if o is None:
        o = mesh.centroid

    # Base plane normal n0 using o,B,C (as in your spec)
    point_A = o
    point_B = o + np.array([0, 10, 0], dtype=float)
    point_C = o + np.array([0, 50, 0], dtype=float)
    n0 = np.cross(point_B - point_A, point_C - point_A)
    n0 = n0 / (np.linalg.norm(n0) + 1e-12)

    # Rotation axis
    if axis_p1 is None:
        axis_p1 = o + np.array([0, 1, 0], dtype=float)   # fallback: world Y
    axis = axis_p1 - o
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    # Build 2D projection frame (fixed across angles to keep consistent image coords)
    origin = np.array([0.0, 0.0, 0.0])
    x_axis = np.array([0.0, 1.0, 0.0])                 # world Y -> image X
    y_axis = np.array([o[0], 0.0, o[2]])               # horizontal dir by (o.x, o.z)
    T = define_transformation_matrix(x_axis, y_axis, origin)
    image_width, image_height = 1000, 1000
    x_limit = image_width / 50
    y_limit = image_height / 50

    # ------- GREEN map (above, 0°) -------
    n_green = n0  # 0°
    v = pts - o[None, :]
    signed_d = v @ n_green
    proj_pts = pts - signed_d[:, None] * n_green[None, :]
    d_abs = np.abs(signed_d)
    above_mask = signed_d > shield

    above_proj = proj_pts[above_mask]
    above_d    = d_abs[above_mask]
    above_xy   = project_to_2d(above_proj, T)

    G = np.clip(above_d / d_max, 0.0, 1.0)
    colors_above = np.stack([np.zeros_like(G), G, np.zeros_like(G)], axis=1)

    fig_a = Figure(figsize=(image_width/100, image_height/100), dpi=100)
    ax_a = fig_a.add_subplot(111)
    ax_a.set_xlim(0, x_limit * 3)
    ax_a.set_ylim(0, y_limit * 3)
    ax_a.scatter(above_xy[:, 0], above_xy[:, 1], c=colors_above, s=5, marker='o')
    ax_a.set_aspect('equal', adjustable='box'); ax_a.autoscale(False); ax_a.axis('off')
    fig_a.savefig(str(out_prefix) + "_green.png", dpi=100)

    # ------- RED maps (below, each angle) -------
    if red_enable_idx is None:
        red_enable_idx = set(range(1, len(angles_deg) + 1))  # enable all if not given

    # angles index mapping
    # idx: 1  2    3     4     5     6     7
    # ang: 0 +10   -10   +30   -30   +45   -45
    for idx, ang in enumerate(angles_deg, start=1):
        if idx not in red_enable_idx:
            continue
        n = rotate_vec_around_axis(n0, axis, np.deg2rad(ang))

        v = pts - o[None, :]
        signed_d = v @ n
        proj_pts = pts - signed_d[:, None] * n[None, :]
        d_abs = np.abs(signed_d)
        below_mask = signed_d < -shield

        below_proj = proj_pts[below_mask]    # for plotting (projected)
        below_d    = d_abs[below_mask]       # |distance|
        below_xyz  = pts[below_mask]         # raw 3D for kNN

        # local descriptors
        kappa, rho, theta = compute_local_descriptors(
            points_xyz=below_xyz,
            d_to_plane=below_d,
            plane_normal=n,
            k=50, sigma=1.0
        )

        # gating + fusion
        psi_d     = psi_from_raw(below_d, 0.0, d_max)
        psi_k     = psi_from_raw(kappa,   0.0, kappa_max)
        psi_rho   = psi_from_raw(rho,     0.0, rho_max)
        psi_theta = psi_from_raw(theta,   0.0, theta_max)

        L = (psi_k + psi_rho + psi_theta) / 3.0
        R = psi_d + alpha * L * psi_d * (1.0 - psi_d)
        R = np.clip(R, 0.0, 1.0)

        below_xy = project_to_2d(below_proj, T)
        colors_below = np.stack([R, np.zeros_like(R), np.zeros_like(R)], axis=1)

        fig_b = Figure(figsize=(image_width/100, image_height/100), dpi=100)
        ax_b = fig_b.add_subplot(111)
        ax_b.set_xlim(0, x_limit * 3)
        ax_b.set_ylim(0, y_limit * 3)
        ax_b.scatter(below_xy[:, 0], below_xy[:, 1], c=colors_below, s=5, marker='o')
        ax_b.set_aspect('equal', adjustable='box'); ax_b.autoscale(False); ax_b.axis('off')

        tag = f"ang_{('p' if ang>0 else ('m' if ang<0 else '0'))}{abs(ang)}"
        fig_b.savefig(f"{out_prefix}_red_{tag}.png", dpi=100)


def parse_enable_set(s: Optional[str], total: int) -> Set[int]:
    """
    Parse "--enable-red" like "1,3,4,7" into {1,3,4,7}.
    If s is None => return {1..total}
    """
    if not s:
        return set(range(1, total + 1))
    ints = set()
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            v = int(token)
            if v < 1 or v > total:
                raise ValueError
            ints.add(v)
        except ValueError:
            raise ValueError(f"Invalid red index '{token}', must be in 1..{total}")
    return ints


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Generate 1 green AM + up to 7 red AMs")
    ap.add_argument("--stl", required=True, help="input STL (normalized & aligned)")
    ap.add_argument("--out", required=True, help="output prefix, e.g., /path/case001_C")

    # Optional anatomical points
    ap.add_argument("--D",  help="dacryon,  xyz format 'x,y,z'")
    ap.add_argument("--Or", help="orbitale, xyz format 'x,y,z'")
    ap.add_argument("--G",  help="gravity  xyz format 'x,y,z' (plane point o)")

    # Angles & switches
    ap.add_argument("--angles", default="0,10,-10,30,-30,45,-45",
                    help="comma list of angles in degrees for 7 red views")
    ap.add_argument("--enable-red", default="1,3,4,7",
                    help="which red indices to export, e.g., '1,3,4,7' (1..7)")

    # Params
    ap.add_argument("--n-points", type=int, default=2_000_000)
    ap.add_argument("--shield", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--d-max", type=float, default=30.0)
    ap.add_argument("--kappa-max", type=float, default=0.2)
    ap.add_argument("--rho-max", type=float, default=5.0)
    ap.add_argument("--theta-max", type=float, default=np.pi/2)

    args = ap.parse_args(argv)

    stl_path = Path(args.stl)
    out_prefix = Path(args.out)

    # parse angles & switches
    angles = tuple(int(x) for x in args.angles.split(","))
    if len(angles) != 7:
        raise ValueError("Expect exactly 7 angles (e.g., 0,10,-10,30,-30,45,-45)")
    red_enable = parse_enable_set(args.enable_red, total=7)

    # parse points
    D  = parse_xyz(args.D)
    Or = parse_xyz(args.Or)
    G  = parse_xyz(args.G)

    if (D is not None) and (Or is not None) and (G is not None):
        o = G
        axis_p1 = 0.5 * (D + Or)
    else:
        # fallback: use centroid as 'o', and world Y as axis
        o = None
        axis_p1 = None

    generate_green_and_reds(
        stl_path=stl_path,
        out_prefix=out_prefix,
        o=o,
        axis_p1=axis_p1,
        angles_deg=angles,
        red_enable_idx=red_enable,
        n_points=args.n_points,
        shield=args.shield,
        alpha=args.alpha,
        d_max=args.d_max,
        kappa_max=args.kappa_max,
        rho_max=args.rho_max,
        theta_max=args.theta_max,
    )


if __name__ == "__main__":
    main()
