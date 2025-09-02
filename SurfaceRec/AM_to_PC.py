#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AM(绿色通道) -> 3D点云 (.xyz / .ply)

规则：
- 用 G 通道做距离幅值： d = (G/255) * d_max
- 将像素(u,v)映射到平面坐标(U,V)∈[0,U_extent]×[0,V_extent]
- 三维点：p = o + U*u_hat + V*v_hat + sign * d * n_hat
  这里与现有代码对齐：默认 sign = -1（即 z = -(G*30)/256 的风格）

用法示例：
python am_to_pointcloud.py \
  --green /path/case001_C_green.png \
  --out-xyz PointCloud_generated/output_gd.xyz \
  --o 0,0,0 --u-axis 0,1,0 --v-axis 1,0,0 \
  --plane-size 150,150 --d-max 30 --stride 1
"""
import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2


def parse_vec3(s: str) -> np.ndarray:
    xs = [float(x) for x in s.split(",")]
    if len(xs) != 3:
        raise ValueError(f"Need 3 numbers, got: {s}")
    return np.array(xs, dtype=float)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length vector.")
    return v / n


def write_xyz(points: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def write_ply_xyzrgb(points: np.ndarray, colors: Optional[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if colors is None:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for x, y, z in points:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        else:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def am_green_to_pointcloud(
    green_img_path: Path,
    o: np.ndarray,            # 平面过点
    u_axis: np.ndarray,       # 平面U轴方向（3D）
    v_axis: np.ndarray,       # 平面V轴方向（3D）
    n_axis: Optional[np.ndarray],  # 平面法向（不传则用 u×v）
    plane_size_xy: Tuple[float, float],  # (U_extent, V_extent)
    d_max: float = 30.0,
    stride: int = 1,
    g_min: int = 1,           # 过滤背景：G <= g_min 认为无效
    sign: float = -1.0,       # 与你现有代码一致，取负号
    keep_rgb: bool = False,   # 是否把原像素颜色写入PLY
):
    img = cv2.imread(str(green_img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {green_img_path}")
    H, W, _ = img.shape

    # OpenCV BGR
    B = img[:, :, 0].astype(np.uint8)
    G = img[:, :, 1].astype(np.uint8)
    R = img[:, :, 2].astype(np.uint8)

    # 若你的AM背景是“纯白(255,255,255)”，也可用该判据过滤
    # mask_valid = ~((B == 255) & (G == 255) & (R == 255))
    # 这里按绿色阈值过滤：
    mask_valid = G > g_min

    u_hat = unit(u_axis)
    v_hat = unit(v_axis)
    n_hat = unit(np.cross(u_hat, v_hat)) if n_axis is None else unit(n_axis)

    U_extent, V_extent = plane_size_xy
    us = np.arange(0, W, stride, dtype=np.int32)
    vs = np.arange(0, H, stride, dtype=np.int32)

    points = []
    colors = [] if keep_rgb else None

    for v in vs:
        V = (v / max(1, H - 1)) * V_extent
        for u in us:
            if not mask_valid[v, u]:
                continue
            g = float(G[v, u])
            d = (g / 255.0) * d_max
            U = (u / max(1, W - 1)) * U_extent

            p_plane = o + U * u_hat + V * v_hat
            p_3d = p_plane + sign * d * n_hat  # 默认负号与旧逻辑一致

            points.append(p_3d)
            if colors is not None:
                b, g8, r = img[v, u]
                colors.append((int(r), int(g8), int(b)))

    if not points:
        return np.empty((0, 3), dtype=np.float32), (None if colors is None else np.empty((0, 3), dtype=np.uint8))
    pts_np = np.asarray(points, dtype=np.float32)
    cols_np = None if colors is None else np.asarray(colors, dtype=np.uint8)
    return pts_np, cols_np


def main():
    ap = argparse.ArgumentParser(description="Convert GREEN AM to 3D point cloud")
    ap.add_argument("--green", required=True, help="Green AM image path")
    ap.add_argument("--out-xyz", default=None, help="Output .xyz path")
    ap.add_argument("--out-ply", default=None, help="Output .ply path (optional)")

    ap.add_argument("--o",       default="0,0,0", help="Plane point o 'x,y,z'")
    ap.add_argument("--u-axis",  default="0,1,0", help="Plane U axis dir 'x,y,z' (默认世界Y)")
    ap.add_argument("--v-axis",  default="1,0,0", help="Plane V axis dir 'x,y,z'")
    ap.add_argument("--n-axis",  default=None,    help="Plane normal dir; omitted => u×v")

    ap.add_argument("--plane-size", default="150,150", help="Plane extent 'U,V'")
    ap.add_argument("--d-max", type=float, default=30.0, help="Max distance at G=255")
    ap.add_argument("--stride", type=int, default=1, help="Pixel subsampling")
    ap.add_argument("--g-min",  type=int, default=1, help="Ignore pixels with G<=g_min")
    ap.add_argument("--sign",   type=float, default=-1.0, help="Distance sign (+1 or -1); default -1 to match old z=-(G*30)/256")
    ap.add_argument("--keep-rgb", action="store_true", help="Write RGB into PLY")
    args = ap.parse_args()

    green = Path(args.green)
    o      = parse_vec3(args.o)
    u_axis = parse_vec3(args.u_axis)
    v_axis = parse_vec3(args.v_axis)
    n_axis = parse_vec3(args.n_axis) if args.n_axis is not None else None

    Uext, Vext = [float(x) for x in args.plane_size.split(",")]
    pts, cols = am_green_to_pointcloud(
        green_img_path=green,
        o=o, u_axis=u_axis, v_axis=v_axis, n_axis=n_axis,
        plane_size_xy=(Uext, Vext),
        d_max=args.d_max, stride=args.stride, g_min=args.g_min,
        sign=args.sign, keep_rgb=args.keep_rgb,
    )

    if args.out_xyz:
        write_xyz(pts, Path(args.out_xyz))
        print(f"✓ XYZ saved: {args.out_xyz} ({len(pts)} pts)")

    if args.out_ply:
        write_ply_xyzrgb(pts, cols if args.keep_rgb else None, Path(args.out_ply))
        print(f"✓ PLY saved: {args.out_ply} ({len(pts)} pts)")


if __name__ == "__main__":
    main()
