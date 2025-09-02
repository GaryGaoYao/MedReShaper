from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import cv2


def parse_vec3(s: str) -> np.ndarray:
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 3:
        raise ValueError(f"Need 3 numbers, got: {s}")
    return np.array(vals, dtype=float)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length vector for axis/normal.")
    return v / n


def write_ply_xyzrgb(path: Path, pts: np.ndarray, cols: np.ndarray | None = None):
    with open(path, "w", encoding="utf-8") as f:
        if cols is not None:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(pts, cols):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for (x, y, z) in pts:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def reconstruct_from_green(
    green_img_path: Path,
    out_path: Path,
    o: np.ndarray,            # plane point
    u_axis: np.ndarray,       # plane U axis (3D)
    v_axis: np.ndarray,       # plane V axis (3D)
    n_axis: np.ndarray | None,# plane normal; if None use u×v
    plane_size_xy: Tuple[float, float],  # (U_extent, V_extent)
    d_max: float = 30.0,
    stride: int = 1,
    g_min: int = 1,           # ignore pixels with G <= g_min (background)
    keep_rgb: bool = True,    # store source RGB in PLY
) -> int:
    img = cv2.imread(str(green_img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {green_img_path}")
    h, w, _ = img.shape

    # OpenCV: BGR
    G = img[:, :, 1].astype(np.float32)

    u_hat = unit(u_axis)
    v_hat = unit(v_axis)
    n_hat = unit(np.cross(u_hat, v_hat)) if n_axis is None else unit(n_axis)

    U_extent, V_extent = plane_size_xy
    us = np.arange(0, w, stride, dtype=np.int32)
    vs = np.arange(0, h, stride, dtype=np.int32)

    pts = []
    cols = [] if keep_rgb else None

    for v in vs:
        # map v -> V in [0, V_extent]
        V = (v / max(1, h - 1)) * V_extent
        for u in us:
            g = G[v, u]
            if g <= g_min:           # skip near-background
                continue
            d = (g / 255.0) * d_max  # distance magnitude (always + along +n)

            U = (u / max(1, w - 1)) * U_extent
            p_plane = o + U * u_hat + V * v_hat
            p_3d = p_plane + d * n_hat

            pts.append(p_3d)
            if cols is not None:
                b, g8, r = img[v, u]
                cols.append((int(r), int(g8), int(b)))

    if not pts:
        print("No points reconstructed. Check thresholds/inputs.")
        return 0

    pts = np.asarray(pts, dtype=np.float32)
    cols_np = np.asarray(cols, dtype=np.uint8) if cols is not None else None
    write_ply_xyzrgb(out_path, pts, cols_np)
    print(f"✓ Wrote {len(pts)} points -> {out_path}")
    return len(pts)


def main():
    ap = argparse.ArgumentParser(description="Reconstruct 3D point cloud from GREEN-channel AM")
    ap.add_argument("--green", required=True, help="Green AM image path")
    ap.add_argument("--out",   required=True, help="Output .ply path")

    ap.add_argument("--o",       default="0,0,0", help="Plane point o 'x,y,z'")
    ap.add_argument("--u-axis",  default="0,1,0", help="Plane U axis dir 'x,y,z' (default world Y)")
    ap.add_argument("--v-axis",  default="1,0,0", help="Plane V axis dir 'x,y,z'")
    ap.add_argument("--n-axis",  default=None,    help="Plane normal dir; if omitted, use u×v")

    ap.add_argument("--plane-size", default="150,150", help="Plane extent U,V (e.g. '150,150')")
    ap.add_argument("--d-max", type=float, default=30.0, help="Max distance at G=255")
    ap.add_argument("--stride", type=int, default=1, help="Pixel subsampling")
    ap.add_argument("--g-min",  type=int, default=1, help="Ignore pixels with G<=g_min")
    ap.add_argument("--no-rgb", action="store_true", help="Do not store RGB in PLY")
    args = ap.parse_args()

    green = Path(args.green)
    out   = Path(args.out)

    o      = parse_vec3(args.o)
    u_axis = parse_vec3(args.u_axis)
    v_axis = parse_vec3(args.v_axis)
    n_axis = parse_vec3(args.n_axis) if args.n_axis is not None else None

    Uext, Vext = [float(x) for x in args.plane_size.split(",")]

    reconstruct_from_green(
        green_img_path=green,
        out_path=out,
        o=o,
        u_axis=u_axis,
        v_axis=v_axis,
        n_axis=n_axis,
        plane_size_xy=(Uext, Vext),
        d_max=args.d_max,
        stride=args.stride,
        g_min=args.g_min,
        keep_rgb=not args.no_rgb,
    )


if __name__ == "__main__":
    main()
