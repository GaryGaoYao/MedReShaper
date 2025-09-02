#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point cloud (.xyz) -> surface (VTK SurfaceReconstructionFilter or Delaunay2D) -> STL

- SurfaceReconstructionFilter: 适合一般三维点云（带一定光顺），需用 Contour 取等值面
- Delaunay2D: 仅适合“近似单片、近似共面”的点云（例如一张投影片的轻微起伏）
用法：
python pc_to_surface.py --xyz /path/output_gd.xyz --out /path/output_gd.stl --method surfrec
python pc_to_surface.py --xyz /path/output_gd.xyz --out /path/output_gd.stl --method delaunay --clean --smooth
"""

import argparse
from pathlib import Path
import numpy as np
import vtk


# ---------- IO ----------
def load_xyz_to_vtk_points(xyz_path: str) -> vtk.vtkPoints:
    pts = vtk.vtkPoints()
    with open(xyz_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            x, y, z = map(float, parts[:3])
            pts.InsertNextPoint(x, y, z)
    return pts


def write_polydata_to_stl(poly: vtk.vtkPolyData, stl_path: str, binary: bool = True):
    Path(stl_path).parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(stl_path)
    writer.SetInputData(poly)
    writer.SetFileTypeToBinary() if binary else writer.SetFileTypeToASCII()
    ok = writer.Write()
    if not ok:
        raise RuntimeError(f"Failed to write STL: {stl_path}")
    print(f"✓ Saved STL -> {stl_path}")


# ---------- utils ----------
def estimate_median_nn_spacing(points: vtk.vtkPoints), k: int = 16) -> float:
    """
    粗估点间距（中位最近邻距离），用于给 SurfaceReconstruction 的 SampleSpacing 提个参考。
    """
    n = points.GetNumberOfPoints()
    if n < 2:
        return 1.0
    import random, math
    idxs = random.sample(range(n), min(5000, n))
    ds = []
    for i in idxs:
        xi, yi, zi = points.GetPoint(i)
        mind = float("inf")
        for j in range(max(0, i-50), min(n, i+50)):  # 粗略邻域；若点很多建议换 Kd-tree
            if j == i: continue
            xj, yj, zj = points.GetPoint(j)
            d = (xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2
            if d < mind:
                mind = d
        if mind < float("inf"):
            ds.append(mind**0.5)
    if not ds:
        return 1.0
    return float(np.median(ds))


def clean_polydata(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.SetTolerance(0.0)    # 精确去重
    clean.Update()
    return clean.GetOutput()


def smooth_polydata(poly: vtk.vtkPolyData, iters: int = 20, passband: float = 0.01) -> vtk.vtkPolyData:
    """
    Windowed Sinc 平滑，尽量保持体积。
    passband 越小越光滑（0.001~0.1合理范围）。
    """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetNumberOfIterations(iters)
    smoother.SetPassBand(passband)
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOn()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()


def ensure_outward_normals(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """
    某些 pipeline 输出法向或单元顺序会反；可在可视化时发现“黑面”。这里用 ReverseSense 翻一下。
    """
    rev = vtk.vtkReverseSense()
    rev.SetInputData(poly)
    rev.ReverseCellsOn()
    rev.ReverseNormalsOn()
    rev.Update()
    return rev.GetOutput()


# ---------- methods ----------
def reconstruct_surface_surfrec(points: vtk.vtkPoints,
                                neighborhood: int = 50,
                                sample_spacing: float | None = None) -> vtk.vtkPolyData:
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    if sample_spacing is None:
        # 粗估 spacing（保守一点）
        ss_est = estimate_median_nn_spacing(points)
        sample_spacing = max(0.2 * ss_est, 1e-3)

    surf = vtk.vtkSurfaceReconstructionFilter()
    surf.SetInputData(poly)
    surf.SetNeighborhoodSize(neighborhood)  # 默认50可用，稀疏可调大、稠密可调小
    surf.SetSampleSpacing(sample_spacing)   # 关键参数
    # surf.SetHoleFilling(True)  # 如果 VTK 版本支持，可开启
    # surf.SetInterpolation(1)   # 细调

    contour = vtk.vtkContourFilter()
    contour.SetInputConnection(surf.GetOutputPort())
    contour.SetValue(0, 0.0)   # 等值面 0.0
    contour.Update()

    out = vtk.vtkPolyData()
    out.ShallowCopy(contour.GetOutput())
    return out


def reconstruct_surface_delaunay2d(points: vtk.vtkPoints) -> vtk.vtkPolyData:
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    # 可选：投影到 PCA 平面再 Delaunay2D（当存在轻微曲度时更稳）
    # 这里直接使用 2D 三角化：
    del2d = vtk.vtkDelaunay2D()
    del2d.SetInputData(poly)
    del2d.SetTolerance(0.01)
    del2d.Update()

    surf = vtk.vtkDataSetSurfaceFilter()
    surf.SetInputConnection(del2d.GetOutputPort())
    surf.Update()

    out = vtk.vtkPolyData()
    out.ShallowCopy(surf.GetOutput())
    return out


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Reconstruct surface from .xyz point cloud to STL")
    ap.add_argument("--xyz", required=True, help="Input XYZ path")
    ap.add_argument("--out", required=True, help="Output STL path")
    ap.add_argument("--method", choices=["surfrec", "delaunay"], default="surfrec",
                    help="surfrec: vtkSurfaceReconstructionFilter; delaunay: vtkDelaunay2D")
    ap.add_argument("--neighborhood", type=int, default=50, help="SurfRec NeighborhoodSize")
    ap.add_argument("--sample-spacing", type=float, default=-1.0,
                    help="SurfRec SampleSpacing; <=0 自动估计")
    ap.add_argument("--clean", action="store_true", help="Run vtkCleanPolyData before save")
    ap.add_argument("--smooth", action="store_true", help="Run WindowedSinc smoothing before save")
    ap.add_argument("--iters", type=int, default=20, help="Smoothing iterations")
    ap.add_argument("--passband", type=float, default=0.01, help="Smoothing passband")
    ap.add_argument("--reverse", action="store_true", help="Reverse cells & normals before save")
    ap.add_argument("--ascii", action="store_true", help="Write ASCII STL (default binary)")
    args = ap.parse_args()

    pts = load_xyz_to_vtk_points(args.xyz)

    if args.method == "surfrec":
        spacing = None if args.sample_spacing <= 0 else float(args.sample_spacing)
        poly = reconstruct_surface_surfrec(pts, neighborhood=args.neighborhood, sample_spacing=spacing)
    else:
        poly = reconstruct_surface_delaunay2d(pts)

    if args.clean:
        poly = clean_polydata(poly)
    if args.smooth:
        poly = smooth_polydata(poly, iters=args.iters, passband=args.passband)
    if args.reverse:
        poly = ensure_outward_normals(poly)

    write_polydata_to_stl(poly, args.out, binary=not args.ascii)


if __name__ == "__main__":
    main()
