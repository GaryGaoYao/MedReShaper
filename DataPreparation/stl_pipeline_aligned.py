#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STL Mesh Normalization & Heatmap Pipeline

Features
--------
1) Normalize STL meshes:
   - center to origin (centroid -> (0,0,0))
   - translate into positive octant
   - rotate so vector (lowest_y_point -> centroid) aligns with +Y axis
   - (optionally) repeat center+positive shift after rotation
2) Generate heatmaps for files matching '*Aligned.stl' (optional dependency `heatmap`)
3) Utilities: count, check-aligned

CLI
---
Normalize all STL files under a folder (recursively) and write *_Aligned.stl to output dir:
    python stl_pipeline.py normalize --input ./data --output ./out --recursive

Generate heatmaps for already aligned models:
    python stl_pipeline.py heatmap --input ./out --pattern "*_Aligned.stl"

Quick count:
    python stl_pipeline.py count --input ./data --recursive

Check if aligned exists anywhere:
    python stl_pipeline.py check --input ./out
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    import trimesh
except Exception as e:  # pragma: no cover
    print("ERROR: `trimesh` is required. Install via `pip install trimesh`.", file=sys.stderr)
    raise

# Optional module for heatmap
try:
    import heatmap  # type: ignore
    HAS_HEATMAP = True
except Exception:
    HAS_HEATMAP = False


# ----------------------------
# Logging
# ----------------------------
LOG = logging.getLogger("stl-pipeline")


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ----------------------------
# Mesh ops
# ----------------------------
def as_trimesh(obj) -> "trimesh.Trimesh":
    """
    Ensure we have a Trimesh instance (if a Scene was loaded, try to merge).
    """
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, trimesh.Scene):
        LOG.debug("Loaded a Scene; attempting to merge into a single Trimesh.")
        combined = trimesh.util.concatenate(tuple(g for g in obj.dump() if isinstance(g, trimesh.Trimesh)))
        if not isinstance(combined, trimesh.Trimesh):
            raise ValueError("Failed to convert Scene to a single Trimesh.")
        return combined
    raise TypeError(f"Unsupported mesh type: {type(obj)}")


def align_origin(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """
    Translate mesh so that centroid is at the origin.
    """
    c = mesh.centroid
    mesh.apply_translation(-c)
    return mesh


def translate_to_positive_octant(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """
    Translate the mesh so that its bounding box minimum is at (0,0,0).
    """
    min_x, min_y, min_z = mesh.bounds[0]
    mesh.apply_translation([-float(min_x), -float(min_y), -float(min_z)])
    return mesh


def rotate_lowest_y_vec_to_plus_y(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """
    Find the vertex with the smallest Y, construct vector from that vertex to the centroid,
    and rotate the mesh so that this vector aligns with +Y axis.
    """
    y = mesh.vertices[:, 1]
    idx = int(np.argmin(y))
    p1 = mesh.vertices[idx]
    p2 = mesh.centroid
    vec = np.asarray(p2) - np.asarray(p1)

    # If degenerate, skip rotation
    n = np.linalg.norm(vec)
    if n == 0:
        LOG.debug("Zero-length vector from lowest-Y to centroid; skip rotation.")
        return mesh

    vec = vec / n
    target = np.array([0.0, 1.0, 0.0], dtype=float)

    # rotation axis = cross(vec, target)
    axis = np.cross(vec, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        LOG.debug("Vector already aligned with +Y; skip rotation.")
        return mesh
    axis = axis / axis_norm

    # angle = arccos(dot(vec, target))
    dot_val = float(np.clip(np.dot(vec, target), -1.0, 1.0))
    angle = float(np.arccos(dot_val))

    T = trimesh.transformations.rotation_matrix(angle, axis, point=p1)
    mesh.apply_transform(T)
    return mesh


def normalize_one_mesh(
    in_path: Path,
    out_path: Path,
    post_recentering: bool = True,
    overwrite: bool = False,
) -> Optional[Path]:
    """
    Load, normalize, and export one mesh.

    Parameters
    ----------
    in_path : Path
        STL input path
    out_path : Path
        Output STL path
    post_recentering : bool
        If True, perform align_origin + translate_to_positive_octant again after rotation.
    overwrite : bool
        If False and out_path exists, skip.

    Returns
    -------
    Optional[Path]
        The output path if written, else None.
    """
    try:
        if out_path.exists() and not overwrite:
            LOG.info(f"Skip (exists): {out_path}")
            return None

        m_raw = trimesh.load(in_path, force="mesh")
        mesh = as_trimesh(m_raw)

        align_origin(mesh)
        translate_to_positive_octant(mesh)
        rotate_lowest_y_vec_to_plus_y(mesh)

        if post_recentering:
            align_origin(mesh)
            translate_to_positive_octant(mesh)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(out_path)
        LOG.info(f"✓ Normalized -> {out_path.name}")
        return out_path
    except Exception as e:
        LOG.error(f"Failed to normalize {in_path}: {e}")
        return None


# ----------------------------
# Discovery & helpers
# ----------------------------
def iter_stl_files(
    root: Path,
    pattern: str = "*.stl",
    recursive: bool = True,
) -> Iterable[Path]:
    if recursive:
        yield from root.rglob(pattern)
    else:
        yield from root.glob(pattern)


def count_stl_files(path: Path, recursive: bool = True, pattern: str = "*.stl") -> int:
    return sum(1 for _ in iter_stl_files(path, pattern=pattern, recursive=recursive))


def any_aligned_exists(path: Path, recursive: bool = True, suffix: str = "_Aligned.stl") -> bool:
    for p in iter_stl_files(path, pattern=f"*{suffix}", recursive=recursive):
        LOG.info(f"Found aligned: {p}")
        return True
    LOG.info("No aligned STL found.")
    return False


# ----------------------------
# Heatmap
# ----------------------------
def generate_heatmaps(
    path: Path,
    pattern: str = "*_Aligned.stl",
    output_suffix: str = ".png",
) -> int:
    """
    Generate heatmaps for files matching a pattern.

    Returns
    -------
    int
        Number of heatmaps successfully generated.
    """
    if not HAS_HEATMAP:
        LOG.error("`heatmap` module not available. Install or add it to PYTHONPATH.")
        return 0

    ok = 0
    for stl_path in iter_stl_files(path, pattern=pattern, recursive=False):
        try:
            png_path = stl_path.with_suffix(output_suffix)
            heatmap.heatmap_generator_single_pic(str(stl_path), str(png_path))  # type: ignore
            LOG.info(f"✓ Heatmap -> {png_path.name}")
            ok += 1
        except Exception as e:
            LOG.error(f"Failed heatmap for {stl_path}: {e}")
    return ok


# ----------------------------
# CLI
# ----------------------------
def cli_normalize(args: argparse.Namespace) -> None:
    in_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    pattern = args.pattern
    suffix = args.suffix

    if not in_dir.exists():
        LOG.error(f"Input path not found: {in_dir}")
        sys.exit(2)

    files = list(iter_stl_files(in_dir, pattern=pattern, recursive=args.recursive))
    if not files:
        LOG.warning("No STL files matched.")
        return

    LOG.info(f"Found {len(files)} STL files.")
    for p in files:
        rel = p.relative_to(in_dir)
        out_path = (out_dir / rel).with_suffix("").with_name(p.stem + suffix)
        normalize_one_mesh(
            in_path=p,
            out_path=out_path,
            post_recentering=not args.no_post_recentering,
            overwrite=args.overwrite,
        )


def cli_heatmap(args: argparse.Namespace) -> None:
    if not HAS_HEATMAP:
        LOG.error("`heatmap` module not available. Please install/provide it first.")
        sys.exit(2)
    path = Path(args.input).expanduser().resolve()
    n = generate_heatmaps(path, pattern=args.pattern, output_suffix=args.ext)
    LOG.info(f"Heatmaps generated: {n}")


def cli_count(args: argparse.Namespace) -> None:
    path = Path(args.input).expanduser().resolve()
    c = count_stl_files(path, recursive=args.recursive, pattern=args.pattern)
    print(c)


def cli_check(args: argparse.Namespace) -> None:
    path = Path(args.input).expanduser().resolve()
    ok = any_aligned_exists(path, recursive=args.recursive, suffix=args.suffix)
    sys.exit(0 if ok else 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="STL normalization & heatmap CLI")
    p.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity (-v, -vv)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # normalize
    pn = sub.add_parser("normalize", help="normalize STL meshes")
    pn.add_argument("--input", required=True, help="input directory")
    pn.add_argument("--output", required=True, help="output directory for *_Aligned.stl")
    pn.add_argument("--pattern", default="*.stl", help="glob pattern (default: *.stl)")
    pn.add_argument("--suffix", default="_Aligned.stl", help="output file suffix (default: _Aligned.stl)")
    pn.add_argument("--recursive", action="store_true", help="recurse into subfolders")
    pn.add_argument("--overwrite", action="store_true", help="overwrite if output exists")
    pn.add_argument("--no-post-recentering", action="store_true", help="do not recenter after rotation")
    pn.set_defaults(func=cli_normalize)

    # heatmap
    ph = sub.add_parser("heatmap", help="generate heatmaps for aligned models")
    ph.add_argument("--input", required=True, help="folder containing *_Aligned.stl")
    ph.add_argument("--pattern", default="*_Aligned.stl", help="pattern to match aligned meshes")
    ph.add_argument("--ext", default=".png", help="output image extension (default: .png)")
    ph.set_defaults(func=cli_heatmap)

    # count
    pc = sub.add_parser("count", help="count STL files")
    pc.add_argument("--input", required=True, help="input directory")
    pc.add_argument("--pattern", default="*.stl", help="glob pattern (default: *.stl)")
    pc.add_argument("--recursive", action="store_true", help="recurse into subfolders")
    pc.set_defaults(func=cli_count)

    # check
    pk = sub.add_parser("check", help="check whether any aligned STL exists")
    pk.add_argument("--input", required=True, help="directory to check")
    pk.add_argument("--suffix", default="_Aligned.stl", help="aligned suffix (default: _Aligned.stl)")
    pk.add_argument("--recursive", action="store_true", help="recurse into subfolders")
    pk.set_defaults(func=cli_check)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
