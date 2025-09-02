#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from es_module import ExplosiveSliding, ESConfig, green_am_to_points
import numpy as np
import cv2
from pathlib import Path

stl_path       = "/path/case001_C_Aligned.stl"       # 预对齐/归一化后的 STL
defect_mask_p  = "/path/defect_mask.png"            # 与 P-AM 同分辨率的二值缺损掩膜(白=1/黑=0 或 255/0)
out_best_green = "/path/best_reconstruction_green.png"
out_best_ply   = "/path/best_reconstruction_green.ply"  # 可选：把最优AM回投为点云

# -------- 2) 解剖轴与平面定义（至少需要 o 和 轴上另一点）--------
# o：Gravity point（可用 STL 质心近似），axis_p1：Dacryon/Orbitale 中点
def estimate_o_and_axis(stl_file: str):
    import trimesh
    mesh = trimesh.load(stl_file)
    o = mesh.centroid.astype(float)
    axis_p1 = o + np.array([0.0, 1.0, 0.0], dtype=float)  # 占位：全局Y
    return o, axis_p1

o, axis_p1 = estimate_o_and_axis(stl_path)

# 若你已有 D & Or 手动点位（dacryon/orbitale），可使用：
# D = np.array([dx,dy,dz], float); Or = np.array([ox,oy,oz], float)
# o = np.array([gx,gy,gz], float)  # gravity point
# axis_p1 = 0.5 * (D + Or)

# -------- 3) ES 参数 --------
cfg = ESConfig(
    alpha_deg=5.0,                  # 步长 α（度）
    beta_deg=30.0,                  # 范围 β（度）=> 遍历 -30,-25,...,0,...,+30
    image_size=(1000, 1000),        # AM 分辨率
    plane_span_xy=(150.0, 150.0),   # 平面覆盖范围（需与你生成AM时一致）
    d_max=30.0,                     # 绿色通道255对应的最大距离
    stride_backproj=2,              # 回投点云的像素步长
    sign_backproj=+1.0,             # 回投沿 +n（如果你以前定义是负方向，改为 -1.0）
    g_min=1
)

# -------- 4) 占位推理函数（把输入绿色AM原样返回）--------
def asm_infer_green_stub(green_img_bgr: np.ndarray) -> np.ndarray:
    # 你也可以在这里做一些小处理，如中值/高斯滤波，用于测试
    # green = cv2.medianBlur(green_img_bgr[:, :, 1], 3)
    # out = green_img_bgr.copy()
    # out[:, :, 1] = green
    # return out
    return green_img_bgr

defect_mask = cv2.imread(defect_mask_p, cv2.IMREAD_GRAYSCALE)
if defect_mask is None:
    raise FileNotFoundError(f"Cannot read defect mask: {defect_mask_p}")
# 统一成 {0,1}
_, defect_mask_bin = cv2.threshold(defect_mask, 127, 1, cv2.THRESH_BINARY)

# -------- 6) 运行 ES --------
es = ExplosiveSliding(stl_path, sample_points=2_000_000)
best_img, (alpha_used, best_angle), stats = es.run(
    o=o, axis_p1=axis_p1, es_cfg=cfg,
    asm_infer_fn=asm_infer_green_stub,
    defect_mask_img=defect_mask_bin
)

print("Best angle (deg):", best_angle)
for ang, scd, m in stats:
    print(f"angle={ang:>+5.1f}°, SCD={scd:.6f}, |Q|={int(m)}")

cv2.imwrite(out_best_green, best_img)
print("✓ Saved best green AM ->", out_best_green)

x_axis = np.array([0.0, 1.0, 0.0], dtype=float)                  # 与 es_module 中一致
y_axis = np.array([o[0], 0.0, o[2]], dtype=float)
Q = green_am_to_points(
    green_img=best_img,
    o=o, u_axis=x_axis, v_axis=y_axis, n_axis=None,
    plane_span_xy=cfg.plane_span_xy, d_max=cfg.d_max,
    stride=cfg.stride_backproj, sign=cfg.sign_backproj, g_min=cfg.g_min
)

def write_ply_xyz(points: np.ndarray, path: str):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

if out_best_ply:
    Path(out_best_ply).parent.mkdir(parents=True, exist_ok=True)
    write_ply_xyz(Q, out_best_ply)
    print("✓ Saved reconstructed point cloud (PLY) ->", out_best_ply)
