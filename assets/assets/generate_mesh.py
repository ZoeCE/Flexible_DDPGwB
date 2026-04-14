"""
生成带方孔的方块插座 (socket)，并进行凸分解导出。
依赖: pip install trimesh vhacdx

所有几何参数集中在顶部，可按需调整。
"""
import trimesh
from pathlib import Path

# ======================= 可调参数 =======================
# 方块外形（半尺寸 × 2 = 实际尺寸）
BOX_EXTENTS = [0.1, 0.1, 0.2]       # [x, y, z] 总尺寸 (m)

# 方孔参数
HOLE_SIZE   = [0.014, 0.014]         # 方孔截面 [x, y] (m)
HOLE_DEPTH  = 0.06                   # 孔深 (m)，从底面向上
HOLE_POSITIONS = [                   # 孔中心在底面的 XY 坐标
    ( 0.035,  0.035),
    ( 0.035, -0.035),
    (-0.035,  0.035),
    (-0.035, -0.035),
]
# ========================================================

here = Path(__file__).resolve().parent

# 1. 创建外壳方块
box = trimesh.creation.box(extents=BOX_EXTENTS)

# 2. 挖方孔（用小方块做布尔差集）
holes = []
offset_z = -(BOX_EXTENTS[2] / 2) + (HOLE_DEPTH / 2)
for (hx, hy) in HOLE_POSITIONS:
    hole = trimesh.creation.box(extents=[HOLE_SIZE[0], HOLE_SIZE[1], HOLE_DEPTH])
    hole.apply_translation([hx, hy, offset_z])
    holes.append(hole)

socket = box.difference(trimesh.util.concatenate(holes))

# 3. 导出整体 mesh（留作参考）
socket.export(str(here / "socket_with_holes.stl"))
print(f"Exported: socket_with_holes.stl  (vertices={len(socket.vertices)}, faces={len(socket.faces)})")

# 4. 凸分解 + 导出子块
parts = socket.convex_decomposition()
for i, p in enumerate(parts):
    p.export(str(here / f"socket_convex_{i}.stl"))

print(f"Done: generated {len(parts)} convex parts (socket_convex_0..{len(parts)-1}.stl)")
print(f"\nConfig hint:")
print(f'  "mesh_prefix": "socket_convex"')
print(f'  "mesh_count":  {len(parts)}')
