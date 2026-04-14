"""
对 socket_with_holes.stl 进行凸分解，生成多个子 STL 文件。
依赖: pip install trimesh pyvhacd

用法: python decompose_socket.py
输出: socket_convex_0.stl, socket_convex_1.stl, ...
"""
import trimesh
from pathlib import Path

here = Path(__file__).resolve().parent
input_path = here / "socket_with_holes.stl"

mesh = trimesh.load(str(input_path))
print(f"Loaded: {input_path}  (vertices={len(mesh.vertices)}, faces={len(mesh.faces)})")

parts = mesh.convex_decomposition()

for i, p in enumerate(parts):
    fname = here / f"socket_convex_{i}.stl"
    p.export(str(fname))

print(f"Done: generated {len(parts)} convex parts (socket_convex_0..{len(parts)-1}.stl)")
