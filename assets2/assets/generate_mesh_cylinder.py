# import trimesh

# # 圆柱外形参数
# outer_radius = 0.08      # 外半径
# inner_radius = 0.02      # 内半径（掏空部分）
# height = 0.2             # 外圆柱高度
# hole_depth = 0.06        # 孔深度

# # 外圆柱（居中于原点）
# outer_cyl = trimesh.creation.cylinder(radius=outer_radius, height=height)

# # 内圆柱（从底面钻进去）
# inner_cyl = trimesh.creation.cylinder(radius=inner_radius, height=hole_depth)

# # ⚙️ 计算偏移量：
# # Trimesh 创建的圆柱默认中心在 z=0，高度方向范围 [-h/2, +h/2]
# # 我们希望内圆柱的底面与外圆柱的底面对齐
# offset_z = -(height / 2) + (hole_depth / 2)
# inner_cyl.apply_translation([0, 0, offset_z])

# # 布尔差集（掏空底部圆孔）
# hollow_cyl = outer_cyl.difference(inner_cyl)

# # 导出 STL
# hollow_cyl.export('cylinder_with_holes.stl')

# print("✅ cylinder_with_holes.stl")

import trimesh

# 圆柱外形参数
outer_radius = 0.08      # 外半径
inner_radius = 0.02      # 内半径（掏空部分）
height = 0.2             # 外圆柱高度
hole_depth = 0.06        # 孔深度

# 外圆柱（居中于原点）
outer_cyl = trimesh.creation.cylinder(radius=outer_radius, height=height)

# 内圆柱（从底面钻进去）
inner_cyl = trimesh.creation.cylinder(radius=inner_radius, height=hole_depth)

# 对齐底面
offset_z = -(height / 2) + (hole_depth / 2)
inner_cyl.apply_translation([0, 0, offset_z])

# 差集得到带孔圆柱
hollow_cyl = outer_cyl.difference(inner_cyl)

# ✅ 使用 convex decomposition（需要安装 pyvhacd）
# pip install pyvhacd
parts = hollow_cyl.convex_decomposition()

# 导出每个凸块
for i, p in enumerate(parts):
    fname = f"hollow_cylinder_convex_{i}.stl"
    p.export(fname)
    print(f"✅ 导出凸几何块 {i}: {fname}")

print(f"总共生成 {len(parts)} 个凸块。")
