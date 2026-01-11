import trimesh

# 方块
overall_height=0.2
box = trimesh.creation.box(extents=[0.1, 0.1, overall_height])

# 四根圆柱
cyls = []
height=0.06
offset_z=(0.2-height)/2
for x in [0.035, -0.035]:
    for y in [0.035, -0.035]:
        c = trimesh.creation.cylinder(radius=0.007, height=0.06)
        c.apply_translation([x, y, -offset_z])
        cyls.append(c)

# 布尔差集
socket = box.difference(trimesh.util.concatenate(cyls))

# 导出 STL
socket.export('socket_with_holes.stl')
