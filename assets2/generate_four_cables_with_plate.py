#!/usr/bin/env python3
"""
Generate iiwa14 variant with four rigid cables and a square plate.
"""

from pathlib import Path
import re


def generate_single_rope_xml(
    rope_name,
    num_segments=10,
    segment_length=0.02,
    root_pos="0 0 0",
    damping=0.02,
    capsule_radius=0.004,
    segment_mass=0.01,
):
    indent_base = " " * 22
    lines = []
    sz = f"{capsule_radius:.6f}"

    root_indent = indent_base
    lines.append(f'{root_indent}<body name="{rope_name}_root" pos="{root_pos}">')
    lines.append(f'{root_indent}  <joint type="ball" damping="{damping}"/>')
    lines.append(f'{root_indent}  <geom type="capsule"')
    lines.append(f'{root_indent}        fromto="0 0 0   0 0 -{segment_length:.6f}"')
    lines.append(f'{root_indent}        size="{sz}"')
    lines.append(f'{root_indent}        mass="{segment_mass}"')
    lines.append(f'{root_indent}        material="rope"/>')
    lines.append("")

    for i in range(1, num_segments - 1):
        indent = root_indent + "  " * i
        lines.append(f'{indent}<body name="{rope_name}_{i}" pos="0 0 -{segment_length:.6f}">')
        lines.append(f'{indent}  <geom type="capsule" fromto="0 0 0   0 0 -{segment_length:.6f}"')
        lines.append(f'{indent}        size="{sz}" mass="{segment_mass}" material="rope"/>')
        lines.append(f'{indent}  <joint type="ball" damping="{damping}"/>')
        lines.append("")

    last_idx = num_segments - 1
    last_indent = root_indent + "  " * last_idx
    lines.append(f'{last_indent}<body name="{rope_name}_{last_idx}" pos="0 0 -{segment_length:.6f}">')
    lines.append(f'{last_indent}  <geom type="capsule" fromto="0 0 0   0 0 -{segment_length:.6f}"')
    lines.append(f'{last_indent}        size="{sz}" mass="{segment_mass}" material="rope"/>')
    lines.append(f'{last_indent}  <joint type="ball" damping="{damping}"/>')
    lines.append("")
    lines.append(f'{last_indent}  <site name="{rope_name}_end" pos="0 0 -{segment_length:.6f}"')
    lines.append(f'{last_indent}        size="0.006" rgba="1 1 0 1"/>')
    lines.append(f'{last_indent}</body>')

    for i in range(last_idx - 1, 0, -1):
        close_indent = root_indent + "  " * i
        lines.append(f"{close_indent}</body>")

    lines.append(f"{root_indent}</body>")
    return "\n".join(lines)


def main():
    # --- 绳索参数（改这里即可；改完运行本脚本重新生成 iiwa14_four_cables_with_plate.xml）---
    # num_segments: 每根绳的刚体段数（链上 capsule 段数）。总绳长 = num_segments * segment_length
    # segment_length: 每一段长度 (m)。例：10 段 × 0.057 ≈ 0.57 m，贴近 plate 吊点到圆柱吊点几何距离
    # rope_damping: 球铰阻尼，越大摆动衰减越快
    # capsule_radius / segment_mass: 绳的视觉半径与每段质量
    NUM_SEGMENTS = 20
    SEGMENT_LENGTH_M = 0.02
    ROPE_DAMPING = 0.02
    CAPSULE_RADIUS = 0.004
    SEGMENT_MASS = 0.01

    assets_dir = Path(__file__).resolve().parent
    input_xml = assets_dir / "iiwa14.xml"
    output_xml = assets_dir / "iiwa14_four_cables_with_plate.xml"

    content = input_xml.read_text(encoding="utf-8")

    # Ensure rope material exists in generated model because rope capsule geoms use material="rope".
    if 'name="rope"' not in content:
        rope_material_line = '    <material name="rope" rgba="0.2 0.2 0.2 1"/>'
        orange_material_line = '    <material class="iiwa" name="orange" rgba="1 0.423529 0.0392157 1"/>'
        if orange_material_line in content:
            content = content.replace(
                orange_material_line,
                f"{orange_material_line}\n{rope_material_line}",
                1,
            )
        else:
            asset_open_tag = "<asset>"
            asset_pos = content.find(asset_open_tag)
            if asset_pos == -1:
                raise ValueError("Cannot find <asset> block in iiwa14.xml")
            insert_after_asset = asset_pos + len(asset_open_tag)
            content = (
                content[:insert_after_asset]
                + f"\n{rope_material_line}"
                + content[insert_after_asset:]
            )

    # Keep attachment_site as compatibility anchor for existing sensors/scripts.
    anchor = '<site pos="0 0 0.045" name="attachment_site" size="0.01" rgba="0 1 1 1" type="sphere" group="1"/>'
    anchor_pos = content.find(anchor)
    if anchor_pos == -1:
        raise ValueError("Cannot find attachment_site in iiwa14.xml")

    insert_pos = anchor_pos + len(anchor)
    before = content[:insert_pos]
    after = content[insert_pos:]

    kw = dict(
        num_segments=NUM_SEGMENTS,
        segment_length=SEGMENT_LENGTH_M,
        damping=ROPE_DAMPING,
        capsule_radius=CAPSULE_RADIUS,
        segment_mass=SEGMENT_MASS,
    )
    rope_fl = generate_single_rope_xml("rope_fl", root_pos="0.05 0.05 0", **kw)
    rope_fr = generate_single_rope_xml("rope_fr", root_pos="0.05 -0.05 0", **kw)
    rope_rl = generate_single_rope_xml("rope_rl", root_pos="-0.05 0.05 0", **kw)
    rope_rr = generate_single_rope_xml("rope_rr", root_pos="-0.05 -0.05 0", **kw)

    hook_block = f"""

                    <!-- end-effector payload plate + rigid cables -->
                    <body name="hook_attachment" pos="0 0 0.045">
                      <geom type="box" size="0.05 0.05 0.01" pos="0 0 0"
                            rgba="0.5 0.5 0.5 0.8" mass="0.1"/>
                      <site name="hook_fl" pos="0.05 0.05 0.05" size="0.01" rgba="0 0 1 1"/>
                      <site name="hook_fr" pos="0.05 -0.05 0.05" size="0.01" rgba="0 0 1 1"/>
                      <site name="hook_rl" pos="-0.05 0.05 0.05" size="0.01" rgba="0 0 1 1"/>
                      <site name="hook_rr" pos="-0.05 -0.05 0.05" size="0.01" rgba="0 0 1 1"/>
                      <camera name="end_effector_cam"
                              pos="0 0 0"
                              xyaxes="1 0 0  0 1 0"
                              fovy="60"/>
{rope_fl}
{rope_fr}
{rope_rl}
{rope_rr}
                    </body>"""

    output_xml.write_text(before + hook_block + after, encoding="utf-8")

    # Keep the training scene XML in sync with rigid-cable model.
    demo_xml = assets_dir / "demo_fourCable_withSteel_withSensor_cylinder.xml"
    if demo_xml.exists():
        demo = demo_xml.read_text(encoding="utf-8")
        demo = re.sub(
            r'<include file="[^"]+"/>',
            '<include file="iiwa14_four_cables_with_plate.xml"/>',
            demo,
            count=1,
        )
        # `rope` is defined in included iiwa14_four_cables_with_plate.xml; avoid duplicated material names.
        demo = re.sub(r'\n\s*<material name="rope"[^>]*/>', "", demo, count=1)

        if 'site1="rope_fl_end"' not in demo:
            demo = re.sub(
                r"<equality>[\s\S]*?</equality>",
                """<equality>
      <!-- 使用刚体绳索末端与负载四吊点连接 -->
      <connect site1="rope_fl_end" site2="lift_fl"/>
      <connect site1="rope_fr_end" site2="lift_fr"/>
      <connect site1="rope_rl_end" site2="lift_rl"/>
      <connect site1="rope_rr_end" site2="lift_rr"/>
  </equality>""",
                demo,
                count=1,
            )
            demo = re.sub(r"\n\s*<!-- 四根吊索 -->[\s\S]*?</tendon>", "", demo, count=1)

        demo_xml.write_text(demo, encoding="utf-8")
        print(f"Updated: {demo_xml}")
    total_m = NUM_SEGMENTS * SEGMENT_LENGTH_M
    print(f"Generated: {output_xml}")
    print(f"  segments={NUM_SEGMENTS}, segment_length={SEGMENT_LENGTH_M} m -> total chain length {total_m:.4f} m")


if __name__ == "__main__":
    main()
