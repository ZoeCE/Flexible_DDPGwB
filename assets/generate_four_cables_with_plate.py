#!/usr/bin/env python3
"""
Generate iiwa14 variant with four rigid cables and a square plate.
"""

from pathlib import Path
import sys
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
    # 绳索沿 plate 局部 +Z 方向生长。
    # 当末端执行器倒立（绕X轴180°）时，局部 +Z 对应世界 -Z，
    # 绳索自然垂向地面，缩短绳末端与 prefab 的距离。
    indent_base = " " * 22
    lines = []
    sz = f"{capsule_radius:.6f}"

    root_indent = indent_base
    lines.append(f'{root_indent}<body name="{rope_name}_root" pos="{root_pos}">')
    lines.append(f'{root_indent}  <joint type="ball" damping="{damping}"/>')
    lines.append(f'{root_indent}  <geom type="capsule"')
    lines.append(f'{root_indent}        fromto="0 0 0   0 0 {segment_length:.6f}"')
    lines.append(f'{root_indent}        size="{sz}"')
    lines.append(f'{root_indent}        mass="{segment_mass}"')
    lines.append(f'{root_indent}        material="rope"/>')
    lines.append("")

    for i in range(1, num_segments - 1):
        indent = root_indent + "  " * i
        lines.append(f'{indent}<body name="{rope_name}_{i}" pos="0 0 {segment_length:.6f}">')
        lines.append(f'{indent}  <geom type="capsule" fromto="0 0 0   0 0 {segment_length:.6f}"')
        lines.append(f'{indent}        size="{sz}" mass="{segment_mass}" material="rope"/>')
        lines.append(f'{indent}  <joint type="ball" damping="{damping}"/>')
        lines.append("")

    last_idx = num_segments - 1
    last_indent = root_indent + "  " * last_idx
    lines.append(f'{last_indent}<body name="{rope_name}_{last_idx}" pos="0 0 {segment_length:.6f}">')
    lines.append(f'{last_indent}  <geom type="capsule" fromto="0 0 0   0 0 {segment_length:.6f}"')
    lines.append(f'{last_indent}        size="{sz}" mass="{segment_mass}" material="rope"/>')
    lines.append(f'{last_indent}  <joint type="ball" damping="{damping}"/>')
    lines.append("")
    lines.append(f'{last_indent}  <site name="{rope_name}_end" pos="0 0 {segment_length:.6f}"')
    lines.append(f'{last_indent}        size="0.006" rgba="1 1 0 1"/>')
    lines.append(f'{last_indent}</body>')

    for i in range(last_idx - 1, 0, -1):
        close_indent = root_indent + "  " * i
        lines.append(f"{close_indent}</body>")

    lines.append(f"{root_indent}</body>")
    return "\n".join(lines)


def main():
    # Read all rope parameters from config.py (single source of truth)
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    import importlib
    import config
    importlib.reload(config)
    from config import DEFAULT_CONFIG

    cfg_rope   = DEFAULT_CONFIG["rope"]
    cfg_prefab = DEFAULT_CONFIG["prefab"]
    cfg_target = DEFAULT_CONFIG["target"]
    NUM_SEGMENTS     = cfg_rope["num_segments"]
    SEGMENT_LENGTH_M = cfg_rope["segment_length"]
    ROPE_DAMPING     = cfg_rope["damping"]
    CAPSULE_RADIUS   = cfg_rope["capsule_radius"]
    SEGMENT_MASS     = cfg_rope["segment_mass"]

    plate_hs = cfg_rope["plate_half_size"]
    plate_mass = cfg_rope["plate_mass"]
    hook_off = cfg_rope["hook_offset"]
    total_rope = NUM_SEGMENTS * SEGMENT_LENGTH_M

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
    h = hook_off
    px, py, pz = plate_hs

    rope_fl = generate_single_rope_xml("rope_fl", root_pos=f"{h} {h} 0", **kw)
    rope_fr = generate_single_rope_xml("rope_fr", root_pos=f"{h} {-h} 0", **kw)
    rope_rl = generate_single_rope_xml("rope_rl", root_pos=f"{-h} {h} 0", **kw)
    rope_rr = generate_single_rope_xml("rope_rr", root_pos=f"{-h} {-h} 0", **kw)

    hook_block = f"""

                    <!-- end-effector payload plate + rigid cables -->
                    <body name="hook_attachment" pos="0 0 0.045">
                      <geom type="box" size="{px} {py} {pz}" pos="0 0 0"
                            rgba="0.5 0.5 0.5 0.8" mass="{plate_mass}"/>
                      <site name="hook_fl" pos="{h} {h} {h}" size="0.01" rgba="0 0 1 1"/>
                      <site name="hook_fr" pos="{h} {-h} {h}" size="0.01" rgba="0 0 1 1"/>
                      <site name="hook_rl" pos="{-h} {h} {h}" size="0.01" rgba="0 0 1 1"/>
                      <site name="hook_rr" pos="{-h} {-h} {h}" size="0.01" rgba="0 0 1 1"/>
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

        # Always rewrite equality block with weld + correct rope mapping
        demo = re.sub(
            r"<equality>[\s\S]*?</equality>",
            """<equality>
      <!-- rope ends to prefab lift sites (fl/rl swapped with fr/rr) -->
      <connect site1="rope_fl_end" site2="lift_fr"/>
      <connect site1="rope_fr_end" site2="lift_fl"/>
      <connect site1="rope_rl_end" site2="lift_rr"/>
      <connect site1="rope_rr_end" site2="lift_rl"/>
  </equality>""",
            demo,
            count=1,
        )
        demo = re.sub(r"\n\s*<!-- 四根吊索 -->[\s\S]*?</tendon>", "", demo, count=1)

        # --- Rewrite prefab body from config ---
        pshape = cfg_prefab["shape"]
        pmass = cfg_prefab["mass"]
        p_lift_z = cfg_prefab["lift_site_offset"]
        p_lift_s = cfg_prefab["lift_site_spread"]
        if pshape == "box":
            bhs = cfg_prefab["box_half_size"]
            prefab_geom = f'<geom type="box" size="{bhs[0]} {bhs[1]} {bhs[2]}" material="concrete" mass="{pmass}"/>'
        else:
            cr = cfg_prefab["cylinder_radius"]
            ch = cfg_prefab["cylinder_half_height"]
            prefab_geom = f'<geom type="cylinder" size="{cr} {ch}" material="concrete" mass="{pmass}"/>'

        prefab_body = f"""<body name="prefab" pos="0.3 0.15 0.5">
      <joint type="free"/>
      {prefab_geom}

      <!-- four lift sites at top corners -->
      <site name="lift_fl" pos=" {p_lift_s}  {p_lift_s}  {p_lift_z}" size="0.01" rgba="1 0 0 1"/>
      <site name="lift_fr" pos=" {p_lift_s} -{p_lift_s}  {p_lift_z}" size="0.01" rgba="1 0 0 1"/>
      <site name="lift_rl" pos="-{p_lift_s}  {p_lift_s}  {p_lift_z}" size="0.01" rgba="1 0 0 1"/>
      <site name="lift_rr" pos="-{p_lift_s} -{p_lift_s}  {p_lift_z}" size="0.01" rgba="1 0 0 1"/>

      <!-- IMU sensor mount -->
      <site name="imu_site" pos="0 0 0" size="0.01" rgba="0 1 0 1"/>
    </body>"""
        demo = re.sub(
            r'<body name="prefab"[\s\S]*?</body>\s*\n',
            prefab_body + '\n',
            demo,
            count=1,
        )

        # --- Rewrite target body from config ---
        tshape = cfg_target["shape"]
        trgba = cfg_target["rgba"]
        rgba_str = f"{trgba[0]} {trgba[1]} {trgba[2]} {trgba[3]}"
        if tshape == "box":
            ths = cfg_target["box_half_size"]
            target_geom = f'<geom type="box" size="{ths[0]} {ths[1]} {ths[2]}" pos="0 0 {ths[2]}"'
        else:
            tr = cfg_target["cylinder_radius"]
            th = cfg_target["cylinder_half_height"]
            target_geom = f'<geom type="cylinder" size="{tr} {th}" pos="0 0 {th}"'
        target_geom += f'\n            rgba="{rgba_str}" contype="0" conaffinity="0"/>'

        target_body = f"""<body name="target" pos="-0.3 0.2 0">
      {target_geom}
    </body>"""
        demo = re.sub(
            r'<body name="target"[\s\S]*?</body>',
            target_body,
            demo,
            count=1,
        )

        # Remove old mesh references if any remain
        demo = re.sub(r'\s*<mesh name="hollow_cylinder_convex_\d+"[^/]*/>', '', demo)
        demo = re.sub(r'\s*<material name="steel"[^/]*/>', '', demo)
        demo = re.sub(r'\s*<material name="red"[^/]*/>', '', demo)

        demo_xml.write_text(demo, encoding="utf-8")
        print(f"Updated: {demo_xml}")
    total_m = NUM_SEGMENTS * SEGMENT_LENGTH_M
    print(f"Generated: {output_xml}")
    print(f"  segments={NUM_SEGMENTS}, segment_length={SEGMENT_LENGTH_M} m -> total chain length {total_m:.4f} m")


if __name__ == "__main__":
    main()
