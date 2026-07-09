#!/usr/bin/env python3
"""
Generate iiwa14 variant with four rigid cables and a square plate.
"""

from pathlib import Path
import sys
import re
import math
import os


def _fmt_qpos_value(value):
    value = float(value)
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _default_home_keyframe_xml(cfg, num_segments):
    """Return the demo XML keyframe used by the default 10-segment model."""
    if int(num_segments) != 10:
        return ""
    arm_home = list(cfg.get("reset", {}).get("init_qpos_arm", []))
    if len(arm_home) != 7:
        return ""
    rope_quats = [1, 0, 0, 0] * (4 * int(num_segments))
    prefab_home = [0.2, 0.3, 0.5, 1, 0, 0, 0]
    qpos = " ".join(_fmt_qpos_value(v) for v in (
        arm_home + rope_quats + prefab_home))
    return (
        '  <!-- Full nq vector: arm home + rope ball joints + prefab free joint -->\n'
        '  <keyframe>\n'
        f'    <key name="home" qpos="{qpos}"/>\n'
        '  </keyframe>')


def _ensure_vision_marker_assets(assets_dir, cfg_vision):
    """Generate marker debug textures and return MuJoCo asset/geom snippets."""
    if not bool(cfg_vision.get("render_markers", True)):
        return "", ""
    opencv_cfg = cfg_vision.get("opencv", {})
    markers = list(opencv_cfg.get("markers", []))
    if not markers:
        return "", ""
    try:
        import cv2
        from vision_rgbd import get_aruco_dictionary
        dictionary = get_aruco_dictionary(
            opencv_cfg.get("dictionary", "DICT_APRILTAG_36h11"),
            opencv_cfg.get("dictionary_fallback", "DICT_4X4_50"))
    except Exception as exc:
        print(f"[vision markers] skipped marker generation: {exc}")
        return "", ""

    tex_px = int(cfg_vision.get("marker_texture_px", 512))
    render_mode = str(cfg_vision.get("marker_render_mode", "geom_grid")).lower()
    border_bits = int(cfg_vision.get("marker_border_bits", 1))
    marker_size = int(getattr(dictionary, "markerSize", 6))
    modules = marker_size + 2 * border_bits
    module_px = max(8, tex_px // max(1, modules))
    grid_px = int(modules * module_px)
    asset_lines = []
    geom_lines = []

    def _valid_png(path):
        try:
            return (path.is_file() and path.stat().st_size > 128 and
                    cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is not None)
        except Exception:
            return False

    for marker in markers:
        marker_id = int(marker.get("id", 0))
        length = float(marker.get("length", 0.06))
        center = [float(v) for v in marker.get("center", [0.0, 0.0, 0.10])]
        x_axis = _normalize_vec(marker.get("x_axis", [1.0, 0.0, 0.0]))
        y_axis_raw = _normalize_vec(marker.get("y_axis", [0.0, 1.0, 0.0]))
        if x_axis is None:
            x_axis = [1.0, 0.0, 0.0]
        if y_axis_raw is None:
            y_axis_raw = [0.0, 1.0, 0.0]
        xy_dot = _dot(x_axis, y_axis_raw)
        y_axis = _normalize_vec([
            y_axis_raw[i] - xy_dot * x_axis[i] for i in range(3)
        ])
        if y_axis is None:
            y_axis = [0.0, 1.0, 0.0] if abs(x_axis[1]) < 0.9 else [1.0, 0.0, 0.0]
        normal = _normalize_vec(_cross(x_axis, y_axis))
        if normal is None:
            normal = [0.0, 0.0, 1.0]
        marker_quat = _quat_wxyz_from_matrix([
            [x_axis[0], y_axis[0], normal[0]],
            [x_axis[1], y_axis[1], normal[1]],
            [x_axis[2], y_axis[2], normal[2]],
        ])
        marker_orient = f'quat="{_format_float_vec(marker_quat)}"'

        def marker_pos(u, v, w):
            return [
                center[i] + float(u) * x_axis[i] +
                float(v) * y_axis[i] + float(w) * normal[i]
                for i in range(3)
            ]

        name = f"vision_marker_{marker_id}"
        img_path = assets_dir / f"{name}.png"
        if not _valid_png(img_path):
            if hasattr(cv2.aruco, "generateImageMarker"):
                try:
                    img = cv2.aruco.generateImageMarker(
                        dictionary, marker_id, tex_px, None, border_bits)
                except TypeError:
                    img = cv2.aruco.generateImageMarker(
                        dictionary, marker_id, tex_px)
            else:
                try:
                    img = cv2.aruco.drawMarker(
                        dictionary, marker_id, tex_px, borderBits=border_bits)
                except TypeError:
                    img = cv2.aruco.drawMarker(dictionary, marker_id, tex_px)
            tmp_path = img_path.with_name(
                f"{img_path.stem}.{os.getpid()}.tmp{img_path.suffix}")
            ok = cv2.imwrite(str(tmp_path), img)
            if not ok or not _valid_png(tmp_path):
                raise RuntimeError(f"failed to write marker PNG: {tmp_path}")
            tmp_path.replace(img_path)
        asset_lines.append(
            f'    <texture name="{name}_tex" type="2d" file="{img_path.name}"/>')
        asset_lines.append(
            f'    <material name="{name}_mat" texture="{name}_tex" '
            f'texrepeat="1 1" texuniform="true" rgba="1 1 1 1"/>')
        asset_lines.append(
            f'    <material name="{name}_white_mat" rgba="1 1 1 1"/>')
        asset_lines.append(
            f'    <material name="{name}_black_mat" rgba="0 0 0 1"/>')
        if render_mode == "texture":
            half = 0.5 * length
            half_z = 0.0005
            px, py, pz = marker_pos(0.0, 0.0, -half_z)
            geom_lines.append(
                f'      <geom name="{name}" type="box" '
                f'size="{half:.6f} {half:.6f} {half_z:.6f}" '
                f'pos="{px:.6f} {py:.6f} {pz:.6f}" {marker_orient} '
                f'material="{name}_mat" contype="0" conaffinity="0" '
                f'mass="0.000001"/>')
            continue

        board_margin = float(marker.get(
            "board_margin", cfg_vision.get("marker_board_margin", 0.008)))
        board_side = float(marker.get("board_length", length + 2.0 * board_margin))
        board_side = max(length, min(board_side, 0.098))
        marker_px = max(8, min(tex_px, int(round(tex_px * length / board_side))))
        if hasattr(cv2.aruco, "generateImageMarker"):
            try:
                marker_img = cv2.aruco.generateImageMarker(
                    dictionary, marker_id, marker_px, None, border_bits)
            except TypeError:
                marker_img = cv2.aruco.generateImageMarker(
                    dictionary, marker_id, marker_px)
        else:
            try:
                marker_img = cv2.aruco.drawMarker(
                    dictionary, marker_id, marker_px, borderBits=border_bits)
            except TypeError:
                marker_img = cv2.aruco.drawMarker(
                    dictionary, marker_id, marker_px)
        pad0 = max(0, (tex_px - marker_px) // 2)
        pad1 = max(0, tex_px - marker_px - pad0)
        marker_canvas = cv2.copyMakeBorder(
            marker_img, pad0, pad1, pad0, pad1,
            cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(str(img_path), marker_canvas)
        base_half_z = 0.0005
        bx, by, bz = marker_pos(0.0, 0.0, -base_half_z)
        geom_lines.append(
            f'      <geom name="{name}_board" type="box" '
            f'size="{0.5 * board_side:.6f} {0.5 * board_side:.6f} {base_half_z:.6f}" '
            f'pos="{bx:.6f} {by:.6f} {bz:.6f}" {marker_orient} '
            f'material="{name}_white_mat" contype="0" conaffinity="0" '
            f'mass="0.00000001"/>')

        if hasattr(cv2.aruco, "generateImageMarker"):
            try:
                grid_img = cv2.aruco.generateImageMarker(
                    dictionary, marker_id, grid_px, None, border_bits)
            except TypeError:
                grid_img = cv2.aruco.generateImageMarker(
                    dictionary, marker_id, grid_px)
        else:
            try:
                grid_img = cv2.aruco.drawMarker(
                    dictionary, marker_id, grid_px, borderBits=border_bits)
            except TypeError:
                grid_img = cv2.aruco.drawMarker(dictionary, marker_id, grid_px)
        cell = length / float(modules)
        cell_half = 0.5 * cell
        black_half_z = 0.00012
        for row in range(modules):
            for col in range(modules):
                px = int((col + 0.5) * module_px)
                py = int((row + 0.5) * module_px)
                if int(grid_img[py, px]) > 127:
                    continue
                u = -0.5 * length + (col + 0.5) * cell
                v = 0.5 * length - (row + 0.5) * cell
                gx, gy, gz = marker_pos(u, v, black_half_z)
                geom_lines.append(
                    f'      <geom name="{name}_blk_{row}_{col}" type="box" '
                    f'size="{cell_half:.6f} {cell_half:.6f} {black_half_z:.6f}" '
                    f'pos="{gx:.6f} {gy:.6f} {gz:.6f}" {marker_orient} '
                    f'material="{name}_black_mat" contype="0" conaffinity="0" '
                    f'mass="0.00000001"/>')
    return "\n".join(asset_lines), "\n".join(geom_lines)


def _format_float_vec(values, precision=6):
    return " ".join(f"{float(v):.{precision}f}" for v in values)


def _normalize_vec(values):
    vec = [float(v) for v in values]
    n = math.sqrt(sum(v * v for v in vec))
    if n < 1e-12:
        return None
    return [v / n for v in vec]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a, b):
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def _rope_marker_specs(num_segments, marker_cfg):
    """Marker placement by normalized rope arclength, independent of segment count."""
    if not bool(marker_cfg.get("enabled", False)):
        return []
    nseg = max(1, int(num_segments))
    frac = max(0.05, min(0.95, float(marker_cfg.get(
        "position_fraction", 0.55))))
    explicit = marker_cfg.get("segment_indices", None)
    if explicit:
        positions = []
        for idx in explicit:
            ii = int(idx)
            if 0 <= ii < nseg:
                positions.append((ii + frac) / float(nseg))
    else:
        n_markers = max(0, int(marker_cfg.get("markers_per_rope", 5)))
        if n_markers <= 0:
            return []
        if bool(marker_cfg.get("lower_half_only", True)):
            positions = [
                0.5 + 0.5 * (mi + frac) / float(n_markers)
                for mi in range(n_markers)
            ]
        elif n_markers == 1:
            positions = [0.5]
        else:
            positions = [
                (mi + frac) / float(n_markers)
                for mi in range(n_markers)
            ]

    specs = []
    for marker_idx, pos_frac in enumerate(positions):
        pos_unit = max(0.0, min(float(pos_frac) * nseg, nseg - 1e-6))
        segment_idx = int(math.floor(pos_unit))
        local_fraction = pos_unit - segment_idx
        specs.append({
            "marker_idx": marker_idx,
            "segment_idx": segment_idx,
            "local_fraction": max(1e-4, min(0.9999, local_fraction)),
        })
    return specs


def _rope_marker_lines(rope_name, segment_idx, marker_idx, indent,
                       segment_length, marker_cfg, local_fraction=None):
    if not bool(marker_cfg.get("enabled", False)):
        return []
    if local_fraction is not None:
        frac = max(1e-4, min(0.9999, float(local_fraction)))
    else:
        frac = max(0.05, min(0.95, float(marker_cfg.get(
            "position_fraction", 0.55))))
    z = float(segment_length) * frac
    marker_radius = max(1e-4, float(marker_cfg.get("marker_radius", 0.004)))
    ring_radius = max(marker_radius, float(marker_cfg.get("ring_radius", 0.006)))
    geom_group = int(marker_cfg.get("geom_group", 2))
    site_group = int(marker_cfg.get("site_group", 2))
    colors = marker_cfg.get("colors", {}) or {}
    rgba = colors.get(rope_name, marker_cfg.get(
        "rgba", [1.0, 0.85, 0.05, 1.0]))
    rgba_str = _format_float_vec(rgba, precision=3)
    base_name = f"{rope_name}_marker_{marker_idx}"
    tiny_mass = float(marker_cfg.get("mass", 1e-8))
    lines = [
        f'{indent}  <site name="{base_name}" pos="0 0 {z:.6f}" '
        f'size="{marker_radius:.6f}" rgba="{rgba_str}" group="{site_group}"/>'
    ]
    offsets = [
        (ring_radius, 0.0, 0.0),
        (0.0, ring_radius, 0.0),
        (-ring_radius, 0.0, 0.0),
        (0.0, -ring_radius, 0.0),
    ]
    for oi, (x, y, _unused) in enumerate(offsets):
        lines.append(
            f'{indent}  <geom name="{base_name}_{oi}" type="sphere" '
            f'size="{marker_radius:.6f}" pos="{x:.6f} {y:.6f} {z:.6f}" '
            f'rgba="{rgba_str}" group="{geom_group}" contype="0" '
            f'conaffinity="0" mass="{tiny_mass:.10f}"/>')
    return lines


def _quat_wxyz_from_matrix(m):
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2][1] - m[1][2]) / s
        qy = (m[0][2] - m[2][0]) / s
        qz = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        qw = (m[2][1] - m[1][2]) / s
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        qw = (m[0][2] - m[2][0]) / s
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        qw = (m[1][0] - m[0][1]) / s
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s
    q = [qw, qx, qy, qz]
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


def _camera_lookat_quat(pos, lookat, up_hint=(0.0, 0.0, 1.0)):
    pos = [float(v) for v in pos]
    lookat = [float(v) for v in lookat]
    forward = _normalize_vec([lookat[i] - pos[i] for i in range(3)])
    if forward is None:
        return None

    # MuJoCo cameras look along local -Z. Matrix columns are local X/Y/Z axes
    # expressed in world coordinates.
    z_axis = [-v for v in forward]
    x_axis = _normalize_vec(_cross(up_hint, z_axis))
    if x_axis is None:
        x_axis = _normalize_vec(_cross((0.0, 1.0, 0.0), z_axis))
    y_axis = _cross(z_axis, x_axis)
    rot = [
        [x_axis[0], y_axis[0], z_axis[0]],
        [x_axis[1], y_axis[1], z_axis[1]],
        [x_axis[2], y_axis[2], z_axis[2]],
    ]
    return _quat_wxyz_from_matrix(rot)


def _camera_orientation_attr(cam, cfg_vision):
    if cam.get("quat", None) is not None:
        return f'quat="{_format_float_vec(cam["quat"])}"'
    if cam.get("xyaxes", None) is not None:
        return f'xyaxes="{_format_float_vec(cam["xyaxes"])}"'

    pos = cam.get("pos", [0, 0, 1])
    lookat = cam.get("lookat", cfg_vision.get("camera_lookat", None))
    if lookat is not None:
        quat = _camera_lookat_quat(pos, lookat)
        if quat is not None:
            return f'quat="{_format_float_vec(quat)}"'

    return f'euler="{_format_float_vec(cam.get("euler", [0, 0, 0]))}"'


def _camera_lookat_point(cam, cfg_vision):
    lookat = cam.get("lookat", cfg_vision.get("camera_lookat", None))
    if lookat is None:
        return None
    return [float(v) for v in lookat]


def _build_visible_rgbd_camera_block(cfg_vision):
    """Return visible, non-colliding camera bodies for the MuJoCo viewer."""
    if not bool(cfg_vision.get("show_camera_models", True)):
        return ""

    cameras = list(cfg_vision.get("cameras", []))[:3]
    if not cameras:
        return ""

    lines = ["    <!-- RGB-D visible camera bodies -->"]
    geom_group = int(cfg_vision.get("camera_model_geom_group", 5))
    for idx, cam in enumerate(cameras, start=1):
        raw_name = str(cam.get("name", f"rgbd_cam_{idx}"))
        name = re.sub(r"[^A-Za-z0-9_]", "_", raw_name).strip("_") or f"rgbd_cam_{idx}"
        pos_vec = [float(v) for v in cam.get("pos", [0, 0, 1])]
        pos = _format_float_vec(pos_vec)
        orient_attr = _camera_orientation_attr(cam, cfg_vision)
        lookat = _camera_lookat_point(cam, cfg_vision)
        if lookat is not None:
            dist = math.sqrt(sum((lookat[i] - pos_vec[i]) ** 2 for i in range(3)))
        else:
            dist = 0.75
        ray_depth = max(0.25, min(1.80, float(dist)))
        fovy = math.radians(float(cam.get("fovy", 70.0)))
        aspect = float(cfg_vision.get("render_width", 640)) / max(
            1.0, float(cfg_vision.get("render_height", 480)))
        half_h = math.tan(0.25 * fovy) * ray_depth
        half_w = half_h * aspect

        # MuJoCo cameras look along local -Z. The frustum rays below make that
        # direction visible without affecting contacts or payload dynamics.
        lines.extend([
            f'    <body name="{name}_body" pos="{pos}" {orient_attr}>',
            f'      <geom name="{name}_housing" type="box" '
            f'size="0.070 0.050 0.035" rgba="0.050 0.180 0.800 1" '
            f'group="{geom_group}" contype="0" conaffinity="0"/>',
            f'      <geom name="{name}_lens" type="cylinder" '
            f'size="0.024 0.012" pos="0 0 -0.048" '
            f'rgba="0.010 0.010 0.014 1" group="{geom_group}" '
            f'contype="0" conaffinity="0"/>',
            f'      <geom name="{name}_axis" type="capsule" '
            f'fromto="0 0 -0.065 0 0 {-ray_depth:.6f}" size="0.006" '
            f'rgba="1.000 0.950 0.100 0.85" group="{geom_group}" '
            f'contype="0" conaffinity="0"/>',
            f'      <geom name="{name}_ray_tl" type="capsule" '
            f'fromto="0 0 -0.065 {half_w:.6f} {half_h:.6f} {-ray_depth:.6f}" size="0.004" '
            f'rgba="0.000 0.750 1.000 0.45" group="{geom_group}" '
            f'contype="0" conaffinity="0"/>',
            f'      <geom name="{name}_ray_tr" type="capsule" '
            f'fromto="0 0 -0.065 {half_w:.6f} {-half_h:.6f} {-ray_depth:.6f}" size="0.004" '
            f'rgba="0.000 0.750 1.000 0.45" group="{geom_group}" '
            f'contype="0" conaffinity="0"/>',
            f'      <geom name="{name}_ray_bl" type="capsule" '
            f'fromto="0 0 -0.065 {-half_w:.6f} {half_h:.6f} {-ray_depth:.6f}" size="0.004" '
            f'rgba="0.000 0.750 1.000 0.45" group="{geom_group}" '
            f'contype="0" conaffinity="0"/>',
            f'      <geom name="{name}_ray_br" type="capsule" '
            f'fromto="0 0 -0.065 {-half_w:.6f} {-half_h:.6f} {-ray_depth:.6f}" size="0.004" '
            f'rgba="0.000 0.750 1.000 0.45" group="{geom_group}" '
            f'contype="0" conaffinity="0"/>',
            f'    </body>',
        ])
    lines.append("    <!-- /RGB-D visible camera bodies -->")
    return "\n".join(lines)


def generate_single_rope_xml(
    rope_name,
    num_segments=10,
    segment_length=0.02,
    root_pos="0 0 0",
    damping=0.02,
    capsule_radius=0.004,
    segment_mass=0.01,
    marker_cfg=None,
):
    # 绳索沿 plate 局部 +Z 方向生长。
    # 当末端执行器倒立（绕X轴180°）时，局部 +Z 对应世界 -Z，
    # 绳索自然垂向地面，缩短绳末端与 prefab 的距离。
    indent_base = " " * 22
    lines = []
    sz = f"{capsule_radius:.6f}"
    marker_cfg = marker_cfg or {}
    marker_specs = _rope_marker_specs(num_segments, marker_cfg)
    marker_lookup = {}
    for spec in marker_specs:
        marker_lookup.setdefault(int(spec["segment_idx"]), []).append(spec)

    root_indent = indent_base
    lines.append(f'{root_indent}<body name="{rope_name}_root" pos="{root_pos}">')
    lines.append(f'{root_indent}  <joint type="ball" damping="{damping}"/>')
    lines.append(f'{root_indent}  <geom type="capsule"')
    lines.append(f'{root_indent}        fromto="0 0 0   0 0 {segment_length:.6f}"')
    lines.append(f'{root_indent}        size="{sz}"')
    lines.append(f'{root_indent}        mass="{segment_mass}"')
    lines.append(f'{root_indent}        material="rope"/>')
    for spec in marker_lookup.get(0, []):
        lines.extend(_rope_marker_lines(
            rope_name, 0, int(spec["marker_idx"]), root_indent,
            segment_length, marker_cfg,
            local_fraction=float(spec["local_fraction"])))
    lines.append("")

    for i in range(1, num_segments - 1):
        indent = root_indent + "  " * i
        lines.append(f'{indent}<body name="{rope_name}_{i}" pos="0 0 {segment_length:.6f}">')
        lines.append(f'{indent}  <geom type="capsule" fromto="0 0 0   0 0 {segment_length:.6f}"')
        lines.append(f'{indent}        size="{sz}" mass="{segment_mass}" material="rope"/>')
        lines.append(f'{indent}  <joint type="ball" damping="{damping}"/>')
        for spec in marker_lookup.get(i, []):
            lines.extend(_rope_marker_lines(
                rope_name, i, int(spec["marker_idx"]), indent,
                segment_length, marker_cfg,
                local_fraction=float(spec["local_fraction"])))
        lines.append("")

    last_idx = num_segments - 1
    last_indent = root_indent + "  " * last_idx
    lines.append(f'{last_indent}<body name="{rope_name}_{last_idx}" pos="0 0 {segment_length:.6f}">')
    lines.append(f'{last_indent}  <geom type="capsule" fromto="0 0 0   0 0 {segment_length:.6f}"')
    lines.append(f'{last_indent}        size="{sz}" mass="{segment_mass}" material="rope"/>')
    lines.append(f'{last_indent}  <joint type="ball" damping="{damping}"/>')
    for spec in marker_lookup.get(last_idx, []):
        lines.extend(_rope_marker_lines(
            rope_name, last_idx, int(spec["marker_idx"]), last_indent,
            segment_length, marker_cfg,
            local_fraction=float(spec["local_fraction"])))
    lines.append("")
    lines.append(f'{last_indent}  <site name="{rope_name}_end" pos="0 0 {segment_length:.6f}"')
    lines.append(f'{last_indent}        size="0.006" rgba="1 1 0 1"/>')
    lines.append(f'{last_indent}</body>')

    for i in range(last_idx - 1, 0, -1):
        close_indent = root_indent + "  " * i
        lines.append(f"{close_indent}</body>")

    lines.append(f"{root_indent}</body>")
    return "\n".join(lines)


def main(config_override=None):
    # Read all rope/payload/target parameters from the active config.  Passing a
    # config lets experiment scripts build easier comparison tasks without
    # changing DEFAULT_CONFIG or the residual-RL training setup.
    if config_override is None:
        project_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_root))
        import importlib
        import config
        importlib.reload(config)
        from config import DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG
    else:
        cfg = config_override

    cfg_rope   = cfg["rope"]
    cfg_rope_markers = cfg.get("rope_markers", {})
    cfg_prefab = cfg["prefab"]
    cfg_target = cfg["target"]
    cfg_vision = cfg.get("vision", {})
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
    marker_assets, marker_geoms = _ensure_vision_marker_assets(
        assets_dir, cfg_vision)

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
        marker_cfg=cfg_rope_markers,
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
        # Older generated demo files can contain a duplicated closing tag plus a
        # truncated keyframe comment. Clean that up before applying the usual
        # idempotent rewrites below.
        demo = re.sub(
            r"\n\s*</mujoco>\s*\n\s*l nq vector:[\s\S]*?(?=\n\s*<!-- Full nq vector:)",
            "\n",
            demo,
            count=1,
        )
        demo = re.sub(
            r'<include file="[^"]+"/>',
            '<include file="iiwa14_four_cables_with_plate.xml"/>',
            demo,
            count=1,
        )
        # `rope` is defined in included iiwa14_four_cables_with_plate.xml; avoid duplicated material names.
        demo = re.sub(r'\n\s*<material name="rope"[^>]*/>', "", demo, count=1)
        demo = re.sub(
            r"\n\s*<!-- OpenCV vision marker assets -->[\s\S]*?<!-- /OpenCV vision marker assets -->",
            "",
            demo,
            count=1,
        )
        if marker_assets:
            marker_asset_block = (
                "\n    <!-- OpenCV vision marker assets -->\n"
                + marker_assets
                + "\n    <!-- /OpenCV vision marker assets -->"
            )
            demo = demo.replace("  </asset>", marker_asset_block + "\n  </asset>", 1)

        # --- Keep three RGB-D camera definitions in the scene XML ---
        demo = re.sub(
            r"\n\s*<!-- RGB-D vision cameras -->[\s\S]*?<!-- /RGB-D vision cameras -->",
            "",
            demo,
            count=1,
        )
        demo = re.sub(
            r"\n\s*<!-- RGB-D visible camera bodies -->[\s\S]*?<!-- /RGB-D visible camera bodies -->",
            "",
            demo,
            count=1,
        )
        camera_lines = []
        for cam in cfg_vision.get("cameras", [])[:3]:
            name = cam.get("name", "rgbd_cam")
            pos = " ".join(str(float(v)) for v in cam.get("pos", [0, 0, 1]))
            orient_attr = _camera_orientation_attr(cam, cfg_vision)
            fovy = float(cam.get("fovy", 70.0))
            camera_lines.append(
                f'    <camera name="{name}" pos="{pos}" {orient_attr} '
                f'fovy="{fovy:.3f}"/>'
            )
        if camera_lines:
            camera_block = (
                "\n    <!-- RGB-D vision cameras -->\n"
                + "\n".join(camera_lines)
                + "\n    <!-- /RGB-D vision cameras -->\n"
            )
            visible_camera_block = _build_visible_rgbd_camera_block(cfg_vision)
            if visible_camera_block:
                camera_block += "\n" + visible_camera_block + "\n"
            demo = demo.replace("    <!-- target:", camera_block + "\n    <!-- target:", 1)

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
        if int(NUM_SEGMENTS) != 10:
            demo = re.sub(
                r"\n\s*<!-- Full nq vector:[\s\S]*?</keyframe>",
                "",
                demo,
                count=1,
            )
        elif "Full nq vector:" not in demo:
            keyframe_xml = _default_home_keyframe_xml(cfg, NUM_SEGMENTS)
            if keyframe_xml:
                demo = re.sub(
                    r"\n\s*</mujoco>\s*$",
                    "\n\n" + keyframe_xml + "\n\n</mujoco>\n",
                    demo,
                    count=1,
                )

        # --- Rewrite prefab body from config ---
        pshape = cfg_prefab["shape"]
        pmass = cfg_prefab["mass"]
        p_lift_z = cfg_prefab["lift_site_offset"]
        p_lift_s = cfg_prefab["lift_site_spread"]

        if pshape == "composite":
            # 多 STL 凸分解拼合体：注入 mesh asset 声明 + 多 geom body
            mesh_prefix = cfg_prefab["mesh_prefix"]
            mesh_count = cfg_prefab["mesh_count"]
            piece_mass = pmass / mesh_count

            # Keep generator idempotent: clear any previous mesh declarations
            # from earlier runs before injecting the current set.
            demo = re.sub(
                rf"\n\s*<mesh name=\"{re.escape(mesh_prefix)}_\d+\"[^>]*/>",
                "",
                demo,
            )

            # 在 <asset> 块末尾注入 mesh 声明
            mesh_assets = "\n".join(
                f'    <mesh name="{mesh_prefix}_{i}" file="{mesh_prefix}_{i}.stl"/>'
                for i in range(mesh_count)
            )
            demo = re.sub(
                r'(</asset>)',
                mesh_assets + r'\n  \1',
                demo,
                count=1,
            )

            # 生成多 geom 行
            geom_lines = "\n      ".join(
                f'<geom type="mesh" mesh="{mesh_prefix}_{i}" material="concrete" mass="{piece_mass:.6f}"/>'
                for i in range(mesh_count)
            )
            prefab_geom = geom_lines

        elif pshape == "socket":
            # 底部带方孔的方块：用原生 box 基元解析拼合，孔为真正的空洞
            import numpy as np
            shs = cfg_prefab["socket_half_size"]         # [hx, hy, hz]
            hole_sz = cfg_prefab["socket_hole_size"]     # [hole_hx, hole_hy]
            hole_d = cfg_prefab["socket_hole_depth"]
            hole_pos = cfg_prefab["socket_hole_positions"]

            hx, hy, hz = shs
            full_h = hz * 2       # 总高
            top_h = full_h - hole_d  # 顶部实心层高度
            bot_h = hole_d           # 底部开孔层高度

            # body 局部坐标：z=0 在几何中心，底面 z=-hz，顶面 z=+hz
            # 顶部实心板：z 从 (-hz + bot_h) 到 (+hz)
            top_cz = -hz + bot_h + top_h / 2
            top_half_z = top_h / 2

            # 底部开孔层：z 从 -hz 到 (-hz + bot_h)
            bot_cz = -hz + bot_h / 2
            bot_half_z = bot_h / 2

            # 收集所有孔的 x/y 边界，生成网格切分断点
            x_cuts = sorted({-hx, hx})
            y_cuts = sorted({-hy, hy})
            for (px, py) in hole_pos:
                x_cuts.extend([px - hole_sz[0]/2, px + hole_sz[0]/2])
                y_cuts.extend([py - hole_sz[1]/2, py + hole_sz[1]/2])
            x_cuts = sorted(set(x_cuts))
            y_cuts = sorted(set(y_cuts))

            # 判断一个网格单元是否落在某个孔内
            def in_hole(cx, cy):
                for (px, py) in hole_pos:
                    if (px - hole_sz[0]/2 - 1e-9 <= cx <= px + hole_sz[0]/2 + 1e-9 and
                        py - hole_sz[1]/2 - 1e-9 <= cy <= py + hole_sz[1]/2 + 1e-9):
                        return True
                return False

            geoms = []
            n_geom = 0
            # 顶部实心板（1 个 geom）
            geoms.append(
                f'<geom name="socket_top" type="box" '
                f'size="{hx} {hy} {top_half_z:.6f}" pos="0 0 {top_cz:.6f}" '
                f'material="concrete"/>'
            )
            n_geom += 1
            # 底部网格：跳过孔所在的单元
            for ix in range(len(x_cuts) - 1):
                for iy in range(len(y_cuts) - 1):
                    x_lo, x_hi = x_cuts[ix], x_cuts[ix + 1]
                    y_lo, y_hi = y_cuts[iy], y_cuts[iy + 1]
                    cx = (x_lo + x_hi) / 2
                    cy = (y_lo + y_hi) / 2
                    if in_hole(cx, cy):
                        continue
                    cell_hx = (x_hi - x_lo) / 2
                    cell_hy = (y_hi - y_lo) / 2
                    if cell_hx < 1e-9 or cell_hy < 1e-9:
                        continue
                    geoms.append(
                        f'<geom name="socket_b{n_geom}" type="box" '
                        f'size="{cell_hx:.6f} {cell_hy:.6f} {bot_half_z:.6f}" '
                        f'pos="{cx:.6f} {cy:.6f} {bot_cz:.6f}" '
                        f'material="concrete"/>'
                    )
                    n_geom += 1

            piece_mass = pmass / n_geom
            # 给每个 geom 补上质量
            geoms = [g.replace('material="concrete"',
                               f'material="concrete" mass="{piece_mass:.6f}"')
                     for g in geoms]
            prefab_geom = "\n      ".join(geoms)

        elif pshape == "box":
            bhs = cfg_prefab["box_half_size"]
            prefab_geom = f'<geom type="box" size="{bhs[0]} {bhs[1]} {bhs[2]}" material="concrete" mass="{pmass}"/>'
        elif pshape == "cylinder":
            cr = cfg_prefab["cylinder_radius"]
            ch = cfg_prefab["cylinder_half_height"]
            prefab_geom = f'<geom type="cylinder" size="{cr} {ch}" material="concrete" mass="{pmass}"/>'
        else:  # composite (mesh STL)
            pass  # already handled above

        prefab_body = f"""<body name="prefab" pos="0.3 0.15 0.5">
      <joint type="free"/>
      {prefab_geom}
{marker_geoms}

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
        target_mode = cfg_target.get("mode", "visual")

        if target_mode == "rebar":
            # 钢筋桩模式：按 rebar_positions 生成多根有碰撞的钢筋
            rr = cfg_target["rebar_radius"]
            rh = cfg_target["rebar_half_height"]
            r_rgba = cfg_target["rebar_rgba"]
            r_rgba_str = f"{r_rgba[0]} {r_rgba[1]} {r_rgba[2]} {r_rgba[3]}"
            positions = cfg_target["rebar_positions"]

            rebar_geoms = "\n      ".join(
                f'<geom name="rebar_{i}" type="cylinder" size="{rr} {rh}" '
                f'pos="{px} {py} {rh}" rgba="{r_rgba_str}" contype="1" conaffinity="1"/>'
                for i, (px, py) in enumerate(positions)
            )
            target_body = f"""<body name="target" pos="0.3 0.25 0">
      {rebar_geoms}
    </body>"""
        else:
            # visual 模式：纯视觉标记，无碰撞
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
            target_body = f"""<body name="target" pos="0.3 0.25 0">
      {target_geom}
    </body>"""
        demo = re.sub(
            r'<body name="target"[\s\S]*?</body>',
            target_body,
            demo,
            count=1,
        )

        # Remove old mesh references if not in composite mode
        if pshape != "composite":
            demo = re.sub(r'\s*<mesh name="\w+_convex_\d+"[^/]*/>', '', demo)
        demo = re.sub(r'\s*<material name="steel"[^/]*/>', '', demo)
        demo = re.sub(r'\s*<material name="red"[^/]*/>', '', demo)

        demo_xml.write_text(demo, encoding="utf-8")
        print(f"Updated: {demo_xml}")
    total_m = NUM_SEGMENTS * SEGMENT_LENGTH_M
    print(f"Generated: {output_xml}")
    print(f"  segments={NUM_SEGMENTS}, segment_length={SEGMENT_LENGTH_M} m -> total chain length {total_m:.4f} m")


if __name__ == "__main__":
    main()
