#!/usr/bin/env python3
"""Build the complete paper Figure 1 from retained simulation captures.

The figure follows the supplied sketch: a successful insertion trajectory on
top and three mechanism panels below.  All geometry is drawn with Pillow so
the output is deterministic and exportable as both PNG and PDF.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import math

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "test_results" / "figure_assets"
STRIP = ASSET / "pipeline_paper_direction_success_search_20260725_v6" / "figure1_success_shadow_horizontal_strip_natural_rebar_v14.png"
SUCCESS = ASSET / "pipeline_paper_direction_success_search_20260725_v6" / "success_search_ep03_step0215_descent_success_paper_side_rgb.png"
ROPE = ASSET / "pipeline_scene_multiview_20260725_v3" / "arm_visible_ep01_step0150_descent_scene_front_rgb.png"
OUT = ROOT / "outputs" / "figure1_complete"
CLOSEUPS = OUT / "closeups"
PNG = OUT / "figure1_complete.png"
PDF = OUT / "figure1_complete.pdf"

W, H = 3200, 1840
NAVY = (18, 45, 72)
BLUE = (25, 103, 180)
CYAN = (28, 157, 174)
ORANGE = (239, 125, 45)
RED = (211, 70, 65)
MUTED = (91, 105, 118)

def font(size, bold=False):
    name = "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
    return ImageFont.truetype(name, size)

def arrow(d, p0, p1, color=BLUE, width=7, head=22):
    d.line([p0, p1], fill=color, width=width)
    a = math.atan2(p1[1]-p0[1], p1[0]-p0[0])
    q1 = (p1[0]-head*math.cos(a-math.pi/6), p1[1]-head*math.sin(a-math.pi/6))
    q2 = (p1[0]-head*math.cos(a+math.pi/6), p1[1]-head*math.sin(a+math.pi/6))
    d.polygon([p1, q1, q2], fill=color)

def panel_base(canvas, box, label, title, subtitle):
    d = ImageDraw.Draw(canvas)
    x, y, w, h = box
    d.rounded_rectangle((x, y, x+w, y+h), radius=14, fill=(248,250,252), outline=(189,201,211), width=4)
    d.text((x+30, y+20), title, font=font(34, True), fill=NAVY)
    if subtitle:
        d.text((x+30, y+60), subtitle, font=font(21), fill=MUTED)
    return (x+28, y+88, w-56, h-113)

def fit(im, size):
    im = im.convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", size, (230,234,238))
    out.paste(im, ((size[0]-im.width)//2, (size[1]-im.height)//2))
    return out

def crop_fit(im, crop, size):
    """Crop in source pixels and fill the destination without letterboxing."""
    return ImageOps.fit(
        im.convert("RGB").crop(crop), size,
        method=Image.Resampling.LANCZOS, centering=(0.5, 0.52))

def draw():
    if not STRIP.exists():
        raise FileNotFoundError(STRIP)
    canvas = Image.new("RGB", (W,H), "white")
    strip = Image.open(STRIP).convert("RGB").resize((W, 720), Image.Resampling.LANCZOS)
    canvas.paste(strip, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 720, W, 760), fill="white")

    boxes = [(55, 850, 980, 890), (1110, 850, 980, 890), (2165, 850, 980, 890)]
    a = panel_base(canvas, boxes[0], "", "High-precision insertion", "")
    b = panel_base(canvas, boxes[1], "", "Wind disturbance", "")
    c = panel_base(canvas, boxes[2], "", "Deformable rope", "")

    # A: real final state with a precision target overlay.
    ax, ay, aw, ah = a
    source_a = CLOSEUPS / "insertion_closeup.png"
    im = crop_fit(Image.open(source_a if source_a.exists() else SUCCESS), (110, 80, 565, 370), (aw, ah-20))
    canvas.paste(im, (ax, ay+15))

    # B: scene frame plus a clean wind vector field.
    bx, by, bw, bh = b
    wind_frames = [CLOSEUPS / f"wind_gust_closeup_{i}.png" for i in (1, 2, 3)]
    if all(p.exists() for p in wind_frames):
        # Same viewpoint, overlaid frames: the payload positions form a swing silhouette.
        crop = (30, 120, 420, 455)
        layers = [crop_fit(Image.open(path), crop, (bw, bh-20)).convert("L") for path in wind_frames]
        composite = Image.blend(layers[0], layers[1], 0.42)
        composite = Image.blend(composite, layers[2], 0.42)
        composite = ImageEnhance.Contrast(composite).enhance(1.45).convert("RGB")
        canvas.paste(composite, (bx, by+5))
    else:
        source_b = CLOSEUPS / "wind_gust_closeup.png"
        bim = fit(Image.open(source_b if source_b.exists() else ROPE), (bw, bh-20))
        canvas.paste(bim, (bx, by+5))

    # C: real end-effector excitation frame with a measured-style trajectory cue.
    cx, cy, cw, ch = c
    source_c = CLOSEUPS / "rope_excitation_closeup.png"
    # Crop out the robot; only cable attachment, rope curvature, and payload remain.
    c_im = crop_fit(Image.open(source_c if source_c.exists() else ROPE), (100, 95, 420, 400), (cw, ch-20))
    canvas.paste(c_im, (cx, cy+5))

    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(PNG, optimize=True)
    canvas.save(PDF, "PDF", resolution=300.0)
    print(f"wrote {PNG}\nwrote {PDF}")

if __name__ == "__main__":
    draw()
