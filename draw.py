"""Generates the artwork for this entry: a baby panda sitting in a
teacup, paws on the rim, steam curling up, a tiny flower on one ear.

Deterministic (fixed seed 63): running `python3 draw.py` always produces
the same art.png. Needs Pillow + numpy (only for generating; the entry
itself, main.py, is standard library only and ships the finished image
as base64 in art.b64).
"""
import base64
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SS = 2  # supersampling factor: draw at 2048, downscale to 1024
W = H = 1024
HERE = Path(__file__).resolve().parent


def vgrad(w, h, stops):
    """Vertical gradient. stops = [(pos0to1, (r,g,b)), ...]."""
    ys = np.linspace(0, 1, h * SS)
    cols = np.zeros((h * SS, 3), dtype=np.float32)
    pos = np.array([s[0] for s in stops])
    arr = np.array([s[1] for s in stops], dtype=np.float32)
    for ch in range(3):
        cols[:, ch] = np.interp(ys, pos, arr[:, ch])
    img = np.repeat(cols[:, None, :], w * SS, axis=1)
    return Image.fromarray(img.astype(np.uint8), "RGB")


def radial_glow(w, h, cx, cy, radius, color, peak=255):
    yy, xx = np.mgrid[0:h * SS, 0:w * SS]
    d = np.sqrt((xx - cx * SS) ** 2 + (yy - cy * SS) ** 2) / (radius * SS)
    a = np.clip(1 - d, 0, 1) ** 2 * peak
    layer = np.zeros((h * SS, w * SS, 4), dtype=np.uint8)
    layer[..., 0], layer[..., 1], layer[..., 2] = color
    layer[..., 3] = a.astype(np.uint8)
    return Image.fromarray(layer, "RGBA")


def catmull_rom(points, samples=24, closed=True):
    """Smooth closed/open spline through points. Returns list of (x,y)."""
    pts = list(points)
    if closed:
        pts = [pts[-1]] + pts + [pts[0], pts[1]]
    else:
        pts = [pts[0]] + pts + [pts[-1]]
    out = []
    for i in range(len(pts) - 3):
        p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        for j in range(samples):
            t = j / samples
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    if not closed:
        out.append(pts[-2])
    return out


def poly(draw, points, **kw):
    draw.polygon([(x * SS, y * SS) for x, y in points], **kw)


def ell(draw, cx, cy, rx, ry, **kw):
    draw.ellipse([cx * SS - rx * SS, cy * SS - ry * SS, cx * SS + rx * SS, cy * SS + ry * SS], **kw)


def soft_shadow(base, cx, cy, rx, ry, alpha=110, blur=24):
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse([cx * SS - rx * SS, cy * SS - ry * SS, cx * SS + rx * SS, cy * SS + ry * SS],
              fill=(0, 0, 0, alpha))
    layer = layer.filter(ImageFilter.GaussianBlur(blur * SS))
    base.alpha_composite(layer)


def vignette(img, strength=46):
    yy, xx = np.mgrid[0:H * SS, 0:W * SS]
    dx = (xx - W * SS / 2) / (W * SS / 2)
    dy = (yy - H * SS / 2) / (H * SS / 2)
    d = np.clip(np.sqrt(dx ** 2 + dy ** 2) - 0.66, 0, 1) ** 2 * strength
    layer = np.zeros((H * SS, W * SS, 4), dtype=np.uint8)
    layer[..., 3] = d.astype(np.uint8)
    img.alpha_composite(Image.fromarray(layer, "RGBA"))
    return img


def big_eye(d, cx, cy, rx=38, ry=44, iris=(58, 36, 28), iris_rx=22, iris_ry=26):
    """Big glossy cute eye: white, dark iris, two highlights."""
    ell(d, cx, cy, rx, ry, fill=(255, 255, 255, 255))
    ell(d, cx, cy + 2, iris_rx, iris_ry, fill=iris + (255,))
    ell(d, cx - iris_rx * 0.35, cy - iris_ry * 0.4, iris_rx * 0.42, iris_ry * 0.42,
        fill=(255, 255, 255, 255))
    ell(d, cx + iris_rx * 0.4, cy + iris_ry * 0.45, iris_rx * 0.2, iris_ry * 0.2,
        fill=(255, 255, 255, 230))


def blush(d, cx, cy, rx=34, ry=20, alpha=120):
    ell(d, cx, cy, rx, ry, fill=(244, 130, 150, alpha))


def draw_art():
    # Seed 63: fixed so every run draws the same panda.
    rng = random.Random(63)
    img = vgrad(W, H, [(0.0, (214, 240, 228)), (0.55, (232, 244, 228)), (1.0, (250, 240, 214))]).convert("RGBA")
    # polka dots
    dl = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dl)
    for gy in range(90, 1000, 130):
        for gx in range(70, 1000, 130):
            ell(dd, gx + (65 if (gy // 130) % 2 else 0), gy, 16, 16, fill=(255, 255, 255, 60))
    img.alpha_composite(dl)
    img.alpha_composite(radial_glow(W, H, 512, 480, 400, (255, 255, 255), 70))
    # saucer + cup (behind panda head, drawn first)
    cup = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    cd = ImageDraw.Draw(cup)
    soft_shadow(cup, 512, 920, 260, 36, alpha=70, blur=26)
    ell(cd, 512, 892, 235, 36, fill=(176, 208, 228, 255))
    ell(cd, 512, 884, 235, 32, fill=(198, 224, 238, 255))
    # handle (drawn BEFORE the body so its ends tuck behind the cup)
    cd.arc([600 * SS, 685 * SS, 730 * SS, 815 * SS], 295, 65, fill=(168, 204, 230, 255), width=15 * SS)
    # cup body
    body = catmull_rom([(350, 640), (370, 830), (420, 872), (604, 872), (654, 830), (674, 640),
                        (640, 668), (384, 668)], closed=True)
    poly(cd, body, fill=(168, 204, 230, 255))
    # cup rim
    ell(cd, 512, 652, 164, 30, fill=(198, 224, 238, 255))
    ell(cd, 512, 648, 150, 24, fill=(120, 90, 70, 255))  # tea
    ell(cd, 512, 646, 138, 20, fill=(146, 106, 82, 255))
    # flower pattern on cup
    for (fx, fy) in [(450, 740), (560, 780), (500, 830)]:
        for ang in range(0, 360, 72):
            px = fx + math.cos(math.radians(ang)) * 11
            py = fy + math.sin(math.radians(ang)) * 11
            ell(cd, px, py, 7, 7, fill=(255, 255, 255, 220))
        ell(cd, fx, fy, 6, 6, fill=(255, 205, 90, 255))
    img.alpha_composite(cup)
    # panda head + paws
    pl = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    pd = ImageDraw.Draw(pl)
    # ears
    ell(pd, 352, 332, 56, 56, fill=(40, 40, 52, 255))
    ell(pd, 672, 332, 56, 56, fill=(40, 40, 52, 255))
    # head
    ell(pd, 512, 480, 192, 182, fill=(255, 255, 255, 255))
    # eye patches (rotated ellipses)
    def patch(cx, cy, ang):
        pts = []
        for t in range(0, 360, 10):
            a = math.radians(t)
            x = 44 * math.cos(a)
            y = 54 * math.sin(a)
            xr = x * math.cos(math.radians(ang)) - y * math.sin(math.radians(ang))
            yr = x * math.sin(math.radians(ang)) + y * math.cos(math.radians(ang))
            pts.append((cx + xr, cy + yr))
        poly(pd, pts, fill=(40, 40, 52, 255))
    patch(438, 468, -18)
    patch(586, 468, 18)
    # eyes inside patches
    big_eye(pd, 442, 472, rx=26, ry=30, iris=(30, 30, 40), iris_rx=14, iris_ry=16)
    big_eye(pd, 582, 472, rx=26, ry=30, iris=(30, 30, 40), iris_rx=14, iris_ry=16)
    # nose + mouth
    ell(pd, 512, 540, 16, 11, fill=(40, 40, 52, 255))
    pd.arc([494 * SS, 548 * SS, 512 * SS, 570 * SS], 10, 170, fill=(90, 90, 110, 255), width=3 * SS)
    pd.arc([512 * SS, 548 * SS, 530 * SS, 570 * SS], 10, 170, fill=(90, 90, 110, 255), width=3 * SS)
    blush(pd, 392, 556, 30, 18, 110)
    blush(pd, 632, 556, 30, 18, 110)
    # paws gripping the rim
    ell(pd, 392, 668, 46, 30, fill=(255, 255, 255, 255))
    ell(pd, 632, 668, 46, 30, fill=(255, 255, 255, 255))
    # tiny flower on right ear
    for ang in range(0, 360, 72):
        px = 672 + math.cos(math.radians(ang)) * 12
        py = 300 + math.sin(math.radians(ang)) * 12
        ell(pd, px, py, 8, 8, fill=(255, 255, 255, 255))
    ell(pd, 672, 300, 7, 7, fill=(255, 205, 90, 255))
    img.alpha_composite(pl)
    # steam curls above the head
    st = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    std = ImageDraw.Draw(st)
    for x0, amp in [(440, 26), (580, 22)]:
        pts = []
        for y in range(300, 130, -8):
            pts.append((x0 + math.sin((300 - y) / 34.0) * amp, y))
        for i in range(len(pts) - 1):
            std.line([pts[i][0] * SS, pts[i][1] * SS, pts[i + 1][0] * SS, pts[i + 1][1] * SS],
                     fill=(255, 255, 255, 120), width=7 * SS)
    img.alpha_composite(st.filter(ImageFilter.GaussianBlur(6 * SS)))
    vignette(img)
    return img.convert("RGB").resize((1024, 1024), Image.LANCZOS)


if __name__ == "__main__":
    art = draw_art()
    art.save(HERE / "art.png", format="PNG", optimize=True)
    (HERE / "art.b64").write_text(
        base64.b64encode((HERE / "art.png").read_bytes()).decode() + "\n")
    print(f"wrote {HERE / 'art.png'} ({(HERE / 'art.png').stat().st_size} bytes)")
