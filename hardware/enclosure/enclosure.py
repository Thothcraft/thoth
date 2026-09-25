"""Thoth node enclosure — parametric 3D-printable case generator.

Two independent options → 4 printable variants:

    TOP     "radar"  (DreamHAT+ BGT60TR13C, integral 60 GHz membrane window)
            "sense"  (Sense HAT bezel — LED matrix + joystick exposed)
    BOTTOM  "slim"   (bare Pi)
            "battery" (PiSugar 3 Plus 5000 mAh module under the Pi)

Build all STLs:

    pip install trimesh manifold3d shapely
    python enclosure.py            # writes stl/*.stl
    python enclosure.py --list     # show parts

A mirrored OpenSCAD source (enclosure.scad) ships alongside for
GUI tweaking — same parameter names.

All dimensions in mm, Z-up, origin at the enclosure's outer
bottom-west-south corner. Numbers marked VERIFY in DIMENSIONS.md come
from datasheet/nominal values and should be measured on your boards
before a first print.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh
from shapely.geometry import Polygon

# ---------------------------------------------------------------------------
# Parameters — everything a maker might want to touch lives here.
# ---------------------------------------------------------------------------


@dataclass
class Params:
    # -- measured board geometry ------------------------------------------
    pcb_w: float = 85.0            # RPi 3B+/4B/5B — all identical
    pcb_d: float = 56.0
    pcb_t: float = 1.6
    hole_inset: tuple = (3.5, 3.5)          # first M2.5 hole from W/S edges
    hole_span: tuple = (58.0, 49.0)         # mounting-hole grid
    # -- stack heights -----------------------------------------------------
    under_board_slim: float = 4.0           # clearance under Pi PCB
    under_board_battery: float = 20.0       # PiSugar 3 Plus ~15 + frame ~4 + margin (VERIFY)
    hat_gap: float = 20.0                   # Pi top → HAT bottom (covers active cooler + tall header)
    hat_t: float = 1.6
    radar_head: float = 8.0                 # radar HAT top side clearance
    sense_head: float = 13.0                # Sense HAT joystick ≈ 10 mm + margin
    # -- shell --------------------------------------------------------------
    wall: float = 2.2
    floor: float = 2.0
    lid_t: float = 2.0
    lid_skirt: float = 5.0                  # cap overlap down over tray wall tops
    clearance: float = 0.35                 # board↔wall and lid↔tray clearance
    corner_r: float = 3.0                   # outer corner radius
    fillet_r: float = 4.2                   # inner corner tower (screw post)
    screw_pilot: float = 2.1                # M2.5 self-tap pilot Ø
    screw_clear: float = 3.0                # lid screw clearance Ø
    post_r: float = 3.0                     # Pi support posts
    # -- retention clips ----------------------------------------------------
    clip_w: float = 6.0
    clip_t: float = 1.0
    clip_h: float = 3.4                     # finger height above wall inner face base
    clip_hook: float = 0.7                  # hook depth over board edge
    # -- port canyons (cut into wall top edge) -------------------------------
    west_window: tuple = (10.0, 46.0)       # y-range opened on west wall
    east_window: tuple = (10.0, 46.0)       # y-range opened on east wall
    north_window: tuple = (30.0, 60.0)      # x-range for CSI/DSI ribbons
    sd_x: float = 39.0                      # SD slot centre from west edge (Pi4/5) (VERIFY)
    sd_w: float = 22.0
    sd_h: float = 4.0                       # slot height below board top
    # -- battery well ports (deep bottom only) -------------------------------
    ps_west_window: tuple = (14.0, 44.0)    # y-range, PiSugar USB-C/µUSB (VERIFY)
    ps_window_h: float = 8.0                # window height above well floor
    ps_button_y: float = 28.0               # power button, south wall (VERIFY)
    ps_button_r: float = 2.5
    # -- radar lid ------------------------------------------------------------
    hat_offset: tuple = (10.0, 0.0)         # HAT board SW corner on Pi board (VERIFY)
    radar_ant_xy: tuple = (32.5, 20.0)      # antenna centre on DreamHAT+ (MEASURE!)
    radar_aperture: tuple = (16.0, 22.0)    # opening W×H at inner face — covers 40°×65° FOV
    radar_membrane: float = 1.0             # 0 → fully open aperture
    # -- sense lid -------------------------------------------------------------
    sense_matrix_xy: tuple = (28.0, 38.0)   # LED matrix centre on HAT (MEASURE!)
    sense_matrix_wh: tuple = (34.0, 34.0)   # 8×8 LED matrix opening
    sense_joy_xy: tuple = (40.0, 14.0)      # joystick centre (MEASURE!)
    sense_joy_r: float = 6.0                # joystick opening radius
    # -- vents -----------------------------------------------------------------
    vent_rows: int = 3
    vent_cols: int = 6
    vent_slot: tuple = (6.0, 2.4)           # w×h of one vent slot
    vent_pitch: tuple = (10.0, 6.0)
    vent_wall_z: float = 0.55               # fraction of wall height where vent row sits
    lid_vent_grid: tuple = (4, 3)           # cols×rows of Ø4 holes in lid plate (east half)
    lid_vent_r: float = 2.2
    lid_vent_pitch: float = 9.0

    # -- derived ---------------------------------------------------------------
    @property
    def inner_w(self) -> float:
        return self.pcb_w + 2 * self.clearance

    @property
    def inner_d(self) -> float:
        return self.pcb_d + 2 * self.clearance

    @property
    def outer_w(self) -> float:
        return self.inner_w + 2 * self.wall

    @property
    def outer_d(self) -> float:
        return self.inner_d + 2 * self.wall

    def under_board(self, battery: bool) -> float:
        return self.under_board_battery if battery else self.under_board_slim

    def head_room(self, top: str) -> float:
        return self.radar_head if top == "radar" else self.sense_head


# ---------------------------------------------------------------------------
# CSG helpers (trimesh + manifold3d backend)
# ---------------------------------------------------------------------------


def _rounded_poly(w: float, d: float, r: float, n: int = 8) -> Polygon:
    """2D rounded rectangle centred at origin."""
    pts = []
    for cx, cy, a0 in ((w / 2 - r, d / 2 - r, 0),
                       (-w / 2 + r, d / 2 - r, 90),
                       (-w / 2 + r, -d / 2 + r, 180),
                       (w / 2 - r, -d / 2 + r, 270)):
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return Polygon(pts)


def rbox(w, d, h, r=0.0):
    """Rounded box, SW corner at (0,0), base at z=0."""
    if r <= 0.01:
        m = trimesh.creation.box((w, d, h))
        m.apply_translation([w / 2, d / 2, h / 2])
        return m
    m = trimesh.creation.extrude_polygon(_rounded_poly(w, d, r), h)
    m.apply_translation([w / 2, d / 2, 0])
    return m


def box(sx, sy, sz, x=0, y=0, z=0, centre_xy=False):
    m = trimesh.creation.box((sx, sy, sz))
    if centre_xy:
        m.apply_translation([x, y, z + sz / 2])
    else:
        m.apply_translation([x + sx / 2, y + sy / 2, z + sz / 2])
    return m


def cyl(r, h, x=0, y=0, z=0, sections=48):
    m = trimesh.creation.cylinder(radius=r, height=h, sections=sections)
    m.apply_translation([x, y, z + h / 2])
    return m


def union_all(meshes):
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.boolean.union(meshes)


def subtract(base, cutters):
    out = base
    for c in cutters:
        out = out.difference(c)
    return out


# ---------------------------------------------------------------------------
# Shared board-position helpers (cavity coordinates, SW inner corner = 0,0)
# ---------------------------------------------------------------------------


def _pi_holes(p: Params):
    ix, iy = p.hole_inset
    sx, sy = p.hole_span
    c = p.clearance
    return [(c + ix, c + iy), (c + ix, c + iy + sy),
            (c + ix + sx, c + iy), (c + ix + sx, c + iy + sy)]


def _hat_xy(p: Params, hat_xy):
    """Map a point in HAT-board coords → cavity XY (HAT SW corner offset)."""
    ox, oy = p.hat_offset
    return (p.clearance + ox + hat_xy[0], p.clearance + oy + hat_xy[1])


# ---------------------------------------------------------------------------
# Tray (bottom shell) — one model, parametric depth
# ---------------------------------------------------------------------------


def build_tray(p: Params, battery: bool) -> trimesh.Trimesh:
    ub = p.under_board(battery)
    head = max(p.radar_head, p.sense_head)   # walls fit the taller (sense) stack
    stack_top = p.floor + ub + p.pcb_t + p.hat_gap + p.hat_t + head
    wall_h = stack_top - p.floor
    iw, id_ = p.inner_w, p.inner_d
    ow, od = p.outer_w, p.outer_d

    # outer body minus inner cavity
    body = rbox(ow, od, stack_top, p.corner_r)
    cavity = box(iw, id_, wall_h + 20, x=p.wall, y=p.wall, z=p.floor)
    # battery well is shallower-only when slim: cavity already to floor; for
    # slim we want floor thickness preserved — cavity stops at z=floor → ok.
    body = body.difference(cavity)

    # inner corner fillet towers (lid screw posts), full height, pilot holes
    cutters = []
    towers = []
    for cx, cy in ((0, 0), (iw, 0), (0, id_), (iw, id_)):
        # tower centre tangent to inner cavity corner
        tx = p.wall + (p.fillet_r * 0.55 if cx == 0 else iw - p.fillet_r * 0.55)
        ty = p.wall + (p.fillet_r * 0.55 if cy == 0 else id_ - p.fillet_r * 0.55)
        towers.append(cyl(p.fillet_r, stack_top, tx, ty, 0))
        cutters.append(cyl(p.screw_pilot / 2, 14, tx, ty,
                           stack_top - 14, sections=32))
    body = union_all([body] + towers)

    # Pi support posts (top at board bottom). Skipped in the battery
    # variant — the PiSugar's own threaded standoffs carry the Pi at the
    # same XY, so tall posts there would collide with the module frame.
    if not battery:
        posts = [cyl(p.post_r, p.floor + ub, p.wall + hx, p.wall + hy)
                 for hx, hy in _pi_holes(p)]
        body = union_all([body] + posts)
        # optional M2.5 pilots in the posts
        for hx, hy in _pi_holes(p):
            cutters.append(cyl(p.screw_pilot / 2, 6, p.wall + hx,
                               p.wall + hy, p.floor + ub - 6,
                               sections=32))

    # retention clips on north wall inner face — cantilever fingers whose
    # tips hook over the Pi top edge.
    board_bot = p.floor + ub
    for fx in (iw * 0.30, iw * 0.70):
        cx = p.wall + fx
        finger = box(p.clip_w, p.clip_t, p.clip_h + p.pcb_t,
                     x=cx - p.clip_w / 2,
                     y=p.wall + id_ - p.clip_t,
                     z=board_bot - p.clip_h)
        hook = box(p.clip_w, p.clip_hook, p.pcb_t + 0.4,
                   x=cx - p.clip_w / 2,
                   y=p.wall + id_ - p.clip_t - p.clip_hook,
                   z=board_bot - 0.4)
        body = union_all([body, finger, hook])

    # -- wall-top canyons --------------------------------------------------
    # West + east: full port window from board bottom to wall top.
    win_h = stack_top - board_bot - 1.0
    wy0, wy1 = p.west_window
    cutters.append(box(p.wall + 0.4, wy1 - wy0, win_h,
                       x=-0.2, y=p.wall + wy0, z=board_bot + 0.5))
    ey0, ey1 = p.east_window
    cutters.append(box(p.wall + 0.4, ey1 - ey0, win_h,
                       x=ow - p.wall - 0.2, y=p.wall + ey0, z=board_bot + 0.5))
    # North: CSI/DSI ribbon window — shallower, top of wall down to hat level
    nx0, nx1 = p.north_window
    cutters.append(box(nx1 - nx0, p.wall + 0.4, 12,
                       x=p.wall + nx0, y=p.wall + id_ - 0.2,
                       z=stack_top - 12))
    # South: SD-card slot straddling the board plane
    cutters.append(box(p.sd_w, p.wall + 0.4, p.sd_h + 2.5,
                       x=p.wall + p.clearance + p.sd_x - p.sd_w / 2,
                       y=-0.2, z=board_bot - 1.0))

    # -- vents in south wall -------------------------------------------------
    vz = p.floor + (stack_top - p.floor) * p.vent_wall_z
    vw, vh = p.vent_slot
    vp, vpy = p.vent_pitch
    vx0 = ow / 2 - (p.vent_cols - 1) * vp / 2
    for i in range(p.vent_cols):
        for j in range(p.vent_rows):
            cutters.append(box(vw, p.wall + 0.6, vh,
                               x=vx0 + i * vp - vw / 2,
                               y=-0.3,
                               z=vz + j * vpy))

    # -- battery well extras ------------------------------------------------
    if battery:
        py0, py1 = p.ps_west_window
        cutters.append(box(p.wall + 0.4, py1 - py0, p.ps_window_h,
                           x=-0.2, y=p.wall + py0,
                           z=p.floor + 3.0))
        # power-button hole through the south wall of the well
        btn = trimesh.creation.cylinder(
            radius=p.ps_button_r, height=p.wall + 0.6, sections=24)
        btn.apply_transform(
            trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
        btn.apply_translation(
            [p.wall + p.clearance + p.ps_button_y, -0.2,
             p.floor + 3.0 + p.ps_window_h / 2])
        cutters.append(btn)

    body = subtract(body, cutters)
    return body


# ---------------------------------------------------------------------------
# Lid — cap style plate, two tops
# ---------------------------------------------------------------------------


def _lid_base(p: Params):
    """Cap-style lid in a LOCAL frame: plate base (seam) at z=0, plate
    occupying [0, lid_t], skirt wrapping DOWN over the tray wall tops for
    `lid_skirt` mm. XY matches the tray's outer box.

    Returns (lid_mesh, cutters) — callers subtract and export.
    """
    ow, od = p.outer_w, p.outer_d
    iw, id_ = p.inner_w, p.inner_d
    skirt_t = 1.6
    inset = p.clearance + skirt_t     # skirt sticks this far past tray outer

    # plate covers the skirt outline (flush cap)
    plate = rbox(ow + 2 * inset, od + 2 * inset, p.lid_t,
                 p.corner_r + inset)
    plate.apply_translation([-inset, -inset, 0])
    # skirt wraps the OUTSIDE of the tray wall top: inner = outer + 2*clr
    sk_inner = rbox(ow + 2 * p.clearance, od + 2 * p.clearance,
                    p.lid_skirt, p.corner_r + p.clearance)
    sk_inner.apply_translation([-p.clearance, -p.clearance, 0])
    sk_outer = rbox(ow + 2 * inset, od + 2 * inset,
                    p.lid_skirt, p.corner_r + inset)
    sk_outer.apply_translation([-inset, -inset, 0])
    skirt = sk_outer.difference(sk_inner)
    skirt.apply_translation([0, 0, -p.lid_skirt])
    lid = union_all([plate, skirt])

    cutters = []
    # corner screw clearance holes through the plate (match tray towers)
    for cx, cy in ((0, 0), (iw, 0), (0, id_), (iw, id_)):
        tx = p.wall + (p.fillet_r * 0.55 if cx == 0 else iw - p.fillet_r * 0.55)
        ty = p.wall + (p.fillet_r * 0.55 if cy == 0 else id_ - p.fillet_r * 0.55)
        cutters.append(cyl(p.screw_clear / 2, p.lid_t + 0.4, tx, ty,
                           -0.2, sections=32))
    # vent grid on the east half of the plate
    cols, rows = p.lid_vent_grid
    gx0 = ow * 0.70
    gy0 = od / 2 - (rows - 1) * p.lid_vent_pitch / 2
    for i in range(cols):
        for j in range(rows):
            cutters.append(cyl(p.lid_vent_r, p.lid_t + 0.4,
                               gx0 + i * p.lid_vent_pitch,
                               gy0 + j * p.lid_vent_pitch,
                               -0.2, sections=24))
    return lid, cutters


def build_lid_radar(p: Params) -> trimesh.Trimesh:
    lid, cutters = _lid_base(p)
    ax, ay = _hat_xy(p, p.radar_ant_xy)
    aw, ah = p.radar_aperture

    if p.radar_membrane > 0.05:
        # aperture = through-pocket in the upper (lid_t − membrane) layer,
        # leaving a `membrane`-thick skin as the 60 GHz radome; a wider
        # shallow pocket below eases placement tolerance over the antenna.
        pocket_h = p.lid_t - p.radar_membrane
        cutters.append(box(aw, ah, pocket_h + 0.2,
                           x=ax - aw / 2, y=ay - ah / 2,
                           z=p.radar_membrane - 0.1))
        cutters.append(box(aw + 2.0, ah + 2.0, 1.2,
                           x=ax - aw / 2 - 1.0, y=ay - ah / 2 - 1.0,
                           z=p.radar_membrane - 1.2))
    else:
        cutters.append(box(aw, ah, p.lid_t + 0.4,
                           x=ax - aw / 2, y=ay - ah / 2, z=-0.2))
    return subtract(lid, cutters)


def build_lid_sense(p: Params) -> trimesh.Trimesh:
    lid, cutters = _lid_base(p)
    mx, my = _hat_xy(p, p.sense_matrix_xy)
    mw, mh = p.sense_matrix_wh
    cutters.append(box(mw, mh, p.lid_t + 0.4,
                       x=mx - mw / 2, y=my - mh / 2, z=-0.2))
    jx, jy = _hat_xy(p, p.sense_joy_xy)
    cutters.append(cyl(p.sense_joy_r, p.lid_t + 0.4, jx, jy,
                       -0.2, sections=48))
    return subtract(lid, cutters)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


PARTS = {
    "base_slim": lambda p: build_tray(p, battery=False),
    "base_battery": lambda p: build_tray(p, battery=True),
    "lid_radar": build_lid_radar,
    "lid_sense": build_lid_sense,
}


def main():
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--part", choices=list(PARTS) + ["all"], default="all")
    ap.add_argument("--out", default="stl")
    ap.add_argument("--check-only", action="store_true",
                    help="build + report watertight/volume, don't write files")
    args = ap.parse_args()

    if args.list:
        print("\n".join(PARTS))
        return

    p = Params()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    names = list(PARTS) if args.part == "all" else [args.part]
    for name in names:
        mesh = PARTS[name](p)
        watertight = bool(mesh.is_watertight)
        b = mesh.bounds
        size = np.round(b[1] - b[0], 2)
        print(f"{name:14s} watertight={watertight} "
              f"vol={mesh.volume / 1000:.1f} cm³ "
              f"size={size[0]}×{size[1]}×{size[2]} mm")
        if not args.check_only:
            path = outdir / f"thoth_{name}.stl"
            mesh.export(str(path))
            print(f"  -> {path}")


if __name__ == "__main__":
    main()
