# Thoth node enclosure

Parametric 3D-printed case for the Thoth sensing node
(Raspberry Pi + DreamHAT+ BGT60TR13C radar). Two bottoms, one lid →
**3 parts**:

```
                 TOP (lid cap)
            ┌─────────────┐
            │  radar      │
            │  DreamHAT+  │
BOTTOM      ├─────────────┤
 slim       │ Pi + HAT    │
 battery    │ + PiSugar 3 │
            │   Plus      │
            └─────────────┘
```

The RPi5 Active Cooler mounts under the DreamHAT's GPIO header inside
the `hat_gap` — it doesn't change any enclosure dimension.

- **Board support**: Raspberry Pi 3B+/4B/5 (same 85×56 mm footprint +
  mounting holes; ports stay fully accessible through the side windows,
  so the Pi version doesn't matter).
- **Mounting**: flat back — double-sided tape / Command strips / VHB.
  Wall-mount the radar variant face-out; desk use for either.
- **Assembly**: drop Pi onto the 4 posts → snap under the 2 clips →
  mount HAT → cap lid → 4× M2.5 self-tap screws. No glue, no inserts.

## Files

| file | what |
| ---- | ---- |
| `enclosure.py` | Python generator (trimesh + manifold3d) — emits watertight STL |
| `enclosure.scad` | Same model in OpenSCAD for GUI tweaking |
| `DIMENSIONS.md` | Every parameter, its source, and what to VERIFY on real boards |
| `stl/` | Pre-built parts: `thoth_base_slim`, `thoth_base_battery`, `thoth_lid_radar` |

## Regenerate the STLs

```bash
pip install trimesh manifold3d shapely
python enclosure.py                 # all 3 parts → stl/
python enclosure.py --part lid_radar
python enclosure.py --check-only    # validate without writing
```

Or open `enclosure.scad`, set `PART`, F6, export STL.

## Print settings

- **Material: PETG** (recommended — tougher than PLA, survives cooler
  exhaust temps, low creep in the clips; still low-loss at 60 GHz).
  PLA works too. Avoid carbon-fiber-filled filaments near the radar
  window — CF is conductive and attenuates mmWave.
- 0.2 mm layers, 0.4 mm nozzle, 3 perimeters, 20 % infill, **no
  supports** anywhere (aperture pocket prints face-up inside the lid).
- Print orientation: **trays as-is** (flat on the floor); **lids
  top-face-down** for a clean exterior surface — the radar membrane is
  on the underside of the plate, unaffected.
- The membrane is 1.2 mm — make sure your slicer doesn't thin it below
  ~1.0 mm (set "minimum thickness" if warned).

## BOM per unit

- 4× M2.5×10 self-tapping screws (lid → corner towers)
- 4× M2.5×6 self-tap (optional — Pi → posts; the 2 clips alone hold it)
- Tall/stacking 40-pin header if using the **active cooler under a HAT**
  (stock header is too short — see DIMENSIONS.md `hat_gap`)
- Double-sided tape / Command strips for wall mounting
- Rubber feet ×4 for desk use (stick-on, 8–12 mm Ø)

## Fit notes

- Clearances are +0.35 mm everywhere (fits most FDM printers on first
  try); if your printer runs tight, drop `clearance` to 0.25.
- The radar aperture is sized for the published 40°H × 65°V FOV of the
  DreamHAT+ at an 8 mm antenna→membrane gap; see `radar_*` params.
- ⚠ Before printing: measure the VERIFY dimensions in `DIMENSIONS.md`
  on your actual boards (PiSugar height/ports, header stack height).
