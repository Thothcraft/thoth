# Dimensions & sources

Coordinate system: Z up, origin at outer bottom west/south corner.
**X = the 85 mm Pi edge** (west short edge carries USB-C/HDMI/audio on
Pi4/5), **Y = the 56 mm edge** (GPIO header runs along north).

## Locked (official mechanical drawings)

| param | value | source |
| ----- | ----- | ------ |
| Pi PCB | 85.0 × 56.0 × 1.6 | Raspberry Pi mech drawing — same for 3B+/4B/5 |
| Mounting holes | M2.5, inset 3.5 / 3.5, span 58 × 49 | same |
| DreamHAT+ FOV | 40° horiz, 65° vert (effective) | vendor datasheet (BGT60TR13C, 58–63.5 GHz, 1TX+3RX AiP 6.5×5.0 mm) |
| DreamHAT+ PCB | 65 × 56.5, GPIO header on bottom edge, notches at top corners | vendor dimension drawing + on-device check |
| DreamHAT+ chip centre | (30, 26.5) from HAT SW corner | measured on device: 30 mm down / 30 mm right from the notch-adjacent top-left corner |
| `hat_gap` | 16.0 | measured: Pi5 Active Cooler (~13.5) + ~2 mm air to HAT underside — the cooler mounts under the HAT header inside this gap and does not change enclosure dims |
| `hat_offset` | (0, −0.5) | HAT SW corner on Pi board — holes share the Pi 58×49 grid, board spans y −0.5…56 |
| PiSugar 3 Plus PCB | 65 × 56 | PiSugar product wiki |
| 60 GHz λ (free space) | ~5 mm | membrane 1.2 mm ≪ λ — thin uniform dielectric |

## Nominal defaults — VERIFY on your hardware

| param | default | why it matters |
| ----- | ------- | -------------- |
| `under_board_battery` | 20.0 | PiSugar 3 Plus ~15 mm incl. cell + ~4 mm pogo frame + margin — **measure the assembled Pi+PiSugar height under the PCB** |
| `radar_ant_xy` | (30, 26.5) | **measured on the real board** — see locked table; keep as param in case boards rev |
| `radar_aperture` | 16 × 22 | opening at membrane level; with `radar_head` = 8 this covers the full 40°H × 65°V cone with ~±2 mm slack |
| `sd_x` | 39 | SD slot centre from the west edge (Pi4/5 bottom edge) |
| `ps_west_window` / `ps_button_y` | 14–44 / 28 | PiSugar USB-C + µUSB + power button positions on the deep variant |
| `radar_head` | 8 | air gap above HAT PCB to lid plate |

## Fixed tolerances

- `clearance` 0.35 mm board↔wall and tray↔lid (first-print friendly)
- `screw_pilot` Ø2.1 mm in posts/towers for M2.5 self-tap
- `screw_clear` Ø3.0 mm through the lid plate
- `clip_t` 1.2 mm cantilever, 1.0 mm hook over the Pi top edge (≥1 mm — fab minimum wall)

## Radar window note

The aperture leaves a **1.2 mm integral membrane** (`radar_membrane`,
set 0 for open). At 60 GHz a thin uniform dielectric wall with an air
gap below is a reasonable hobbyist radome — uniform thickness, no ribs
or seams crossing the antenna face. The recessed outer pocket makes the
membrane flush-protected. If you need max range, set
`radar_membrane = 0` for a fully open aperture.
