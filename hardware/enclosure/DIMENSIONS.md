# Dimensions & sources

Coordinate system: Z up, origin at outer bottom west/south corner.
**X = the 85 mm Pi edge** (west short edge carries USB-C/HDMI/audio on
Pi4/5), **Y = the 56 mm edge** (GPIO header runs along north).

## Locked (official mechanical drawings)

| param | value | source |
| ----- | ----- | ------ |
| Pi PCB | 85.0 × 56.0 × 1.6 | Raspberry Pi mech drawing — same for 3B+/4B/5 |
| Mounting holes | M2.5, inset 3.5 / 3.5, span 58 × 49 | same |
| Sense HAT PCB | 65.1 × 56.5 × ~1.6 | Raspberry Pi HAT spec / Sense HAT mech |
| DreamHAT+ FOV | 40° horiz, 65° vert (effective) | vendor datasheet (BGT60TR13C, 58–63.5 GHz, 1TX+3RX AiP 6.5×5.0 mm) |
| PiSugar 3 Plus PCB | 65 × 56 | PiSugar product wiki |
| 60 GHz λ (free space) | ~5 mm | membrane 1.0 mm ≪ λ — thin uniform dielectric |

## Nominal defaults — VERIFY on your hardware

| param | default | why it matters |
| ----- | ------- | -------------- |
| `under_board_battery` | 20.0 | PiSugar 3 Plus ~15 mm incl. cell + ~4 mm pogo frame + margin — **measure the assembled Pi+PiSugar height under the PCB** |
| `hat_gap` | 20.0 | Pi top → HAT bottom. Active Cooler ≈ 12–13 mm + gap. Stock 8.5 mm header **won't clear the cooler** — you need a tall stacking header; measure the real stack |
| `hat_offset` | (10, 0) | HAT SW corner on the Pi board (HATs are narrower than the Pi) |
| `radar_ant_xy` | (32.5, 20) | **antenna centre on the DreamHAT+** — measured from the HAT's SW corner. The aperture + membrane sit directly over this point; a ±2 mm error is absorbed by the pocket flare, more is not |
| `radar_aperture` | 16 × 22 | opening at membrane level; with `radar_head` = 8 this covers the full 40°H × 65°V cone with ~±2 mm slack |
| `sense_matrix_xy` / `sense_matrix_wh` | (28, 38) / 34×34 | 8×8 LED matrix centre/size on the HAT |
| `sense_joy_xy` / `sense_joy_r` | (40, 14) / 6 | joystick centre + opening radius (Ø12) |
| `sd_x` | 39 | SD slot centre from the west edge (Pi4/5 bottom edge) |
| `ps_west_window` / `ps_button_y` | 14–44 / 28 | PiSugar USB-C + µUSB + power button positions on the deep variant |
| `head` radar/sense | 8 / 13 | air gap above HAT PCB to lid; sense = joystick height + margin |

## Fixed tolerances

- `clearance` 0.35 mm board↔wall and tray↔lid (first-print friendly)
- `screw_pilot` Ø2.1 mm in posts/towers for M2.5 self-tap
- `screw_clear` Ø3.0 mm through the lid plate
- `clip_t` 1.0 mm PETG cantilever, 0.7 mm hook over the Pi top edge

## Radar window note

The aperture leaves a **1.0 mm integral membrane** (`radar_membrane`,
set 0 for open). At 60 GHz a thin uniform dielectric wall with an air
gap below is a reasonable hobbyist radome — uniform thickness, no ribs
or seams crossing the antenna face. The recessed outer pocket makes the
membrane flush-protected. If you need max range, set
`radar_membrane = 0` for a fully open aperture.
