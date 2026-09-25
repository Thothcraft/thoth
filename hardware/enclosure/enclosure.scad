// Thoth node enclosure — parametric OpenSCAD source.
// Mirror of enclosure.py (same parameter names). Select the part, F6
// render, File → Export → Export as STL.
//
//   PART = "base_slim" | "base_battery" | "lid_radar" | "lid_sense"
//
// All mm. Z-up. Origin: outer bottom west/south corner.
// Params marked // VERIFY: measure on YOUR boards before printing.

PART = "base_slim";

// ---------------- board geometry ----------------
pcb_w = 85;  pcb_d = 56;  pcb_t = 1.6;
hole_inset = [3.5, 3.5];  hole_span = [58, 49];

// ---------------- stack heights -----------------
under_board_slim    = 4;
under_board_battery = 20;              // PiSugar 3 Plus stack  // VERIFY
hat_gap      = 20;                     // Pi top → HAT bottom (active cooler + tall header)
hat_t        = 1.6;
radar_head   = 8;
sense_head   = 13;                     // joystick ≈10 mm

// ---------------- shell --------------------------
wall = 2.2;  floor_t = 2.0;  lid_t = 2.0;  lid_skirt = 5.0;
clearance = 0.35;  corner_r = 3.0;  fillet_r = 4.2;
screw_pilot = 2.1;  screw_clear = 3.0;  post_r = 3.0;

// ---------------- clips ---------------------------
clip_w = 6;  clip_t = 1.0;  clip_h = 3.4;  clip_hook = 0.7;

// ---------------- port canyons ---------------------
west_window  = [10, 46];
east_window  = [10, 46];
north_window = [30, 60];
sd_x = 39;  sd_w = 22;  sd_h = 4;      // VERIFY

// ---------------- battery well ----------------------
ps_west_window = [14, 44];  ps_window_h = 8;    // VERIFY
ps_button_y = 28;  ps_button_r = 2.5;           // VERIFY

// ---------------- radar lid --------------------------
hat_offset = [10, 0];                            // HAT SW corner on Pi  // VERIFY
radar_ant_xy   = [32.5, 20];                     // antenna centre on HAT // MEASURE
radar_aperture = [16, 22];                       // covers 40°H × 65°V FOV
radar_membrane = 1.0;                            // 0 = fully open

// ---------------- sense lid --------------------------
sense_matrix_xy = [28, 38];                      // MEASURE
sense_matrix_wh = [34, 34];
sense_joy_xy    = [40, 14];                      // MEASURE
sense_joy_r     = 6;

// ---------------- vents ------------------------------
vent_rows = 3;  vent_cols = 6;
vent_slot = [6, 2.4];  vent_pitch = [10, 6];
vent_wall_z = 0.55;
lid_vent_grid = [4, 3];  lid_vent_r = 2.2;  lid_vent_pitch = 9;

// ---------------- derived ----------------------------
inner_w = pcb_w + 2*clearance;
inner_d = pcb_d + 2*clearance;
outer_w = inner_w + 2*wall;
outer_d = inner_d + 2*wall;
ub = (PART == "base_battery") ? under_board_battery : under_board_slim;
head = max(radar_head, sense_head);   // tray fits both tops
stack_top = floor_t + ub + pcb_t + hat_gap + hat_t + head;
wall_h = stack_top - floor_t;
board_bot = floor_t + ub;
skirt_t = 1.6;  inset = clearance + skirt_t;

pi_holes = [ [hole_inset[0], hole_inset[1]],
             [hole_inset[0], hole_inset[1]+hole_span[1]],
             [hole_inset[0]+hole_span[0], hole_inset[1]],
             [hole_inset[0]+hole_span[0], hole_inset[1]+hole_span[1]] ];

tower_xy = [ [wall + fillet_r*0.55,              wall + fillet_r*0.55],
             [wall + inner_w - fillet_r*0.55,    wall + fillet_r*0.55],
             [wall + fillet_r*0.55,              wall + inner_d - fillet_r*0.55],
             [wall + inner_w - fillet_r*0.55,    wall + inner_d - fillet_r*0.55] ];

// ---------------- helpers ------------------------------
module rbox(w, d, h, r=0) {
  if (r <= 0.01)
    cube([w, d, h]);
  else
    hull()
      for (cx = [r, w - r]) for (cy = [r, d - r])
        translate([cx, cy, 0]) cylinder(r=r, h=h, $fn=40);
}

// ---------------- tray ----------------------------------
module tray() {
  difference() {
    union() {
      difference() {
        rbox(outer_w, outer_d, stack_top, corner_r);
        translate([wall, wall, floor_t])
          cube([inner_w, inner_d, wall_h + 20]);
      }
      // corner towers (lid screw posts)
      for (t = tower_xy)
        translate([t[0], t[1], 0]) cylinder(r=fillet_r, h=stack_top);
      // Pi support posts
      for (hh = pi_holes)
        translate([wall + clearance + hh[0], wall + clearance + hh[1], 0])
          cylinder(r=post_r, h=floor_t + ub);
      // retention clips, north wall inner face
      for (fx = [inner_w*0.30, inner_w*0.70]) {
        cx = wall + fx;
        translate([cx - clip_w/2, wall + inner_d - clip_t,
                   board_bot - clip_h])
          cube([clip_w, clip_t, clip_h + pcb_t]);
        translate([cx - clip_w/2, wall + inner_d - clip_t - clip_hook,
                   board_bot - 0.4])
          cube([clip_w, clip_hook, pcb_t + 0.4]);
      }
    }
    // tower pilots
    for (t = tower_xy)
      translate([t[0], t[1], stack_top - 14])
        cylinder(d=screw_pilot, h=14, $fn=24);
    // post pilots
    for (hh = pi_holes)
      translate([wall + clearance + hh[0], wall + clearance + hh[1],
                 floor_t + ub - 6])
        cylinder(d=screw_pilot, h=6, $fn=24);
    // port canyons — west + east full port windows
    win_h = stack_top - board_bot - 1.0;
    translate([-0.2, wall + west_window[0], board_bot + 0.5])
      cube([wall + 0.4, west_window[1] - west_window[0], win_h]);
    translate([outer_w - wall - 0.2, wall + east_window[0],
               board_bot + 0.5])
      cube([wall + 0.4, east_window[1] - east_window[0], win_h]);
    // north ribbon window
    translate([wall + north_window[0], wall + inner_d - 0.2,
               stack_top - 12])
      cube([north_window[1] - north_window[0], wall + 0.4, 12]);
    // south SD slot
    translate([wall + clearance + sd_x - sd_w/2, -0.2, board_bot - 1.0])
      cube([sd_w, wall + 0.4, sd_h + 2.5]);
    // south wall vent grille
    for (i = [0:vent_cols-1]) for (j = [0:vent_rows-1])
      translate([outer_w/2 - (vent_cols-1)*vent_pitch[0]/2
                   + i*vent_pitch[0] - vent_slot[0]/2,
                 -0.3,
                 floor_t + wall_h*vent_wall_z + j*vent_pitch[1]])
        cube([vent_slot[0], wall + 0.6, vent_slot[1]]);
    // battery well ports
    if (PART == "base_battery") {
      translate([-0.2, wall + ps_west_window[0], floor_t + 3.0])
        cube([wall + 0.4, ps_west_window[1]-ps_west_window[0],
              ps_window_h]);
      translate([wall + clearance + ps_button_y, -0.3,
                 floor_t + 3.0 + ps_window_h/2])
        rotate([-90, 0, 0])
          cylinder(r=ps_button_r, h=wall + 0.6, $fn=24);
    }
  }
}

// ---------------- lid (local frame: plate base z=0) -----
module lid_plate() {
  translate([-inset, -inset, 0])
    rbox(outer_w + 2*inset, outer_d + 2*inset, lid_t,
         corner_r + inset);
}
module lid_skirt_shell() {
  difference() {
    translate([-inset, -inset, -lid_skirt])
      rbox(outer_w + 2*inset, outer_d + 2*inset, lid_skirt,
           corner_r + inset);
    translate([-clearance, -clearance, -lid_skirt])
      rbox(outer_w + 2*clearance, outer_d + 2*clearance,
           lid_skirt + 0.1, corner_r + clearance);
  }
}
module lid_base() {
  difference() {
    union() { lid_plate(); lid_skirt_shell(); }
    for (t = tower_xy)
      translate([t[0], t[1], -0.2])
        cylinder(d=screw_clear, h=lid_t + 0.4, $fn=24);
    cols = lid_vent_grid[0];  rows = lid_vent_grid[1];
    for (i = [0:cols-1]) for (j = [0:rows-1])
      translate([outer_w*0.70 + i*lid_vent_pitch,
                 outer_d/2 - (rows-1)*lid_vent_pitch/2
                   + j*lid_vent_pitch, -0.2])
        cylinder(r=lid_vent_r, h=lid_t + 0.4, $fn=24);
  }
}
module lid_radar() {
  ax = wall + clearance + hat_offset[0] + radar_ant_xy[0];
  ay = wall + clearance + hat_offset[1] + radar_ant_xy[1];
  aw = radar_aperture[0];  ah = radar_aperture[1];
  difference() {
    lid_base();
    if (radar_membrane > 0.05) {
      translate([ax - aw/2, ay - ah/2, radar_membrane - 0.1])
        cube([aw, ah, lid_t - radar_membrane + 0.2]);
      translate([ax - aw/2 - 1, ay - ah/2 - 1,
                 radar_membrane - 1.2])
        cube([aw + 2, ah + 2, 1.2]);
    } else {
      translate([ax - aw/2, ay - ah/2, -0.2])
        cube([aw, ah, lid_t + 0.4]);
    }
  }
}
module lid_sense() {
  mx = wall + clearance + hat_offset[0] + sense_matrix_xy[0];
  my = wall + clearance + hat_offset[1] + sense_matrix_xy[1];
  jx = wall + clearance + hat_offset[0] + sense_joy_xy[0];
  jy = wall + clearance + hat_offset[1] + sense_joy_xy[1];
  difference() {
    lid_base();
    translate([mx - sense_matrix_wh[0]/2, my - sense_matrix_wh[1]/2, -0.2])
      cube([sense_matrix_wh[0], sense_matrix_wh[1], lid_t + 0.4]);
    translate([jx, jy, -0.2])
      cylinder(r=sense_joy_r, h=lid_t + 0.4, $fn=48);
  }
}

// ---------------- dispatch -----------------------------
if (PART == "base_slim" || PART == "base_battery") tray();
else if (PART == "lid_radar") lid_radar();
else if (PART == "lid_sense") lid_sense();
else echo("unknown PART");
