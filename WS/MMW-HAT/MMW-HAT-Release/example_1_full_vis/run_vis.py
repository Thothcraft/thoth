import sys
import os

# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

import utility.udp_real_time_vis
port = 9575
num_rows = 1
num_cols = 1
plots = [(0, 0, "Range", "Doppler")]
cfg_dir = "../radar_config/config_3rx_3m"
num_angle_bins = 16
# fn = "data/mmw_udp"    # set None to disable saving data to file
fn = None
utility.udp_real_time_vis.main(port, cfg_dir, num_rows, num_cols, plots, num_angle_bins, fn)

