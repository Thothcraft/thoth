import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

from utility.helper import find_setting_in_directory
from utility.mmw_cube_proc_v0 import CubeProcessor


def main(data_fn, proc_save_dir, plots, cfg_dir, num_angle_bins):
    setting_fn = find_setting_in_directory(cfg_dir)
    with open(setting_fn, 'r') as file:
        setting = json.load(file)
    mmw_proc = CubeProcessor(setting, num_azimuth_bin=num_angle_bins, num_elevation_bin=num_angle_bins)

    directory = os.path.dirname(proc_save_dir)
    if directory:  # Check if a directory is specified in the path
        os.makedirs(directory, exist_ok=True)

    # Open the file in binary read mode
    with open(data_fn, "rb") as file:
        while True:
            version_bytes = file.read(4)
            if not version_bytes:
                break
            version = int.from_bytes(version_bytes, byteorder='little', signed=False)  # Read the first N bytes
            if version == 0:
                seq = int.from_bytes(file.read(4), byteorder='little', signed=False)
                data_len = int.from_bytes(file.read(4), byteorder='little', signed=False)
                mmw_data = file.read(data_len)
                mmw_proc.process_raw_data(mmw_data)

                print(f"seq: {seq}")
                for plot in plots:
                    axis_0_name, axis_1_name = plot
                    img = mmw_proc.vis_2d(axis_0_name, axis_1_name)
                    img = np.log10(img)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.savefig(os.path.join(directory, str(seq) + "_" + axis_0_name + "_" + axis_1_name +".jpg"), bbox_inches='tight', pad_inches=0)
                    plt.close()

if __name__ == '__main__':
    data_fn = "data/mmw_spi_20241119_081114_994.bin"
    proc_save_dir = "mmw_proc/"
    plots = [("Range", "Doppler"),
             ("Azimuth", "Range"),
             ("Azimuth", "Doppler")]
    cfg_dir = "../radar_config/config_3rx_3m"
    num_angle_bins = 16
    main(data_fn, proc_save_dir, plots, cfg_dir, num_angle_bins)
