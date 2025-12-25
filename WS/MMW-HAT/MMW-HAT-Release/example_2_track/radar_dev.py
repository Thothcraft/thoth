import json
import os
import socket
import sys

import numpy as np

# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

from utility.helper import parse_radar_cfg, find_setting_in_directory, parse_full_frame, read_uint12, split_samples


class RadarDev:
    def __init__(self, port, setting_fn):
        self.port = port
        self.sock = None
        with open(setting_fn, 'r') as file:
            setting = json.load(file)
        self.cfg = parse_radar_cfg(setting)

    def open_radar_device(self):
        # Set up UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', self.port))

    def get_next_frame(self):
        raw_data, _ = self.sock.recvfrom(65536)  # Adjust buffer size as needed. It should be large enough.
        raw_data = list(raw_data)
        (version, seq, data_len, raw_data) = parse_full_frame(raw_data)
        adc_data = read_uint12(raw_data)
        adc_data_split = split_samples(adc_data, 1, self.cfg["num_chirps_per_frame"], self.cfg["num_samples_per_chirp"], self.cfg["num_antennas"])
        adc_data_split = np.transpose(adc_data_split[0,:,:,:], (2, 0, 1))
        return adc_data_split

