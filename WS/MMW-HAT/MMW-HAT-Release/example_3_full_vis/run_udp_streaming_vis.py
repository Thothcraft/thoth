import sys
import os

# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

import utility.udp_streaming

# IP = "192.168.236.1"
IP = "127.0.0.1"    # set the IP to receive data
PORT = 9575
CFG_DIR = "../radar_config/config_3rx_3m"
# FN = "data/mmw_spi"    # set None to disable saving data to file
FN = None
utility.udp_streaming.main(IP, PORT, CFG_DIR, FN)