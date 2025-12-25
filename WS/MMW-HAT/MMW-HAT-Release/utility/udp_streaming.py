import json
import logging
import socket

from utility.BGT60TR13C import BGT60TR13C
from utility.helper import find_register_config_in_directory, find_setting_in_directory, calculate_frame_size

def main(UDP_IP, UDP_PORT, CFG_DIR, FN=None):
    logging.info("Calling streaming main function.")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    bgt60tr13c = BGT60TR13C(spi_speed=50_000_000, save_to_file=FN)
    bgt60tr13c.check_chip_id()
    reg_fn = find_register_config_in_directory(CFG_DIR)
    bgt60tr13c.load_register_config_file(reg_fn)
    setting_fn = find_setting_in_directory(CFG_DIR)
    with open(setting_fn, 'r') as file:
        setting_data = json.load(file)
    frame_size = calculate_frame_size(setting_data)

    # Calculate the total number of samples per frame.
    # 4096 samples = 2048 words, which is 25% FIFO.
    # The burst read cannot be too large due to SPI buffer limit.
    bgt60tr13c.set_fifo_parameters(frame_size, 4096, 2048)
    bgt60tr13c.start()
    try:
        while True:
            sample_bytes = bgt60tr13c.frame_buffer.get()
            sock.sendto(bytes(sample_bytes), (UDP_IP, UDP_PORT))
    except KeyboardInterrupt:
        logging.info("Stopped by user")
        bgt60tr13c.stop()
    finally:
        sock.close()

if __name__ == "__main__":
    main("127.0.0.1", 9575, "../radar_config/config_3rx_3m")
