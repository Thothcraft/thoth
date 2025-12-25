import json
import numpy as np
from collections import deque
from internal.DBF import DBF
from internal.doppler import DopplerAlgo



def range_angle_map_2_x_y_map_binary(polar_image, r, theta, x, y, sz):
    output = np.zeros((len(x), len(y)))
    indices = np.nonzero(polar_image)
    for idx_0, idx_1 in zip(indices[0], indices[1]):
        r_temp = r[idx_0]
        theta_temp = np.deg2rad(theta[idx_1])
        x_temp = r_temp * np.cos(theta_temp)
        y_temp = r_temp * np.sin(theta_temp)
        x_idx = np.argmin(np.abs(x - x_temp))
        y_idx = np.argmin(np.abs(y - y_temp))
        output[np.max([0, x_idx - sz]):np.min([len(x) - 1, x_idx + sz]),
        np.max([0, y_idx - sz]):np.min([len(y) - 1, y_idx + sz])] = 1
    return output


class SigProc:
    def __init__(self, processing_config_fn, radar_config):
        with open(processing_config_fn, 'r') as file:
            self.processing_config = json.load(file)
        self.radar_config = radar_config
        self.num_beams = self.processing_config["num_beams"]
        self.max_angle_deg = self.processing_config["max_angle_deg"]
        self.num_rx_antennas = self.radar_config['num_antennas']
        # self.range_angle_map_buffer = deque(maxlen=self.processing_config["buffer_len"])
        self.xy_map_buffer = deque(maxlen=self.processing_config["buffer_len"])
        self.location_buffer = deque(maxlen=self.processing_config["buffer_len"])
        self.buffer_decay = self.processing_config["buffer_decay"]
        bw = self.radar_config['bandwidth']
        c = 3e8
        self.max_range_m = c / (2 * bw) * self.radar_config['num_samples_per_chirp'] / 2
        self.range_bin = np.arange(0,
                                   self.radar_config['num_samples_per_chirp']) * self.max_range_m / self.radar_config['num_samples_per_chirp']
        self.angle_bin = np.linspace(-self.max_angle_deg, self.max_angle_deg, self.num_beams)

        self.x_bin = np.arange(0, self.processing_config["max_x"], self.processing_config["spatial_resolution"])
        self.y_bin = np.arange(-self.processing_config["max_y"], self.processing_config["max_y"],
                               self.processing_config["spatial_resolution"])
        self.dead_zone = self.processing_config["dead_zone"]

        self.detection_num_frames = self.processing_config["detection"]["num_frames"]
        self.detection_range_win = self.processing_config["detection"]["range_win"]
        self.detection_angle_win = self.processing_config["detection"]["angle_win"]


        # Create objects for Range-Doppler and DBF
        self.doppler = DopplerAlgo(self.radar_config, self.num_rx_antennas)
        self.dbf = DBF(self.num_rx_antennas, num_beams=self.num_beams, max_angle_degrees=self.max_angle_deg)

    def range_angle_map(self, frame):
        # frame: (num_rx_antennas x num_chirps_per_frame x num_samples_per_chirp)
        rd_spectrum = np.zeros((self.radar_config['num_samples_per_chirp'], 2 * self.radar_config['num_chirps_per_frame'],
                                self.num_rx_antennas), dtype=complex)

        for i_ant in range(self.num_rx_antennas):  # For each antenna
            # Current RX antenna (num_chirps_per_frame x num_samples_per_chirp)
            mat = frame[i_ant, :, :]

            # Compute Doppler spectrum
            dfft_dbfs = self.doppler.compute_doppler_map(mat, i_ant)
            rd_spectrum[:, :, i_ant] = dfft_dbfs

        # Compute Range-Angle map
        rd_beam_formed = self.dbf.run(rd_spectrum)
        beam_range_energy = np.linalg.norm(rd_beam_formed, axis=1) / np.sqrt(self.num_beams)
        return beam_range_energy

    def target_validation(self, beam_range_energy, range_idx, angle_idx):
        range_resolution = self.range_bin[1] - self.range_bin[0]
        range_win_m = self.detection_range_win
        range_win_half_len = int(range_win_m / range_resolution / 2)
        range_win_half_len = np.min((range_win_half_len, range_idx))
        range_win_half_len = np.min((range_win_half_len, len(self.range_bin) - 1 - range_idx))

        angle_resolution = self.angle_bin[1] - self.angle_bin[0]
        angle_win_deg = self.detection_angle_win
        angle_win_half_len = int(angle_win_deg / angle_resolution / 2)
        angle_win_half_len = np.min((angle_win_half_len, angle_idx))
        angle_win_half_len = np.min((angle_win_half_len, len(self.angle_bin) - 1 - angle_idx))

        if range_win_half_len < 2 or angle_win_half_len < 2:
            return False
        else:
            return True

    def target_detection(self, beam_range_energy):
        range_resolution = self.range_bin[1] - self.range_bin[0]
        range_clear_idx = int(self.dead_zone / range_resolution)
        beam_range_energy_cropped = beam_range_energy[range_clear_idx:, :]
        range_idx, angle_idx = np.unravel_index(beam_range_energy_cropped.argmax(), beam_range_energy_cropped.shape)
        range_idx += range_clear_idx

        # range_idx, angle_idx = np.unravel_index(beam_range_energy.argmax(), beam_range_energy.shape)

        beam_range_map = np.zeros_like(beam_range_energy)
        if self.target_validation(beam_range_energy, range_idx, angle_idx):
            beam_range_map[range_idx, angle_idx] = 1
            self.location_buffer.append([self.range_bin[range_idx], self.angle_bin[angle_idx]])
        else:
            self.location_buffer.append(None)

        xy_map = range_angle_map_2_x_y_map_binary(beam_range_map, self.range_bin, self.angle_bin, self.x_bin,
                                                  self.y_bin, 2)
        self.xy_map_buffer.append(xy_map)

        weights = self.buffer_decay ** np.arange(len(self.xy_map_buffer))
        weights = weights[::-1]
        weights = weights[:, np.newaxis, np.newaxis]

        xy_map_buffer_temp = np.array(self.xy_map_buffer)
        weighted_xy_map_buffer = xy_map_buffer_temp * weights
        final_xy_map = np.max(weighted_xy_map_buffer, axis=0)

        latest_target_detection_idx = None
        for i in range(len(self.location_buffer)):
            if self.location_buffer[len(self.location_buffer) - 1 - i] is not None:
                latest_target_detection_idx = i
                break

        if latest_target_detection_idx is None or latest_target_detection_idx > self.detection_num_frames:
            location = np.array([np.nan, np.nan])
            score = np.nan
        else:
            last_valid_location = self.location_buffer[len(self.location_buffer) - 1 - latest_target_detection_idx]
            r = last_valid_location[0]
            theta = last_valid_location[1]
            location = np.array([r * np.cos(np.deg2rad(theta)), r * np.sin(np.deg2rad(theta))])
            score = self.buffer_decay ** latest_target_detection_idx

        return final_xy_map, location, score

    def update(self, frame):
        beam_range_energy = self.range_angle_map(frame)
        final_xy_map, location, score = self.target_detection(beam_range_energy)

        gui_plot = {"x_axis": self.x_bin,
                    "y_axis": self.y_bin,
                    "map": final_xy_map}

        return location, score, gui_plot
