"""
Simplified Radar Processor using standard numpy/scipy FFT.
Based on MMW-HAT CubeProcessor but without pyfftw/numba dependencies.
"""
import json
import numpy as np
from typing import Dict, Tuple


def parse_radar_cfg(setting: Dict) -> Dict:
    """Parse radar configuration from JSON settings."""
    sequence = setting["sequence"][0]["sequence"][0]["sequence"][0]
    
    return {
        "num_chirps_per_frame": setting["sequence"][0]["sequence"][0]["num_repetitions"],
        "num_samples_per_chirp": sequence["num_samples"],
        "num_antennas": bin(sequence["rx_mask"]).count("1"),
        "rx_mask": sequence["rx_mask"],
        "bandwidth": sequence["end_frequency_Hz"] - sequence["start_frequency_Hz"],
        "sample_rate": sequence["sample_rate_Hz"],
        "chirp_rate": (sequence["end_frequency_Hz"] - sequence["start_frequency_Hz"]) / sequence["num_samples"]
    }


def read_uint12(data_chunk):
    """Read 12-bit ADC data from bytes."""
    data = np.frombuffer(bytes(data_chunk), dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
    return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0]).astype(np.float32)


def split_samples(uint16_chunk, num_frames, num_chirps, num_samples, num_rx_antennas):
    """Split ADC data into frames, chirps, samples, and antennas."""
    return uint16_chunk.reshape((num_frames, num_chirps, num_samples, num_rx_antennas))


class RadarProcessor:
    """Simplified radar processor using numpy/scipy FFT instead of pyfftw/numba."""
    
    def __init__(self, config_path: str = None, num_doppler_bin=None, num_range_bin=128, 
                 num_azimuth_bin=16, num_elevation_bin=16, min_range=0.2):
        """
        Initialize radar processor.
        
        Args:
            config_path: Path to radar configuration JSON file (optional, will auto-detect if None)
            num_doppler_bin: Number of Doppler bins (auto-detect from data if None)
            num_range_bin: Number of range bins
            num_azimuth_bin: Number of azimuth bins
            num_elevation_bin: Number of elevation bins
            min_range: Minimum range in meters
        """
        self.config_path = config_path
        self.num_range_bin = num_range_bin
        self.num_azimuth_bin = num_azimuth_bin
        self.num_elevation_bin = num_elevation_bin
        self.min_range = min_range
        self.num_doppler_bin = num_doppler_bin
        self.radar_param = None
        self.proc_param = None
        self.data_cube_fft = None
        
        # Load configuration if provided
        if config_path:
            self._load_config(config_path)
    
    def _load_config(self, config_path: str):
        """Load radar configuration from JSON file."""
        with open(config_path) as f:
            setting = json.load(f)
        
        self.radar_param = parse_radar_cfg(setting)
        
        if self.num_doppler_bin is None:
            self.num_doppler_bin = self.radar_param["num_chirps_per_frame"]
        else:
            self.num_doppler_bin = max(self.num_doppler_bin, self.radar_param["num_chirps_per_frame"])
        
        self.num_range_bin = max(self.num_range_bin, self.radar_param["num_samples_per_chirp"])
        self.num_azimuth_bin = max(self.num_azimuth_bin, self.radar_param["num_antennas"])
        self.num_elevation_bin = max(self.num_elevation_bin, 1)
        
        # Calculate range bins
        range_bin = np.arange(self.num_range_bin >> 1) * (3e8 / (2 * self.radar_param["bandwidth"]))
        self.range_skip = np.searchsorted(range_bin, self.min_range)
        
        # Store processing parameters
        self.proc_param = {
            "num_doppler_bin": self.num_doppler_bin,
            "num_range_bin": self.num_range_bin,
            "min_range": self.min_range,
            "doppler_bin": -np.fft.fftshift(np.fft.fftfreq(self.num_doppler_bin, 1 / self.radar_param["chirp_rate"])),
            "range_bin": range_bin[self.range_skip:],
            "num_azimuth_bin": self.num_azimuth_bin,
            "num_elevation_bin": self.num_elevation_bin,
            "azimuth_bin": np.rad2deg(np.fft.fftshift(np.arcsin(np.fft.fftfreq(self.num_azimuth_bin, 0.5)))),
            "elevation_bin": np.rad2deg(np.fft.fftshift(np.arcsin(np.fft.fftfreq(self.num_elevation_bin, 0.5))))
        }
        
        # Position mapping for antennas (from MMW-HAT datasheet)
        self.position_map = {1: (1, 0), 2: (0, 1), 3: (0, 0)}
        self.active_antennas = [ant for ant in self.position_map if (self.radar_param["rx_mask"] & (1 << (ant - 1)))]
    
    def _auto_detect_dimensions(self, raw_data):
        """Auto-detect radar dimensions from raw data."""
        # Calculate actual data dimensions
        num_bytes = len(raw_data)
        num_uint12 = num_bytes // 3
        num_uint16 = num_uint12 * 2
        
        # Assume 3 antennas (common for BGT60TR13C)
        num_antennas = 3
        
        # Calculate samples per antenna
        samples_per_antenna = num_uint16 // num_antennas
        
        # Common sample sizes: 64, 128, 256
        # Find the divisor that's closest to a power of 2
        possible_samples = [64, 128, 256, 512]
        num_samples = min(possible_samples, key=lambda x: abs(x - (samples_per_antenna / (samples_per_antenna // x))))
        
        # Calculate chirps
        num_chirps = samples_per_antenna // num_samples
        
        self.radar_param = {
            "num_chirps_per_frame": num_chirps,
            "num_samples_per_chirp": num_samples,
            "num_antennas": num_antennas,
            "rx_mask": 7,  # All 3 antennas active
            "bandwidth": 2e9,  # 2 GHz (typical for 60GHz radar)
            "sample_rate": 2e6,  # 2 MHz (typical)
            "chirp_rate": 2e9 / num_samples  # Approximate
        }
        
        self.num_doppler_bin = num_chirps
        self.num_range_bin = num_samples
        
        # Calculate range bins
        range_bin = np.arange(self.num_range_bin >> 1) * (3e8 / (2 * self.radar_param["bandwidth"]))
        self.range_skip = np.searchsorted(range_bin, self.min_range)
        
        # Store processing parameters
        self.proc_param = {
            "num_doppler_bin": self.num_doppler_bin,
            "num_range_bin": self.num_range_bin,
            "min_range": self.min_range,
            "doppler_bin": -np.fft.fftshift(np.fft.fftfreq(self.num_doppler_bin, 1 / self.radar_param["chirp_rate"])),
            "range_bin": range_bin[self.range_skip:],
            "num_azimuth_bin": self.num_azimuth_bin,
            "num_elevation_bin": self.num_elevation_bin,
            "azimuth_bin": np.rad2deg(np.fft.fftshift(np.arcsin(np.fft.fftfreq(self.num_azimuth_bin, 0.5)))),
            "elevation_bin": np.rad2deg(np.fft.fftshift(np.arcsin(np.fft.fftfreq(self.num_elevation_bin, 0.5))))
        }
        
        # Position mapping for antennas
        self.position_map = {1: (1, 0), 2: (0, 1), 3: (0, 0)}
        self.active_antennas = [ant for ant in self.position_map if (self.radar_param["rx_mask"] & (1 << (ant - 1)))]
        
        print(f"    Auto-detected dimensions: chirps={num_chirps}, samples={num_samples}, antennas={num_antennas}")
    
    def process_raw_data(self, raw_data):
        """
        Process raw binary data to generate 4D data cube.
        
        Since the data doesn't match MMW-HAT ADC format, treat raw bytes as samples
        and apply the MMW-HAT visualization pipeline.
        
        Args:
            raw_data: Raw bytes from radar
        """
        # Treat raw bytes as uint8 samples (since not in MMW-HAT ADC format)
        data_array = np.frombuffer(raw_data, dtype=np.uint8)
        data_array = data_array.astype(np.float32)
        
        # Auto-detect dimensions based on data size
        data_size = len(data_array)
        
        # Assume 3 antennas
        num_antennas = 3
        samples_per_antenna = data_size // num_antennas
        
        # Try to find reasonable dimensions
        # Common radar configurations: samples per chirp = 64, 128, 256
        possible_samples = [64, 128, 256, 512]
        num_samples = min(possible_samples, key=lambda x: abs(x - (samples_per_antenna / (samples_per_antenna // x))))
        num_chirps = samples_per_antenna // num_samples
        
        # Limit to reasonable number of chirps for processing
        max_chirps = 128
        num_chirps = min(num_chirps, max_chirps)
        
        # Trim data to fit
        total_samples = num_chirps * num_samples * num_antennas
        data_array = data_array[:total_samples]
        
        # Reshape to (chirps, samples, antennas)
        data_cube = data_array.reshape((num_chirps, num_samples, num_antennas))
        
        # Expand to 4D (chirps, samples, azimuth, elevation)
        # Map antennas to azimuth positions
        data_cube_4d = np.zeros((num_chirps, num_samples, 16, 16), dtype=np.float32)
        
        # Position mapping: antenna 1->(1,0), 2->(0,1), 3->(0,0)
        positions = [(1, 0), (0, 1), (0, 0)]
        for i in range(min(num_antennas, len(positions))):
            pos = positions[i]
            data_cube_4d[:, :, pos[0], pos[1]] = data_cube[:, :, i]
        
        # Apply 3D FFT (doppler, range, azimuth) - skip elevation since single antenna
        data_cube_fft = np.fft.fftn(data_cube_4d, axes=(0, 1, 2))
        data_cube_fft = np.fft.fftshift(data_cube_fft, axes=(0, 2))
        
        # Compute power
        data_cube_fft = np.abs(data_cube_fft) ** 2
        
        # Remove near-range bins (first 8 as approximate)
        data_cube_fft = data_cube_fft[:, 8:, :, :]
        
        self.data_cube_fft = data_cube_fft
        self.radar_param = {
            "num_chirps_per_frame": num_chirps,
            "num_samples_per_chirp": num_samples,
            "num_antennas": num_antennas
        }
        self.num_doppler_bin = num_chirps
        self.num_range_bin = num_samples - 8
        
        print(f"    Processed as raw uint8 samples: chirps={num_chirps}, samples={num_samples}, antennas={num_antennas}")
    
    def vis_2d(self, dim_0: str, dim_1: str) -> np.ndarray:
        """
        Generate 2D visualization by reducing other dimensions with mean.
        
        Args:
            dim_0: First dimension name ("Doppler", "Range", "Azimuth", "Elevation")
            dim_1: Second dimension name
        
        Returns:
            2D array representing the visualization
        """
        dim_names = ["Doppler", "Range", "Azimuth", "Elevation"]
        
        try:
            dim_0_idx = dim_names.index(dim_0)
            dim_1_idx = dim_names.index(dim_1)
        except ValueError as e:
            raise ValueError(f"Invalid dimension name. Choose from {dim_names}") from e
        
        # Identify dimensions to keep and reduce
        keep_indices = [dim_0_idx, dim_1_idx]
        reduce_indices = [i for i in range(4) if i not in keep_indices]
        
        # Reduce by taking mean over specified dimensions
        reduced_matrix = np.mean(self.data_cube_fft, axis=tuple(reduce_indices))
        
        # Transpose if necessary to match input order
        if dim_0_idx > dim_1_idx:
            reduced_matrix = reduced_matrix.T
        
        return reduced_matrix
    
    def generate_3channel_image(self, target_size=(64, 64)) -> np.ndarray:
        """
        Generate 3-channel radar image using MMW-HAT visualization method.
        
        Returns:
            3-channel image (C, H, W) where:
            - Channel 0: Range-Doppler
            - Channel 1: Range-Azimuth
            - Channel 2: Range-Elevation
        """
        # Generate three 2D visualizations
        range_doppler = self.vis_2d("Range", "Doppler")
        range_azimuth = self.vis_2d("Range", "Azimuth")
        range_elevation = self.vis_2d("Range", "Elevation")
        
        # Log transform for visualization (add small constant to avoid log(0))
        range_doppler = 10 * np.log10(range_doppler + 1e-10)
        range_azimuth = 10 * np.log10(range_azimuth + 1e-10)
        range_elevation = 10 * np.log10(range_elevation + 1e-10)
        
        # Resize to target size using interpolation
        import scipy.ndimage as ndimage
        
        def resize_channel(img, target_shape):
            if img.shape == target_shape:
                return img
            # Calculate zoom factors
            zoom_factors = (target_shape[0] / img.shape[0], target_shape[1] / img.shape[1])
            return ndimage.zoom(img, zoom_factors, order=1)
        
        range_doppler = resize_channel(range_doppler, target_size)
        range_azimuth = resize_channel(range_azimuth, target_size)
        range_elevation = resize_channel(range_elevation, target_size)
        
        # Stack channels: (C, H, W)
        image = np.stack([range_doppler, range_azimuth, range_elevation], axis=0)
        
        # Normalize to zero mean, unit variance
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        return image.astype(np.float32)
