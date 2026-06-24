import numpy as np
from utility.helper import read_uint12, split_samples, abs2_numba_complex64, parse_radar_cfg
from utility.FFTW import FFTWProcessor

class CubeProcessor:
    def __init__(self, setting, num_doppler_bin=0, num_range_bin=0, min_range=0.2, num_azimuth_bin=16, num_elevation_bin=16, threads=4, mti_alpha=None):
        """
        Initialize the SignalProcessor for creating a 4-D data cube and optionally performing MTI processing.

        Args:
            setting (dict): Configuration settings for the radar.
            num_doppler_bin (int): Number of Doppler bins.
            num_range_bin (int): Number of range bins.
            min_range (float): Minimum range value after processing. Range lower than it will be removed.
            num_azimuth_bin (int): Number of azimuth bins.
            num_elevation_bin (int): Number of elevation bins.
            threads (int): Number of threads for FFT processing.
            mti_alpha (float or None): MTI alpha factor, controlling the decay rate of historical data.
                                       If None, MTI processing is disabled.
        """

        self.radar_param = parse_radar_cfg(setting)

        # Set bin sizes, using defaults if input bin size is smaller
        num_doppler_bin = max(num_doppler_bin, self.radar_param["num_chirps_per_frame"])
        num_range_bin = max(num_range_bin, self.radar_param["num_samples_per_chirp"])
        num_azimuth_bin = max(num_azimuth_bin, self.radar_param["num_azimuth_antennas"])
        num_elevation_bin = max(num_elevation_bin, self.radar_param["num_elevation_antennas"])
        range_bin = np.arange(num_range_bin>>1) * (3e8 / (2 * self.radar_param["bandwidth"]))
        self.range_skip = np.searchsorted(range_bin, min_range)

        # Calculate and store radar parameters in mmw_param
        proc_param = {
            "num_doppler_bin": num_doppler_bin,
            "num_range_bin": num_range_bin,
            "min_range": min_range,
            "doppler_bin": -np.fft.fftshift(np.fft.fftfreq(num_doppler_bin, 1/self.radar_param["chirp_rate"])), # Note that there is a minus sign.
            "range_bin": range_bin[self.range_skip:],
            "num_azimuth_bin": num_azimuth_bin,
            "num_elevation_bin": num_elevation_bin,
            "azimuth_bin": np.rad2deg(np.fft.fftshift(np.arcsin(np.fft.fftfreq(num_azimuth_bin, 0.5)))),
            "elevation_bin": np.rad2deg(np.fft.fftshift(np.arcsin(np.fft.fftfreq(num_elevation_bin, 0.5))))
        }
        self.proc_param = proc_param

        # MTI decay factor
        self.mti_alpha = mti_alpha

        # Position mapping based on radar configuration
        # The datasheet shows:
        # RX2(0,1)          TX
        # RX3(0,0)    RX1(1,0)
        self.position_map = {1: (1, 0), 2: (0, 1), 3: (0, 0)}
        self.active_antennas = [ant for ant in self.position_map if (self.radar_param["rx_mask"] & (1 << (ant - 1)))]

        # Initialize FFTW processor for efficient FFTs
        # Dimensions: Doppler, Range, Azimuth, Elevation
        self.fftw_proc = FFTWProcessor(
            (self.proc_param["num_doppler_bin"], self.proc_param["num_range_bin"], self.proc_param["num_azimuth_bin"], self.proc_param["num_elevation_bin"]),
            axes=(0, 1, 2, 3),
            precision='float32',
            threads=threads
        )

        # Initialize the previous data cube buffer for MTI, if MTI is enabled
        self.previous_data_cube = None
        self.data_cube_fft = None

    def mti_process(self, data_cube):
        """
        Apply Moving Target Indicator (MTI) processing to the raw data cube.

        This method applies a decayed history subtraction to highlight moving objects.

        Args:
            data_cube (np.ndarray): The 4-D raw data cube (Doppler, Range, Azimuth, Elevation).

        Returns:
            np.ndarray: The MTI-processed data cube.
        """
        if self.previous_data_cube is None:
            # Initialize previous data cube with the current frame if no previous frame exists
            self.previous_data_cube = np.copy(data_cube)
            return np.copy(data_cube)
        else:
            # Apply MTI by subtracting the previous data cube
            mti_cube = data_cube - self.previous_data_cube
            # Update the previous data cube with exponential decay
            self.previous_data_cube = self.mti_alpha * data_cube + (1 - self.mti_alpha) * self.previous_data_cube
            return mti_cube

    def process_raw_data(self, raw_data):
        """
        Process raw ADC data to generate a 4-D data cube and perform MTI (if enabled).

        Args:
            raw_data (list): Raw data received from the radar.

        Returns:
            np.ndarray: FFT-processed data cube with optional MTI applied.
        """
        # Parse and split raw ADC data
        adc_data = read_uint12(raw_data)
        adc_data_split = split_samples(adc_data, 1, self.radar_param["num_chirps_per_frame"], self.radar_param["num_samples_per_chirp"], self.radar_param["num_antennas"])

        # Create an empty data cube and place data in appropriate positions for each active antenna
        data_cube = np.zeros((self.proc_param["num_doppler_bin"], self.proc_param["num_range_bin"], self.proc_param["num_azimuth_bin"], self.proc_param["num_elevation_bin"]), dtype=np.float32)
        for i, antenna in enumerate(self.active_antennas):
            pos = self.position_map[antenna]
            data_cube[0:self.radar_param["num_chirps_per_frame"], 0:self.radar_param["num_samples_per_chirp"], pos[0], pos[1]] = adc_data_split[0, :, :, i]

        # Apply MTI processing if mti_alpha is set
        if self.mti_alpha is not None:
            data_cube = self.mti_process(data_cube)

        self.fftw_proc.input_array[:, :, :] = data_cube
        data_cube_fft = self.fftw_proc.run()
        data_cube_fft = np.fft.fftshift(data_cube_fft, axes=(0,2,3))  # Only range bins match with DC
        data_cube_fft = abs2_numba_complex64(data_cube_fft)  # Compute the squared norm of complex values
        data_cube_fft = data_cube_fft[:, self.range_skip:self.proc_param["num_range_bin"] >> 1, :, :]
        self.data_cube_fft = data_cube_fft

    def vis_2d(self, dim_0, dim_1):
        """
        Generates a 2D matrix by reducing the other dimensions with mean, based on specified axes.

        Args:
            dim_0 (str): Name of the dimension for the 0-axis.
            dim_1 (str): Name of the dimension for the 1-axis.

        Returns:
            np.ndarray: The resulting 2D matrix after reducing other dimensions.
        """
        dim_names = ["Doppler", "Range", "Azimuth", "Elevation"]

        # Map dimension names to their indices
        try:
            dim_0_idx = dim_names.index(dim_0)
            dim_1_idx = dim_names.index(dim_1)
        except ValueError as e:
            raise ValueError(f"Invalid dimension name. Choose from {dim_names}") from e

        # Identify the dimensions to keep and the dimensions to reduce
        keep_indices = [dim_0_idx, dim_1_idx]
        reduce_indices = [i for i in range(4) if i not in keep_indices]

        # Reduce the matrix by taking the mean over the specified dimensions
        reduced_matrix = np.mean(self.data_cube_fft, axis=tuple(reduce_indices))

        # Ensure the order of axes matches the function inputs
        if dim_0_idx > dim_1_idx:
            reduced_matrix = reduced_matrix.T  # Transpose if necessary

        return reduced_matrix
