import numpy as np
import scipy
import pyfftw
import time


class FFTWProcessor:
    def __init__(self, dims, axes=None, precision='float32', threads=1):
        """
        Initialize the FFTWProcessor with the specified dimensions, axes, precision, and number of threads.

        Args:
            dims (tuple): Dimensions of the data to process.
            axes (tuple): Axes over which to perform the FFT.
            precision (str): Precision of the FFT ('float32' or 'float64').
            threads (int): Number of threads to use for FFT computation.
        """
        self.dims = dims
        self.axes = axes if axes is not None else tuple(range(len(dims)))
        self.threads = threads

        # Set the data type based on precision
        if precision == 'float32':
            self.dtype = np.complex64
        elif precision == 'float64':
            self.dtype = np.complex128
        else:
            raise ValueError("Precision must be 'float32' or 'float64'")

        # Allocate aligned input and output arrays
        self.input_array = pyfftw.empty_aligned(dims, dtype=self.dtype, n=16)
        self.output_array = pyfftw.empty_aligned(dims, dtype=self.dtype, n=16)

        # Create the FFTW object with the specified configuration
        self.fft_object = pyfftw.FFTW(self.input_array, self.output_array, axes=self.axes,
                                      direction='FFTW_FORWARD', threads=self.threads)


    def run(self):
        # Execute the FFT
        self.fft_object()

        # Return the FFT result
        return self.output_array

# Main function to demonstrate and validate the FFTWProcessor class
def main():
    # Set the dimensions for the 4D array
    dims = (64, 128, 16, 16)

    # Generate a random 4D complex array for input
    input_data = (np.random.randn(*dims) + 1j * np.random.randn(*dims)).astype('complex64')

    # Instantiate FFTWProcessor for fast FFT processing
    fft_processor = FFTWProcessor(dims, axes=(0, 1, 2, 3), precision='float32', threads=4)

    fft_processor.input_array[:] = input_data
    # Perform FFT using FFTWProcessor
    fftw_result = fft_processor.run()

    # The error tolerance is not strict. The n-d fft is a sequence-dependent and the error may accumulate.

    # Validate the result with scipy's fft (only for testing, not part of the class)
    scipy_result = scipy.fft.fftn(input_data)
    print("Are the results close?", np.allclose(fftw_result, scipy_result, rtol=1e-02, atol=1e-05))

    # Validate the result with numpy's fft (only for testing, not part of the class)
    np_result = np.fft.fftn(input_data)
    print("Are the results close?", np.allclose(fftw_result, np_result, rtol=1e-02, atol=1e-05))

    # Timing comparison
    def time_function(func, *args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        return (end_time - start_time) * 1e3  # Return time in milliseconds

    # Benchmark NumPy FFT (for validation timing)
    numpy_time_total = 0
    for _ in range(10):
        numpy_time_total += time_function(scipy.fft.fftn, input_data)
    numpy_time_avg = numpy_time_total / 10

    # Benchmark pyFFTW FFT with pre-planning
    pyfftw_time_total = 0
    for _ in range(10):
        pyfftw_time_total += time_function(fft_processor.run)
    pyfftw_time_avg = pyfftw_time_total / 10

    # Print the timing results
    print(f"Average time taken by scipy FFT: {numpy_time_avg:.3f} ms")
    print(f"Average time taken by pyFFTW FFT with pre-planning: {pyfftw_time_avg:.3f} ms")


if __name__ == "__main__":
    main()
