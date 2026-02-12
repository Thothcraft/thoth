import os
import copy
import time
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def make_ml_models():
    """Return 5 sklearn classifiers for ML experiments."""
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('et', et), ('xgb', xgb), ('svm', svm)],
        voting='soft', n_jobs=-1,
    )
    return [
        ('RandomForest', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ('ExtraTrees', ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ('XGBoost', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)),
        ('SVM_RBF', SVC(kernel='rbf', C=10, gamma='scale', random_state=42)),
        ('Ensemble', ensemble),
    ]


def run_ml_experiment(name, train_ds, test_ds=None):
    """Run all 5 ML models on a dataset."""
    results = {}
    for model_name, model in make_ml_models():
        print(f"\n--- {model_name} ---")
        job = TrainingJob(
            model=model, train_dataset=train_ds, test_dataset=test_ds,
            test_size=0.2, batch_size=64, epochs=50, lr=1e-3
        )
        results[model_name] = job.run()
    return results


# =============================================================================
# CSI Subcarrier Selection Mask (HT20 non-STBC, 64 subcarriers / 128 bytes)
# Based on ESP-CSI documentation: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi.html
# Reference: https://github.com/espressif/esp-csi/issues/114
# =============================================================================
# Boolean mask: True = keep subcarrier, False = exclude subcarrier
# Total 64 subcarriers (indices 0-63), selecting 52 valid LLTF subcarriers

CSI_SUBCARRIER_MASK = np.array([
    False,  # 0:  Null guard subcarrier (lower edge protection)
    False,  # 1:  Null guard subcarrier (lower edge protection)
    False,  # 2:  Null guard subcarrier (lower edge protection)
    False,  # 3:  Null guard subcarrier (lower edge protection)
    False,  # 4:  Null guard subcarrier (lower edge protection)
    False,  # 5:  Null guard subcarrier (lower edge protection)
    True,   # 6:  LLTF valid subcarrier (negative frequency, index -26)
    True,   # 7:  LLTF valid subcarrier (negative frequency, index -25)
    True,   # 8:  LLTF valid subcarrier (negative frequency, index -24)
    True,   # 9:  LLTF valid subcarrier (negative frequency, index -23)
    True,   # 10: LLTF valid subcarrier (negative frequency, index -22)
    True,   # 11: LLTF valid subcarrier (negative frequency, index -21)
    True,   # 12: LLTF valid subcarrier (negative frequency, index -20)
    True,   # 13: LLTF valid subcarrier (negative frequency, index -19)
    True,   # 14: LLTF valid subcarrier (negative frequency, index -18)
    True,   # 15: LLTF valid subcarrier (negative frequency, index -17)
    True,   # 16: LLTF valid subcarrier (negative frequency, index -16)
    True,   # 17: LLTF valid subcarrier (negative frequency, index -15)
    True,   # 18: LLTF valid subcarrier (negative frequency, index -14)
    True,   # 19: LLTF valid subcarrier (negative frequency, index -13)
    True,   # 20: LLTF valid subcarrier (negative frequency, index -12)
    True,   # 21: LLTF pilot subcarrier (index -11) - contains channel info
    True,   # 22: LLTF valid subcarrier (negative frequency, index -10)
    True,   # 23: LLTF valid subcarrier (negative frequency, index -9)
    True,   # 24: LLTF valid subcarrier (negative frequency, index -8)
    True,   # 25: LLTF pilot subcarrier (index -7) - contains channel info
    True,   # 26: LLTF valid subcarrier (negative frequency, index -6)
    True,   # 27: LLTF valid subcarrier (negative frequency, index -5)
    True,   # 28: LLTF valid subcarrier (negative frequency, index -4)
    True,   # 29: LLTF valid subcarrier (negative frequency, index -3)
    True,   # 30: LLTF valid subcarrier (negative frequency, index -2)
    True,   # 31: LLTF valid subcarrier (negative frequency, index -1)
    False,  # 32: DC subcarrier (center frequency, always null)
    True,   # 33: LLTF valid subcarrier (positive frequency, index +1)
    True,   # 34: LLTF valid subcarrier (positive frequency, index +2)
    True,   # 35: LLTF valid subcarrier (positive frequency, index +3)
    True,   # 36: LLTF valid subcarrier (positive frequency, index +4)
    True,   # 37: LLTF valid subcarrier (positive frequency, index +5)
    True,   # 38: LLTF valid subcarrier (positive frequency, index +6)
    True,   # 39: LLTF pilot subcarrier (index +7) - contains channel info
    True,   # 40: LLTF valid subcarrier (positive frequency, index +8)
    True,   # 41: LLTF valid subcarrier (positive frequency, index +9)
    True,   # 42: LLTF valid subcarrier (positive frequency, index +10)
    True,   # 43: LLTF pilot subcarrier (index +11) - contains channel info
    True,   # 44: LLTF valid subcarrier (positive frequency, index +12)
    True,   # 45: LLTF valid subcarrier (positive frequency, index +13)
    True,   # 46: LLTF valid subcarrier (positive frequency, index +14)
    True,   # 47: LLTF valid subcarrier (positive frequency, index +15)
    True,   # 48: LLTF valid subcarrier (positive frequency, index +16)
    True,   # 49: LLTF valid subcarrier (positive frequency, index +17)
    True,   # 50: LLTF valid subcarrier (positive frequency, index +18)
    True,   # 51: LLTF valid subcarrier (positive frequency, index +19)
    True,   # 52: LLTF valid subcarrier (positive frequency, index +20)
    True,   # 53: LLTF valid subcarrier (positive frequency, index +21)
    True,   # 54: LLTF valid subcarrier (positive frequency, index +22)
    True,   # 55: LLTF valid subcarrier (positive frequency, index +23)
    True,   # 56: LLTF valid subcarrier (positive frequency, index +24)
    True,   # 57: LLTF valid subcarrier (positive frequency, index +25)
    True,   # 58: LLTF valid subcarrier (positive frequency, index +26)
    False,  # 59: Null guard subcarrier (upper edge protection)
    False,  # 60: Null guard subcarrier (upper edge protection)
    False,  # 61: Null guard subcarrier (upper edge protection)
    False,  # 62: Null guard subcarrier (upper edge protection)
    False,  # 63: Null guard subcarrier (upper edge protection)
], dtype=bool)  # Total: 52 True values (valid subcarriers)


# =============================================================================
# ProcessingBlock base class
# =============================================================================
class ProcessingBlock(ABC):
    """Abstract base class for all CSI processing blocks.

    All blocks operate on a data dictionary, preserving and accumulating errors.
    Subclasses must implement `process(data)`.

    Parameters
    ----------
    verbose : bool
        If True, print detailed information about the processing step including
        input/output shapes, data statistics, and transformation details.
        Useful for debugging and understanding the data flow. Default False.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def _log(self, msg):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"  [{self.__class__.__name__}] {msg}")

    @abstractmethod
    def process(self, data):
        """Process the data dictionary. Must be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, data):
        if isinstance(data, dict):
            if 'errors' not in data:
                data['errors'] = []
            try:
                return self.process(data)
            except Exception as e:
                data['errors'].append(f"{self.__class__.__name__}: {e}")
                return data
        return self.process(data)


# =============================================================================
# CSI Loader
# =============================================================================
class CSI_Loader(ProcessingBlock):
    """Loads CSI data from CSV, converts 128 I/Q values to 64 complex numbers.

    As a ProcessingBlock, it can be the first block in a pipeline.
    Accepts either a filepath string or a data dict with 'filepath' key.

    Parameters
    ----------
    verbose : bool
        If True, print loading progress: file path, total lines read,
        valid lines parsed, skipped lines, and CSI array shape/dtype.
        Shows parsing errors if any. Default False.
    """

    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.filepath = None

    def is_valid(self):
        return (self.filepath is not None
                and os.path.exists(self.filepath)
                and self.filepath.endswith('.csv'))

    def process(self, data):
        # Accept filepath as string or from data dict
        if isinstance(data, str):
            filepath = data
        elif isinstance(data, dict) and 'filepath' in data:
            filepath = data['filepath']
        else:
            raise ValueError(f"CSI_Loader expects filepath string or dict with 'filepath' key, got {type(data)}")

        errors = []
        total_lines = 0
        read_lines = 0
        self.filepath = filepath
        if not self.is_valid():
            raise ValueError(f"Invalid CSI file: {filepath}")

        df = pd.read_csv(filepath, header=0, on_bad_lines='skip')
        total_lines = len(df)

        all_rssi = df['rssi'].values
        raw_csi = df['data'].values

        csi_complex = []
        valid_rssi = []
        for numline, line in enumerate(raw_csi):
            try:
                csi_row = [int(x) for x in line[1:-1].split(",")]
                if len(csi_row) != 128:
                    errors.append(f"Line {numline}: expected 128 values, got {len(csi_row)}")
                    continue
                imag = np.array(csi_row[0::2])
                real = np.array(csi_row[1::2])
                csi_complex.append(real + 1j * imag)
                valid_rssi.append(all_rssi[numline])
                read_lines += 1
            except Exception as e:
                errors.append(f"Line {numline}: {e}")

        csi_complex = np.array(csi_complex)
        rssi = np.array(valid_rssi, dtype=np.float64)

        self._log(f"Loaded {filepath}")
        self._log(f"Total lines: {total_lines}, Valid: {read_lines}, Skipped: {total_lines - read_lines}")
        self._log(f"CSI shape: {csi_complex.shape}, dtype: {csi_complex.dtype}")
        self._log(f"RSSI shape: {rssi.shape}, range: [{rssi.min():.1f}, {rssi.max():.1f}]")
        if errors:
            self._log(f"Errors ({len(errors)}): {errors[:3]}{'...' if len(errors) > 3 else ''}")

        return {
            'csi': csi_complex,
            'rssi': rssi,
            'total_lines': total_lines,
            'read_lines': read_lines,
            'errors': errors,
        }


# =============================================================================
# Processing Blocks
# =============================================================================
class FeatureSelector(ProcessingBlock):
    """Applies a boolean mask to exclude subcarriers/features.

    Parameters
    ----------
    mask : np.ndarray or None
        Boolean mask array. True = keep, False = exclude. Default CSI_SUBCARRIER_MASK.
    key : str
        Data dict key to filter. Default 'csi'.
    verbose : bool
        If True, print mask statistics (total features, kept features, excluded),
        input/output shapes, and which subcarrier indices are kept. Default False.
    """

    def __init__(self, mask=None, key='csi', verbose=False):
        super().__init__(verbose)
        self.mask = mask if mask is not None else CSI_SUBCARRIER_MASK
        self.key = key
        if not isinstance(self.mask, np.ndarray) or self.mask.dtype != bool:
            raise TypeError(f"mask must be boolean np.ndarray, got {type(self.mask)}")

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"data['{self.key}'] must be np.ndarray, got {type(arr)}")
            if arr.ndim != 2 or arr.shape[1] != len(self.mask):
                raise ValueError(f"expected 2D array with {len(self.mask)} cols, got shape {arr.shape}")
            out = arr[:, self.mask]
            self._log(f"Input: {arr.shape} -> Output: {out.shape}")
            self._log(f"Mask: {len(self.mask)} total, {self.mask.sum()} kept, {(~self.mask).sum()} excluded")
            data[self.key] = out
            return data
        if not isinstance(data, np.ndarray):
            raise TypeError(f"expected np.ndarray, got {type(data)}")
        return data[:, self.mask]


class AmplitudeExtractor(ProcessingBlock):
    """Extracts amplitude |z| from complex CSI. Input: complex, Output: float64.

    Parameters
    ----------
    key : str
        Data dict key containing complex CSI. Default 'csi'.
    output_key : str
        Key to store amplitude result. Default 'amplitude'.
    verbose : bool
        If True, print input complex array shape/dtype, output amplitude shape,
        and amplitude statistics (min, max, mean, std). Default False.
    """

    def __init__(self, key='csi', output_key='amplitude', verbose=False):
        super().__init__(verbose)
        self.key = key
        self.output_key = output_key

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray) or not np.iscomplexobj(arr):
                raise TypeError(f"data['{self.key}'] must be complex np.ndarray, got dtype {getattr(arr, 'dtype', type(arr))}")
            amp = np.abs(arr).astype(np.float64)
            self._log(f"Input: {arr.shape} {arr.dtype} -> Output: {amp.shape} {amp.dtype}")
            self._log(f"Amplitude range: [{amp.min():.4f}, {amp.max():.4f}], mean={amp.mean():.4f}, std={amp.std():.4f}")
            data[self.output_key] = amp
            return data
        if not isinstance(data, np.ndarray) or not np.iscomplexobj(data):
            raise TypeError(f"expected complex np.ndarray")
        return np.abs(data).astype(np.float64)


class PhaseExtractor(ProcessingBlock):
    """Extracts phase angle from complex CSI. Input: complex, Output: float64 radians.

    Parameters
    ----------
    key : str
        Data dict key containing complex CSI. Default 'csi'.
    output_key : str
        Key to store phase result. Default 'phase'.
    unwrap : bool
        If True, unwrap phase discontinuities along subcarrier axis. Default False.
    verbose : bool
        If True, print input shape, output shape, phase range in radians,
        and whether unwrapping was applied. Default False.
    """

    def __init__(self, key='csi', output_key='phase', unwrap=False, verbose=False):
        super().__init__(verbose)
        self.key = key
        self.output_key = output_key
        self.unwrap = unwrap

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray) or not np.iscomplexobj(arr):
                raise TypeError(f"data['{self.key}'] must be complex np.ndarray, got dtype {getattr(arr, 'dtype', type(arr))}")
            phase = np.angle(arr).astype(np.float64)
            if self.unwrap:
                phase = np.unwrap(phase, axis=1)
            self._log(f"Input: {arr.shape} -> Output: {phase.shape}")
            self._log(f"Phase range: [{phase.min():.4f}, {phase.max():.4f}] rad, unwrap={self.unwrap}")
            data[self.output_key] = phase
            return data
        if not isinstance(data, np.ndarray) or not np.iscomplexobj(data):
            raise TypeError(f"expected complex np.ndarray")
        phase = np.angle(data).astype(np.float64)
        if self.unwrap:
            phase = np.unwrap(phase, axis=1)
        return phase


class WindowTransformer(ProcessingBlock):
    """Windows (n_samples, features) -> (n_windows, win_len, features) or (n_windows, win_len*features).

    Parameters
    ----------
    window_length : int
        Number of samples per window.
    key : str
        Data dict key to window. Default 'amplitude'.
    mode : str
        'sequential' keeps 3D shape, 'flattened' reshapes to 2D. Default 'sequential'.
    stride : int or None
        Step between windows. Default equals window_length (non-overlapping).
    verbose : bool
        If True, print input samples, window count, output shape, overlap percentage,
        and samples discarded due to incomplete final window. Default False.
    """

    def __init__(self, window_length, key='amplitude', mode='sequential', stride=None, verbose=False):
        super().__init__(verbose)
        if not isinstance(window_length, int) or window_length <= 0:
            raise ValueError(f"window_length must be positive int, got {window_length}")
        if mode not in ('sequential', 'flattened'):
            raise ValueError(f"mode must be 'sequential' or 'flattened', got '{mode}'")
        self.window_length = window_length
        self.key = key
        self.mode = mode
        self.stride = stride if stride is not None else window_length
        if not isinstance(self.stride, int) or self.stride <= 0:
            raise ValueError(f"stride must be positive int, got {self.stride}")

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                raise ValueError(f"data['{self.key}'] must be 2D np.ndarray, got {getattr(arr, 'shape', type(arr))}")
            data[self.key] = self._window(arr)
            return data
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError(f"expected 2D np.ndarray")
        return self._window(data)

    def _window(self, arr):
        n_samples, n_features = arr.shape
        n_windows = (n_samples - self.window_length) // self.stride + 1
        if n_windows <= 0:
            raise ValueError(f"Not enough samples ({n_samples}) for window_length={self.window_length}")
        windows = np.array([arr[i * self.stride : i * self.stride + self.window_length] for i in range(n_windows)])
        if self.mode == 'flattened':
            windows = windows.reshape(n_windows, -1)
        return windows


class FFTTransformer(ProcessingBlock):
    """Applies FFT along time axis. Works on 2D or 3D arrays.

    Parameters
    ----------
    key : str
        Data dict key to transform. Default 'amplitude'.
    mode : str
        Output mode: 'complex', 'magnitude', or 'db'. Default 'magnitude'.
    real_only : bool
        If True, use rfft (real FFT) for efficiency. Default True.
    axis : int
        Axis along which to compute FFT. Default -2 (time axis).
    verbose : bool
        If True, print input/output shapes, frequency bin count, mode used,
        and output value range. Default False.
    """

    def __init__(self, key='amplitude', mode='magnitude', real_only=True, axis=-2, verbose=False):
        super().__init__(verbose)
        if mode not in ('complex', 'magnitude', 'db'):
            raise ValueError(f"mode must be 'complex', 'magnitude', or 'db', got '{mode}'")
        self.key = key
        self.mode = mode
        self.real_only = real_only
        self.axis = axis

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray) or arr.ndim not in (2, 3):
                raise ValueError(f"data['{self.key}'] must be 2D or 3D np.ndarray")
            data[self.key] = self._fft(arr)
            return data
        if not isinstance(data, np.ndarray) or data.ndim not in (2, 3):
            raise ValueError(f"expected 2D or 3D np.ndarray")
        return self._fft(data)

    def _fft(self, arr):
        squeezed = False
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
            squeezed = True
        if self.real_only:
            fft_out = np.fft.rfft(arr, axis=self.axis)
        else:
            fft_out = np.fft.fft(arr, axis=self.axis)
        if self.mode == 'complex':
            result = fft_out
        elif self.mode == 'magnitude':
            result = np.abs(fft_out)
        elif self.mode == 'db':
            result = 20 * np.log10(np.abs(fft_out) + 1e-9)
        if squeezed:
            result = result.squeeze(axis=0)
        self._log(f"FFT: {arr.shape} -> {result.shape}")
        return result


class WindowFlattener(ProcessingBlock):
    """Flattens 3D windowed data to 2D for classification.

    Converts (n_windows, dim1, dim2) -> (n_windows, dim1 * dim2).

    Parameters
    ----------
    key : str
        Data dict key to flatten. Default 'features'.
    verbose : bool
        If True, print input/output shapes. Default False.
    """

    def __init__(self, key='features', verbose=False):
        super().__init__(verbose)
        self.key = key

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"data['{self.key}'] must be np.ndarray, got {type(arr)}")
            if arr.ndim == 3:
                n_windows = arr.shape[0]
                flattened = arr.reshape(n_windows, -1)
                self._log(f"Flatten: {arr.shape} -> {flattened.shape}")
                data[self.key] = flattened
            elif arr.ndim == 2:
                self._log(f"Already 2D: {arr.shape}")
            else:
                raise ValueError(f"Expected 2D or 3D array, got {arr.ndim}D")
            return data
        if isinstance(data, np.ndarray) and data.ndim == 3:
            return data.reshape(data.shape[0], -1)
        return data


class Augmentor(ProcessingBlock):
    """Applies data augmentation to CSI amplitude/phase arrays.

    Augmentation types (all independently toggleable):
    - gaussian_noise: Add Gaussian noise to simulate interference.
    - amplitude_scaling: Random per-sample scaling for distance variations.
    - time_warp: Stretch/compress sequences for temporal drifts.

    Parameters
    ----------
    key : str
        Data dict key to augment. Default 'amplitude'.
    output_key : str or None
        If set, store result under this key instead of overwriting input key.
        This allows keeping original data alongside augmented data.
    gaussian_noise : bool
        Enable Gaussian noise injection. Default True.
    noise_std : float or tuple
        Noise standard deviation. If tuple (lo, hi), sampled uniformly per sample.
        Default (0.1, 0.5).
    amplitude_scaling : bool
        Enable random amplitude scaling. Default True.
    scale_range : tuple
        (min_scale, max_scale) multiplier range. Default (0.8, 1.2).
    time_warp : bool
        Enable time-warping (stretch/compress along time axis). Default True.
    warp_range : tuple
        (min_factor, max_factor) as fractions of original length.
        Default (0.9, 1.1) for ±10%.
    seed : int or None
        Random seed for reproducibility. Default None.
    verbose : bool
        If True, print augmentation parameters and output shape. Default False.
    """

    def __init__(self, key='amplitude', output_key=None, gaussian_noise=True, noise_std=(0.1, 0.5),
                 amplitude_scaling=True, scale_range=(0.8, 1.2),
                 time_warp=True, warp_range=(0.9, 1.1), seed=None, verbose=False):
        self.key = key
        self.output_key = output_key if output_key is not None else key
        self.gaussian_noise = gaussian_noise
        self.noise_std = noise_std
        self.amplitude_scaling = amplitude_scaling
        self.scale_range = scale_range
        self.time_warp = time_warp
        self.warp_range = warp_range
        self.rng = np.random.RandomState(seed)
        super().__init__(verbose)

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"data['{self.key}'] must be np.ndarray, got {type(arr)}")
            aug = self._augment(arr)
            self._log(f"Input: {arr.shape} -> Output: {aug.shape}")
            self._log(f"Augmentations: noise={self.gaussian_noise}, scale={self.amplitude_scaling}, warp={self.time_warp}")
            if self.gaussian_noise:
                self._log(f"  Noise std: {self.noise_std}")
            if self.amplitude_scaling:
                self._log(f"  Scale range: {self.scale_range}")
            data[self.output_key] = aug
            return data
        if not isinstance(data, np.ndarray):
            raise TypeError(f"expected np.ndarray, got {type(data)}")
        return self._augment(data)

    def _augment(self, arr):
        arr = arr.copy()
        if self.gaussian_noise:
            arr = self._add_noise(arr)
        if self.amplitude_scaling:
            arr = self._scale(arr)
        if self.time_warp:
            arr = self._warp(arr)
        return arr

    def _add_noise(self, arr):
        if isinstance(self.noise_std, (tuple, list)):
            std = self.rng.uniform(self.noise_std[0], self.noise_std[1])
        else:
            std = self.noise_std
        noise = self.rng.normal(0, std, size=arr.shape)
        return arr + noise

    def _scale(self, arr):
        lo, hi = self.scale_range
        if arr.ndim == 2:
            scales = self.rng.uniform(lo, hi, size=(arr.shape[0], 1))
        elif arr.ndim == 3:
            scales = self.rng.uniform(lo, hi, size=(arr.shape[0], 1, 1))
        else:
            scales = self.rng.uniform(lo, hi)
        return arr * scales

    def _warp(self, arr):
        from scipy.interpolate import interp1d
        if arr.ndim == 2:
            n_samples, n_features = arr.shape
            factor = self.rng.uniform(self.warp_range[0], self.warp_range[1])
            new_len = max(2, int(round(n_samples * factor)))
            x_old = np.linspace(0, 1, n_samples)
            x_new = np.linspace(0, 1, new_len)
            f = interp1d(x_old, arr, axis=0, kind='linear', fill_value='extrapolate')
            return f(x_new)
        elif arr.ndim == 3:
            n_windows, win_len, n_features = arr.shape
            warped = []
            for i in range(n_windows):
                factor = self.rng.uniform(self.warp_range[0], self.warp_range[1])
                new_len = max(2, int(round(win_len * factor)))
                x_old = np.linspace(0, 1, win_len)
                x_new = np.linspace(0, 1, new_len)
                f = interp1d(x_old, arr[i], axis=0, kind='linear', fill_value='extrapolate')
                warped_win = f(x_new)
                x_back = np.linspace(0, 1, new_len)
                x_orig = np.linspace(0, 1, win_len)
                f_back = interp1d(x_back, warped_win, axis=0, kind='linear', fill_value='extrapolate')
                warped.append(f_back(x_orig))
            return np.array(warped)
        return arr


class STFTTransformer(ProcessingBlock):
    """Applies Short-Time Fourier Transform along the time axis.

    Converts a 2D array (n_samples, n_features) into a 3D spectrogram
    (n_frames, n_freq_bins, n_features) or flattened 2D output.

    Parameters
    ----------
    key : str
        Data dict key to transform. Default 'amplitude'.
    nperseg : int
        Length of each STFT segment (window size). Default 64.
    noverlap : int or None
        Number of overlapping samples between segments. Default nperseg // 2.
    window : str
        Window function name (passed to scipy.signal.stft). Default 'hann'.
    mode : str
        Output mode: 'magnitude', 'power', 'complex', or 'db'. Default 'magnitude'.
    output_key : str or None
        If set, store result under this key instead of overwriting input key.
    verbose : bool
        If True, print input shape, output shape, and STFT parameters. Default False.
    """

    def __init__(self, key='amplitude', nperseg=64, noverlap=None,
                 window='hann', mode='magnitude', output_key=None, verbose=False):
        super().__init__(verbose)
        if mode not in ('complex', 'magnitude', 'power', 'db'):
            raise ValueError(f"mode must be 'complex', 'magnitude', 'power', or 'db', got '{mode}'")
        self.key = key
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.window = window
        self.mode = mode
        self.output_key = output_key if output_key is not None else key

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                raise ValueError(f"data['{self.key}'] must be 2D np.ndarray, got {getattr(arr, 'shape', type(arr))}")
            result = self._stft(arr)
            self._log(f"Input: {arr.shape} -> Output: {result.shape}")
            self._log(f"STFT params: nperseg={self.nperseg}, noverlap={self.noverlap}, window='{self.window}', mode='{self.mode}'")
            self._log(f"Output: (n_frames, n_freq_bins, n_features) = {result.shape}")
            data[self.output_key] = result
            return data
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError(f"expected 2D np.ndarray, got {getattr(data, 'shape', type(data))}")
        return self._stft(data)

    def _stft(self, arr):
        from scipy.signal import stft
        n_samples, n_features = arr.shape
        spectrograms = []
        for col in range(n_features):
            f, t, Zxx = stft(arr[:, col], fs=1.0, window=self.window,
                             nperseg=self.nperseg, noverlap=self.noverlap)
            if self.mode == 'complex':
                spectrograms.append(Zxx)
            elif self.mode == 'magnitude':
                spectrograms.append(np.abs(Zxx))
            elif self.mode == 'power':
                spectrograms.append(np.abs(Zxx) ** 2)
            elif self.mode == 'db':
                spectrograms.append(20 * np.log10(np.abs(Zxx) + 1e-9))
        # Stack: (n_freq_bins, n_time_frames, n_features)
        result = np.stack(spectrograms, axis=-1)
        # Transpose to (n_time_frames, n_freq_bins, n_features)
        result = result.transpose(1, 0, 2)
        return result


class Normalizer(ProcessingBlock):
    """
    Z-score normalizes per subcarrier/feature column (mean=0, std=1).

    Handles amplitude scaling shifts from distance/environment changes.
    Computes statistics along the time axis (axis=0 for 2D, per-window for 3D).

    Parameters
    ----------
    key : str
        Data dict key to normalize. Default 'amplitude'.
    output_key : str or None
        If set, store result under this key instead of overwriting input key.
    eps : float
        Small constant added to std to avoid division by zero. Default 1e-8.
    verbose : bool
        If True, print input shape, output shape, and normalization statistics. Default False.
    """

    def __init__(self, key='amplitude', output_key=None, eps=1e-8, verbose=False):
        super().__init__(verbose)
        self.key = key
        self.output_key = output_key if output_key is not None else key
        self.eps = eps

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"data['{self.key}'] must be np.ndarray, got {type(arr)}")
            norm = self._normalize(arr)
            self._log(f"Input: {arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]")
            self._log(f"Output: {norm.shape}, range=[{norm.min():.4f}, {norm.max():.4f}]")
            self._log(f"Post-norm mean={norm.mean():.6f}, std={norm.std():.6f}")
            data[self.output_key] = norm
            return data
        if not isinstance(data, np.ndarray):
            raise TypeError(f"expected np.ndarray, got {type(data)}")
        return self._normalize(data)

    def _normalize(self, arr):
        if arr.ndim == 2:
            mean = arr.mean(axis=0, keepdims=True)
            std = arr.std(axis=0, keepdims=True)
            return (arr - mean) / (std + self.eps)
        elif arr.ndim == 3:
            mean = arr.mean(axis=1, keepdims=True)
            std = arr.std(axis=1, keepdims=True)
            return (arr - mean) / (std + self.eps)
        return arr


class ConcatBlock(ProcessingBlock):
    """Concatenates multiple keys from one or more data dicts.

    Supports three modes:
    - Single dict: concatenates values of multiple keys along feature axis.
    - List of dicts, axis=-1 (feature concat): takes one key from each dict
      and concatenates along the feature axis. All dicts must have the same
      number of samples (rows).
    - List of dicts, axis=0 (sample concat): takes one key from each dict
      and concatenates along the sample axis (row-wise). All dicts must have
      the same number of features (columns). This is the mode to use when
      joining original + augmented data.

    Parameters
    ----------
    keys : list of str
        Keys to concatenate from data dict(s).
    output_key : str
        Key to store concatenated result. Default 'features'.
    axis : int
        Concatenation axis. -1 for feature concat, 0 for sample concat. Default -1.
    verbose : bool
        If True, print keys being concatenated, individual array shapes,
        output shape, and axis used. Default False.
    """

    def __init__(self, keys, output_key='features', axis=-1, verbose=False):
        super().__init__(verbose)
        if not isinstance(keys, (list, tuple)) or len(keys) < 1:
            raise ValueError("keys must be a non-empty list of strings")
        self.keys = keys
        self.output_key = output_key
        self.axis = axis

    def process(self, data):
        # Case 1: list of dicts — take one key from each and concat
        if isinstance(data, (list, tuple)):
            arrays = []
            errors = []
            for i, d in enumerate(data):
                if not isinstance(d, dict):
                    raise TypeError(f"element {i} must be dict, got {type(d)}")
                errors.extend(d.get('errors', []))
                key = self.keys[i] if i < len(self.keys) else self.keys[0]
                if key not in d:
                    raise KeyError(f"key '{key}' not found in dict {i}")
                arrays.append(d[key])

            # Flatten 3D to 2D if needed before concat
            flat = []
            for a in arrays:
                if a.ndim == 1:
                    flat.append(a.reshape(-1, 1))
                elif a.ndim == 3:
                    flat.append(a.reshape(a.shape[0], -1))
                else:
                    flat.append(a)

            if self.axis == 0:
                # Sample-wise concat: column counts must match
                n_cols = flat[0].shape[1]
                for i, a in enumerate(flat):
                    if a.shape[1] != n_cols:
                        raise ValueError(
                            f"column mismatch: array 0 has {n_cols} cols, "
                            f"array {i} has {a.shape[1]}")
            else:
                # Feature-wise concat: row counts must match
                n_rows = flat[0].shape[0]
                for i, a in enumerate(flat):
                    if a.shape[0] != n_rows:
                        raise ValueError(
                            f"row mismatch: array 0 has {n_rows} rows, "
                            f"array {i} has {a.shape[0]}")

            result = data[0].copy() if isinstance(data[0], dict) else {}
            result[self.output_key] = np.concatenate(flat, axis=self.axis)
            result['errors'] = errors
            return result

        # Case 2: single dict — concat multiple keys from it
        if isinstance(data, dict):
            arrays = []
            for key in self.keys:
                if key not in data:
                    raise KeyError(f"key '{key}' not found in data")
                arr = data[key]
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    elif arr.ndim == 3:
                        arr = arr.reshape(arr.shape[0], -1)
                    arrays.append(arr)
                else:
                    raise TypeError(f"data['{key}'] must be np.ndarray, got {type(arr)}")

            n_rows = arrays[0].shape[0]
            for i, a in enumerate(arrays):
                if a.shape[0] != n_rows:
                    raise ValueError(f"row mismatch: '{self.keys[0]}' has {n_rows} rows, '{self.keys[i]}' has {a.shape[0]}")

            data[self.output_key] = np.concatenate(arrays, axis=self.axis)
            return data

        raise TypeError(f"expected dict or list of dicts, got {type(data)}")


# =============================================================================
# Pipeline
# =============================================================================
class Pipeline:
    """Chains ProcessingBlocks. First block should be CSI_Loader.
    
    Called with a filepath, passes it through all blocks sequentially.
    """

    def __init__(self, blocks):
        self.blocks = blocks
        for b in blocks:
            if not isinstance(b, ProcessingBlock):
                raise TypeError(f"All blocks must be ProcessingBlock, got {type(b)}")
        if len(blocks) == 0 or not isinstance(blocks[0], CSI_Loader):
            raise TypeError("First block must be CSI_Loader")

    def __call__(self, filepath):
        data = filepath  # Start with filepath, CSI_Loader will handle it
        for block in self.blocks:
            data = block(data)
        return data


# =============================================================================
# DatasetFile
# =============================================================================
class DatasetFile:
    """Represents a single CSI file with labels and a processing pipeline.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    pipeline : Pipeline
        Processing pipeline to apply.
    labels : list of str
        Labels for this file (e.g. ['drink', 'activity']).
        First label is primary unless primary_label is set.
    primary_label : str, optional
        Override which label is primary.
    """

    def __init__(self, filepath, pipeline, labels, primary_label=None):
        self.filepath = filepath
        self.pipeline = pipeline
        self.labels = labels if isinstance(labels, list) else [labels]
        self.primary_label = primary_label if primary_label else self.labels[0]
        self._data = None

    def load(self):
        """Apply pipeline and cache result."""
        self._data = self.pipeline(self.filepath)
        return self._data

    @property
    def data(self):
        if self._data is None:
            self.load()
        return self._data


# =============================================================================
# TrainingDataset
# =============================================================================
class TrainingDataset:
    """Aggregates multiple DatasetFile objects into X, y arrays for training.

    Parameters
    ----------
    dataset_files : list of DatasetFile
    feature_key : str
        Key in the processed data dict to use as features. Default 'features'.
    label_map : dict, optional
        Maps label strings to integers. Auto-generated if None.
    balance : bool
        If True, trim oversampled classes to match the smallest class. Default False.
    """

    def __init__(self, dataset_files, feature_key='features', label_map=None, balance=False):
        self.dataset_files = dataset_files
        self.feature_key = feature_key
        self.label_map = label_map
        self.balance = balance
        self._X = None
        self._y = None

    def build(self):
        """Load all files and build X, y arrays."""
        all_X = []
        all_y = []
        all_labels = set()

        for df in self.dataset_files:
            all_labels.add(df.primary_label)

        if self.label_map is None:
            self.label_map = {label: i for i, label in enumerate(sorted(all_labels))}

        for df in self.dataset_files:
            data = df.data
            if self.feature_key not in data:
                raise KeyError(f"feature_key '{self.feature_key}' not found in data for {df.filepath}")
            X = data[self.feature_key]
            if X.ndim == 3:
                X = X.reshape(X.shape[0], -1)
            y_val = self.label_map[df.primary_label]
            y = np.full(X.shape[0], y_val, dtype=np.int64)
            all_X.append(X)
            all_y.append(y)

        self._X = np.concatenate(all_X, axis=0)
        self._y = np.concatenate(all_y, axis=0)

        if self.balance:
            self._balance_classes()

        return self._X, self._y

    def _balance_classes(self):
        """Trim oversampled classes to match the smallest class."""
        classes, counts = np.unique(self._y, return_counts=True)
        min_count = counts.min()
        balanced_idx = []
        rng = np.random.RandomState(42)
        for cls in classes:
            cls_idx = np.where(self._y == cls)[0]
            if len(cls_idx) > min_count:
                cls_idx = rng.choice(cls_idx, size=min_count, replace=False)
            balanced_idx.append(cls_idx)
        balanced_idx = np.concatenate(balanced_idx)
        balanced_idx.sort()
        self._X = self._X[balanced_idx]
        self._y = self._y[balanced_idx]

    @property
    def X(self):
        if self._X is None:
            self.build()
        return self._X

    @property
    def y(self):
        if self._y is None:
            self.build()
        return self._y

    @property
    def num_classes(self):
        return len(self.label_map) if self.label_map else 0

    @property
    def input_size(self):
        return self.X.shape[1] if self._X is not None else 0

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """Index into the dataset. Returns (x, y) tuple."""
        return self.X[idx], self.y[idx]

    def get_torch_dataset(self):
        """Returns a PyTorch TensorDataset."""
        import torch
        from torch.utils.data import TensorDataset
        X_t = torch.FloatTensor(self.X)
        y_t = torch.LongTensor(self.y)
        return TensorDataset(X_t, y_t)


# FederatedPartitioner
class FederatedPartitioner:
    """Partitions a TrainingDataset using Dirichlet distribution (pure numpy).

    Creates non-IID federated data splits without external dependencies.

    Parameters
    ----------
    dataset : TrainingDataset
        The dataset to partition.
    num_partitions : int
        Number of federated partitions (clients).
    alpha : float
        Dirichlet concentration parameter. Lower = more heterogeneous.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, dataset, num_partitions, alpha=0.5, seed=42):
        self.dataset = dataset
        self.num_partitions = num_partitions
        self.alpha = alpha

        rng = np.random.RandomState(seed)
        X, y = dataset.X, dataset.y
        classes = np.unique(y)
        n_classes = len(classes)

        # Dirichlet partitioning: for each class, sample proportions across clients
        partition_indices = [[] for _ in range(num_partitions)]
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet([alpha] * num_partitions)
            # Convert proportions to counts
            counts = (proportions * len(cls_idx)).astype(int)
            # Distribute remainder
            remainder = len(cls_idx) - counts.sum()
            for i in range(remainder):
                counts[i % num_partitions] += 1
            # Assign indices
            start = 0
            for pid in range(num_partitions):
                partition_indices[pid].append(cls_idx[start:start + counts[pid]])
                start += counts[pid]

        self._partition_indices = [np.concatenate(idxs) for idxs in partition_indices]

    def load_partition(self, partition_id):
        """Load a partition as a TrainingDataset-compatible object.

        Parameters
        ----------
        partition_id : int
            Partition index (0 to num_partitions-1).

        Returns
        -------
        TrainingDataset
            A new TrainingDataset containing only the partition's data.
        """
        idx = self._partition_indices[partition_id]
        part_ds = TrainingDataset.__new__(TrainingDataset)
        part_ds.dataset_files = []
        part_ds.feature_key = self.dataset.feature_key
        part_ds.label_map = self.dataset.label_map
        part_ds.balance = False
        part_ds._X = self.dataset.X[idx]
        part_ds._y = self.dataset.y[idx]
        return part_ds


# =============================================================================
# TrainingJob
# =============================================================================
class TrainingJob:
    """Trains a PyTorch nn.Module or sklearn model on a TrainingDataset.

    Accepts either a single dataset (split internally via test_size) or
    separate train_dataset and test_dataset.

    Parameters
    ----------
    model : torch.nn.Module or sklearn estimator
    train_dataset : TrainingDataset
        Training data. Required.
    test_dataset : TrainingDataset, optional
        Separate test data. If provided, test_size is ignored.
    test_size : float
        Fraction of train_dataset to hold out for testing. Default 0.2.
        Ignored when test_dataset is provided.
    batch_size : int
        Batch size for PyTorch training. Default 64.
    epochs : int
        Number of epochs for PyTorch training. Default 50.
    lr : float
        Learning rate for PyTorch training. Default 1e-3.
    device : str
        'cuda' or 'cpu'. Default auto-detect.
    posttraining_pipeline : list of callable, optional
        A list of processing blocks (callables) applied to X_test before
        testing evaluation, after training is complete. Each callable receives
        a 2D np.ndarray and returns a 2D np.ndarray. Default None.
    """
    def __init__(self, model, train_dataset, test_dataset=None, test_size=0.2,
                 batch_size=64, epochs=50, lr=1e-3, device=None,
                 posttraining_pipeline=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.test_size = test_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.posttraining_pipeline = posttraining_pipeline or []
        self.metrics = {}
        self._is_torch = self._check_torch(model)
        if device is None:
            if self._is_torch:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = 'cpu'
        else:
            self.device = device

    @staticmethod
    def _check_torch(model):
        try:
            import torch.nn as nn
            return isinstance(model, nn.Module)
        except ImportError:
            return False

    def _prepare_data(self):
        """Build train/test splits from datasets.

        Time-series aware: for each class, the last test_size fraction of
        samples (in original order) is used for testing. No shuffling or
        stratified random splitting — later data is always test data to
        avoid information leakage.
        """
        if self.test_dataset is not None:
            X_train, y_train = self.train_dataset.X, self.train_dataset.y
            X_test, y_test = self.test_dataset.X, self.test_dataset.y
        else:
            X, y = self.train_dataset.X, self.train_dataset.y
            train_idx = []
            test_idx = []
            for cls in np.unique(y):
                cls_idx = np.where(y == cls)[0]
                n_test = max(1, int(len(cls_idx) * self.test_size))
                split_point = len(cls_idx) - n_test
                train_idx.append(cls_idx[:split_point])
                test_idx.append(cls_idx[split_point:])
            train_idx = np.concatenate(train_idx)
            test_idx = np.concatenate(test_idx)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
        return X_train, X_test, y_train, y_test

    def _apply_posttraining(self, X_test):
        """Apply posttraining pipeline blocks to X_test."""
        for block in self.posttraining_pipeline:
            X_test = block(X_test)
        return X_test

    def run(self):
        """Execute training, apply posttraining pipeline, then evaluate."""
        X_train, X_test, y_train, y_test = self._prepare_data()

        n_total = X_train.shape[0] + X_test.shape[0]
        n_features = X_train.shape[1]
        n_classes = self.train_dataset.num_classes
        print(f"Dataset: {n_total} samples, {n_features} features, {n_classes} classes")
        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        print(f"Label map: {self.train_dataset.label_map}")

        # --- Phase 1: Train ---
        if self._is_torch:
            train_result = self._fit_torch(X_train, y_train)
        else:
            train_result = self._fit_sklearn(X_train, y_train)

        # --- Phase 2: Posttraining pipeline on test data ---
        if self.posttraining_pipeline:
            print(f"Applying posttraining pipeline ({len(self.posttraining_pipeline)} blocks) to test data...")
            X_test = self._apply_posttraining(X_test)
            print(f"Post-pipeline test shape: {X_test.shape}")

        # --- Phase 3: Evaluate ---
        if self._is_torch:
            self.metrics = self._evaluate_torch(X_train, y_train, X_test, y_test, train_result)
        else:
            self.metrics = self._evaluate_sklearn(X_train, y_train, X_test, y_test, train_result)

        return self.metrics

    # ----- sklearn -----
    def _fit_sklearn(self, X_train, y_train):
        print(f"\nTraining sklearn model: {type(self.model).__name__}")
        t0 = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - t0
        return {'train_time_s': round(train_time, 2)}

    def _evaluate_sklearn(self, X_train, y_train, X_test, y_test, train_result):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        avg = 'weighted' if self.train_dataset.num_classes > 2 else 'binary'
        metrics = {
            'model_type': 'sklearn',
            'model_name': type(self.model).__name__,
            'train_time_s': train_result['train_time_s'],
            'train_accuracy': round(accuracy_score(y_train, y_pred_train), 4),
            'test_accuracy': round(accuracy_score(y_test, y_pred_test), 4),
            'test_precision': round(precision_score(y_test, y_pred_test, average=avg, zero_division=0), 4),
            'test_recall': round(recall_score(y_test, y_pred_test, average=avg, zero_division=0), 4),
            'test_f1': round(f1_score(y_test, y_pred_test, average=avg, zero_division=0), 4),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(),
        }
        self._print_metrics(metrics)
        return metrics

    # ----- PyTorch -----
    def _fit_torch(self, X_train, y_train):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device(self.device)
        self.model = self.model.to(device)

        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        print(f"\nTraining PyTorch model on {device} for {self.epochs} epochs...")
        history = {'train_loss': [], 'train_acc': []}

        t0 = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

            train_loss = running_loss / total
            train_acc = correct / total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            if (epoch + 1) % max(1, self.epochs // 10) == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

        train_time = time.time() - t0
        return {'train_time_s': round(train_time, 2), 'history': history}

    def _evaluate_torch(self, X_train, y_train, X_test, y_test, train_result):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        device = torch.device(self.device)
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(y_batch.numpy())

        y_pred = np.array(all_preds)
        y_true = np.array(all_true)
        avg = 'weighted' if self.train_dataset.num_classes > 2 else 'binary'

        history = train_result.get('history', {})
        metrics = {
            'model_type': 'pytorch',
            'model_name': type(self.model).__name__,
            'train_time_s': train_result['train_time_s'],
            'epochs': self.epochs,
            'train_accuracy': round(history['train_acc'][-1], 4) if history.get('train_acc') else None,
            'test_accuracy': round(accuracy_score(y_true, y_pred), 4),
            'test_precision': round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
            'test_recall': round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
            'test_f1': round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'history': history,
        }
        self._print_metrics(metrics)
        return metrics

    @staticmethod
    def _print_metrics(m):
        print(f"\n{'='*50}")
        print(f"Results: {m['model_name']} ({m['model_type']})")
        print(f"{'='*50}")
        print(f"  Train time:     {m['train_time_s']}s")
        print(f"  Train accuracy: {m['train_accuracy']}")
        print(f"  Test accuracy:  {m['test_accuracy']}")
        print(f"  Test precision: {m['test_precision']}")
        print(f"  Test recall:    {m['test_recall']}")
        print(f"  Test F1:        {m['test_f1']}")
        print(f"  Confusion matrix:")
        for row in m['confusion_matrix']:
            print(f"    {row}")


# =============================================================================
# Testing
# =============================================================================
def load_csi_datasets(train_dir, test_dir, window_len, verbose=False):
    """Load and return train/test TrainingDatasets from CSI CSV files."""
    pipeline = Pipeline([
        CSI_Loader(verbose=verbose),
        FeatureSelector(verbose=verbose),
        AmplitudeExtractor(verbose=verbose),
        Normalizer(key='amplitude', output_key='norm', verbose=verbose),
        ConcatBlock(keys=['norm', 'rssi'], output_key='features', verbose=verbose),
        Augmentor(key='features', output_key='aug', gaussian_noise=True, noise_std=(0.1, 0.3),
                  amplitude_scaling=True, scale_range=(0.85, 1.15), time_warp=False, seed=42, verbose=verbose),
        ConcatBlock(keys=['features', 'aug'], output_key='features', axis=0, verbose=verbose),
        WindowTransformer(window_length=window_len, key='features', mode='sequential', verbose=verbose),
        FFTTransformer(key='features', mode='magnitude', real_only=True, axis=-2, verbose=verbose),
        WindowFlattener(key='features', verbose=verbose),
    ])

    test_pipeline = Pipeline([
        CSI_Loader(verbose=verbose),
        FeatureSelector(verbose=verbose),
        AmplitudeExtractor(verbose=verbose),
        Normalizer(key='amplitude', verbose=verbose),
        ConcatBlock(keys=['amplitude', 'rssi'], output_key='features', verbose=verbose),
        WindowTransformer(window_length=window_len, key='features', mode='sequential', verbose=verbose),
        FFTTransformer(key='features', mode='magnitude', real_only=True, axis=-2, verbose=verbose),
        WindowFlattener(key='features', verbose=verbose),
    ])

    labels = ['drink', 'eat', 'empty', 'smoke', 'watch', 'work']
    train_files = {l: f'{train_dir}/{l}.csv' for l in labels}
    test_files  = {l: f'{test_dir}/{l}2.csv' for l in labels}

    ds_files = [DatasetFile(p, pipeline, [l]) for l, p in train_files.items()]
    train_ds = TrainingDataset(ds_files, feature_key='features', balance=True)
    train_ds.build()

    test_ds_files = [DatasetFile(p, test_pipeline, [l]) for l, p in test_files.items()]
    test_ds = TrainingDataset(test_ds_files, feature_key='features', label_map=train_ds.label_map, balance=True)
    test_ds.build()

    return train_ds, test_ds


# =============================================================================
# ML Experiments (sklearn only)
# =============================================================================
if __name__ == '__main__':
    TRAIN_DIR = '../../../wifi_sensing_data/thoth_data/train'
    TEST_DIR = '../../../wifi_sensing_data/thoth_data/test'
    WINDOW_LEN = 1500

    print("=" * 70)
    print("ML EXPERIMENTS (5 sklearn models)")
    print("=" * 70)

    combined_ds, test_ds = load_csi_datasets(TRAIN_DIR, TEST_DIR, WINDOW_LEN, verbose=True)
    n_features, n_classes = combined_ds.X.shape[1], combined_ds.num_classes

    print(f"\n{'='*70}")
    print("DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"  Training samples:  {combined_ds.X.shape[0]}")
    print(f"  Test samples:      {test_ds.X.shape[0]}")
    print(f"  Features:          {n_features}")
    print(f"  Classes:           {n_classes}")
    print(f"  Label map:         {combined_ds.label_map}")
    print(f"  Window length:     {WINDOW_LEN}")
    for label, idx in combined_ds.label_map.items():
        c_tr = (combined_ds.y == idx).sum()
        c_te = (test_ds.y == idx).sum()
        print(f"    {label}: train={c_tr}, test={c_te}")

    # ---- Experiment A: time-series split ----
    print(f"\n{'='*70}")
    print("EXPERIMENT A: Time-series split (last 20% per class)")
    print(f"{'='*70}")
    split_results = run_ml_experiment("split", combined_ds)

    # ---- Experiment B: separate test set ----
    print(f"\n{'='*70}")
    print("EXPERIMENT B: Separate test dataset")
    print(f"{'='*70}")
    separate_results = run_ml_experiment("separate", combined_ds, test_ds)

    # ---- Summary ----
    print(f"\n{'='*80}")
    print(f"{'Model':<20} | {'Split Acc':>9} {'Split F1':>9} | {'Sep Acc':>9} {'Sep F1':>9}")
    print("-" * 70)
    for name in split_results:
        sa, sf = split_results[name]['test_accuracy'], split_results[name]['test_f1']
        ea, ef = separate_results[name]['test_accuracy'], separate_results[name]['test_f1']
        print(f"{name:<20} | {sa:>9.4f} {sf:>9.4f} | {ea:>9.4f} {ef:>9.4f}")

    best_split = max(split_results.items(), key=lambda x: x[1]['test_f1'])
    best_sep   = max(separate_results.items(), key=lambda x: x[1]['test_f1'])
    print(f"\nBest (Split):    {best_split[0]} F1={best_split[1]['test_f1']:.4f}")
    print(f"Best (Separate): {best_sep[0]} F1={best_sep[1]['test_f1']:.4f}")
    print("=" * 70)