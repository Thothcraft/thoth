import json
import math
from collections import deque

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter

from internal.DBF import DBF
from internal.doppler import DopplerAlgo
from internal.fft_spectrum import fft_spectrum


class _Track:
    def __init__(self, track_id, detection):
        self.id = track_id
        self.position = detection["position"].copy()
        self.velocity = np.zeros(3, dtype=float)
        self.dimensions = detection["dimensions"].copy()
        self.dimension_history = deque([self.dimensions.copy()], maxlen=15)
        self.snr_db = detection["snr_db"]
        self.normalized_peak = detection["normalized_peak"]
        self.radial_velocity_mps = detection["radial_velocity_mps"]
        self.hits = 1
        self.hit_streak = 1
        self.misses = 0
        self.age = 1
        self.confidence = 0.42
        self.confirmed = False
        self.presence_mode = "motion"

    def predict(self, dt):
        self.position += self.velocity * dt
        self.age += 1

    def update(self, detection, dt, alpha, beta):
        residual = detection["position"] - self.position
        self.position += alpha * residual
        if dt > 0:
            self.velocity += beta * residual / dt
        speed = float(np.linalg.norm(self.velocity))
        if speed > 1.5:
            self.velocity *= 1.5 / speed
        self.dimension_history.append(detection["dimensions"].copy())
        self.dimensions = np.median(np.array(self.dimension_history), axis=0)
        self.snr_db += 0.3 * (detection["snr_db"] - self.snr_db)
        self.normalized_peak += 0.3 * (
            detection["normalized_peak"] - self.normalized_peak
        )
        self.radial_velocity_mps += 0.3 * (
            detection["radial_velocity_mps"] - self.radial_velocity_mps
        )
        self.hits += 1
        self.hit_streak += 1
        self.misses = 0
        self.confidence = min(1.0, self.confidence + 0.3)
        self.presence_mode = "motion"

    def hold_static(self, position, normalized_peak):
        self.position += 0.08 * (position - self.position)
        self.velocity *= 0.2
        self.misses = 0
        self.confidence = min(1.0, self.confidence + 0.025)
        self.presence_mode = "static"
        self.normalized_peak += 0.08 * (normalized_peak - self.normalized_peak)

    def miss(self):
        self.misses += 1
        self.hit_streak = 0
        self.confidence *= 0.97
        self.velocity *= 0.55


class SigProc:
    """Three-RX range, azimuth, elevation processing and person-level tracking."""

    def __init__(self, processing_config_fn, radar_config):
        with open(processing_config_fn, "r") as file:
            self.processing_config = json.load(file)
        self.radar_config = radar_config

        if radar_config["rx_mask"] != 7 or radar_config["num_antennas"] != 3:
            raise ValueError("3D tracking requires RX1, RX2 and RX3 (rx_mask=7)")

        self.num_azimuth_beams = int(self.processing_config["num_azimuth_beams"])
        self.num_elevation_beams = int(self.processing_config["num_elevation_beams"])
        self.max_azimuth_deg = float(self.processing_config["max_azimuth_deg"])
        self.max_elevation_deg = float(self.processing_config["max_elevation_deg"])

        c = 3e8
        samples = int(radar_config["num_samples_per_chirp"])
        bandwidth = float(radar_config["bandwidth"])
        self.max_range_m = c / (2 * bandwidth) * samples / 2
        self.range_bin = np.arange(samples) * self.max_range_m / samples
        self.azimuth_bin = np.linspace(
            -self.max_azimuth_deg, self.max_azimuth_deg, self.num_azimuth_beams
        )
        self.elevation_bin = np.linspace(
            -self.max_elevation_deg, self.max_elevation_deg, self.num_elevation_beams
        )

        chirps = int(radar_config["num_chirps_per_frame"])
        self.doppler_frequency_bin = np.fft.fftshift(
            np.fft.fftfreq(2 * chirps, d=1.0 / radar_config["chirp_rate"])
        )
        center_frequency_hz = float(self.processing_config.get("center_frequency_hz", 59e9))
        self.velocity_bin = self.doppler_frequency_bin * (c / center_frequency_hz) / 2.0

        self.dead_zone = float(self.processing_config["dead_zone"])
        detection_cfg = self.processing_config["detection"]
        shadow_cfg = self.processing_config["motion_shadow"]
        cluster_cfg = self.processing_config["object_cluster"]
        tracking_cfg = self.processing_config["tracking"]
        self.normalized_threshold = float(
            detection_cfg.get("normalized_threshold", 0.45)
        )
        self.cfar_training_range = int(detection_cfg.get("cfar_training_range", 17))
        self.cfar_training_angle = int(detection_cfg.get("cfar_training_angle", 17))
        self.cfar_guard_range = int(detection_cfg.get("cfar_guard_range", 5))
        self.cfar_guard_angle = int(detection_cfg.get("cfar_guard_angle", 7))
        self.max_candidates = int(detection_cfg.get("max_candidates", 12))
        self.max_targets = int(detection_cfg.get("max_targets", 4))
        self.min_peak_separation_m = float(detection_cfg.get("min_peak_separation_m", 0.45))
        self.min_peak_separation_deg = float(detection_cfg.get("min_peak_separation_deg", 12.0))
        self.person_merge_distance_m = float(detection_cfg.get("person_merge_distance_m", 0.9))
        self.extent_drop_db = float(detection_cfg.get("extent_drop_db", 6.0))
        self.shadow_floor_percentile = float(shadow_cfg.get("floor_percentile", 60.0))
        self.shadow_peak_percentile = float(shadow_cfg.get("peak_percentile", 99.5))
        self.shadow_threshold = float(shadow_cfg.get("display_threshold", 0.28))
        self.shadow_max_points = int(shadow_cfg.get("max_points_per_frame", 240))
        self.cluster_threshold = float(cluster_cfg.get("intensity_threshold", 0.55))
        self.cluster_min_cells = int(cluster_cfg.get("min_cells", 7))
        self.cluster_min_intensity = float(cluster_cfg.get("min_integrated_intensity", 4.0))
        self.confirm_hits = int(tracking_cfg.get("confirmation_hits", 3))
        self.tentative_max_misses = int(tracking_cfg.get("tentative_max_misses", 2))
        self.max_misses = int(tracking_cfg.get("max_misses", 20))
        self.association_distance_m = float(tracking_cfg.get("association_distance_m", 0.95))
        self.track_merge_distance_m = float(tracking_cfg.get("track_merge_distance_m", 2.0))
        self.track_alpha = float(tracking_cfg.get("alpha", 0.34))
        self.track_beta = float(tracking_cfg.get("beta", 0.06))
        self.frame_period_s = 1.0 / float(radar_config["frame_rate"])

        self._tracks = []
        self._next_track_id = 1
        self.last_detection = {
            "detected": False,
            "targets": [],
            "noise_floor_db": None,
            "threshold_normalized": self.normalized_threshold,
        }
        self.last_motion_shadow = {
            "points": np.empty((0, 3), dtype=float),
            "intensity": np.empty(0, dtype=float),
        }
        self.last_intensity_views = {
            "range_m": self.range_bin.copy(),
            "azimuth_deg": self.azimuth_bin.copy(),
            "elevation_deg": self.elevation_bin.copy(),
            "xy": np.zeros((len(self.range_bin), len(self.azimuth_bin)), dtype=float),
            "yz": np.zeros((len(self.range_bin), len(self.elevation_bin)), dtype=float),
        }
        self._motion_strength_ema = None

        self.doppler = DopplerAlgo(radar_config, 3)
        self.azimuth_dbf = DBF(
            2,
            num_beams=self.num_azimuth_beams,
            max_angle_degrees=self.max_azimuth_deg,
        )
        self.elevation_dbf = DBF(
            2,
            num_beams=self.num_elevation_beams,
            max_angle_degrees=self.max_elevation_deg,
        )
        self.system_mode = "precision"
        self._rd_spectrum = np.empty(
            (samples, 2 * chirps, 3), dtype=complex
        )
        self._static_spectrum = np.empty(
            (samples, chirps, 3), dtype=complex
        )

    def set_system_mode(self, mode):
        """Trade angular resolution for latency without dropping radar frames."""
        selected = mode if mode in {"responsive", "balanced", "precision"} else "balanced"
        beam_count = {"responsive": 17, "balanced": 25, "precision": 49}[selected]
        if selected == self.system_mode and beam_count == self.num_azimuth_beams:
            return
        self.system_mode = selected
        self.num_azimuth_beams = beam_count
        self.num_elevation_beams = beam_count
        self.azimuth_bin = np.linspace(
            -self.max_azimuth_deg, self.max_azimuth_deg, beam_count
        )
        self.elevation_bin = np.linspace(
            -self.max_elevation_deg, self.max_elevation_deg, beam_count
        )
        self.azimuth_dbf = DBF(
            2, num_beams=beam_count, max_angle_degrees=self.max_azimuth_deg
        )
        self.elevation_dbf = DBF(
            2, num_beams=beam_count, max_angle_degrees=self.max_elevation_deg
        )

    @staticmethod
    def _display_intensity(energy):
        """Normalize one range-angle response using the MMW-HAT display scale."""
        energy_db = 20.0 * np.log10(np.maximum(energy, np.finfo(float).tiny))
        finite = energy_db[np.isfinite(energy_db)]
        if not finite.size:
            return np.zeros_like(energy_db, dtype=float)
        floor = float(np.percentile(finite, 55.0))
        ceiling = float(np.percentile(finite, 99.5))
        span = max(1.0, ceiling - floor)
        return np.clip((energy_db - floor) / span, 0.0, 1.0)

    def range_angle_products(self, frame):
        rd_spectrum = self._rd_spectrum
        static_spectrum = self._static_spectrum
        for antenna in range(3):
            rd_spectrum[:, :, antenna] = self.doppler.compute_doppler_map(
                frame[antenna], antenna
            )
            static_spectrum[:, :, antenna] = fft_spectrum(
                frame[antenna], self.doppler.range_window
            ).T

        # FIFO order for rx_mask=7 is RX1, RX2, RX3. The package array is
        # L-shaped: RX1/RX3 measure azimuth; RX2/RX3 measure elevation.
        azimuth_cube = self.azimuth_dbf.run(rd_spectrum[:, :, [0, 2]])
        elevation_cube = self.elevation_dbf.run(rd_spectrum[:, :, [1, 2]])
        static_azimuth_cube = self.azimuth_dbf.run(static_spectrum[:, :, [0, 2]])
        static_elevation_cube = self.elevation_dbf.run(static_spectrum[:, :, [1, 2]])
        azimuth_energy = np.linalg.norm(azimuth_cube, axis=1) / math.sqrt(
            self.num_azimuth_beams
        )
        static_energy = np.linalg.norm(static_azimuth_cube, axis=1) / math.sqrt(
            self.num_azimuth_beams
        )
        return (
            azimuth_energy,
            azimuth_cube,
            elevation_cube,
            static_energy,
            static_azimuth_cube,
            static_elevation_cube,
        )

    @staticmethod
    def _robust_noise_floor_db(energy_db):
        finite = energy_db[np.isfinite(energy_db)]
        if finite.size == 0:
            return -300.0
        cutoff = np.percentile(finite, 70)
        return float(np.median(finite[finite <= cutoff]))

    def _candidate_peaks(
        self, energy_db, noise_floor_db, azimuth_cube, elevation_cube, motion_strength
    ):
        start = int(np.searchsorted(self.range_bin, self.dead_zone))
        power = np.power(10.0, energy_db / 10.0)
        training_size = (self.cfar_training_range, self.cfar_training_angle)
        guard_size = (self.cfar_guard_range, self.cfar_guard_angle)
        training_cells = training_size[0] * training_size[1]
        guard_cells = guard_size[0] * guard_size[1]
        training_sum = uniform_filter(power, size=training_size, mode="nearest") * training_cells
        guard_sum = uniform_filter(power, size=guard_size, mode="nearest") * guard_cells
        local_noise = np.maximum(
            (training_sum - guard_sum) / max(1, training_cells - guard_cells),
            np.finfo(float).tiny,
        )
        local_snr_db = 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny) / local_noise)
        self.last_cfar_peak_db = float(np.max(local_snr_db[start:, 2:-2]))
        # Occupancy is gated on the same normalized temporal map rendered by
        # /presence. CFAR SNR remains diagnostic metadata only.
        local_maxima = motion_strength == maximum_filter(
            motion_strength, size=(3, 7), mode="nearest"
        )
        peak_mask = local_maxima & (motion_strength >= self.normalized_threshold)
        peak_mask[:start] = False
        peak_mask[:, :2] = False
        peak_mask[:, -2:] = False
        peaks = []
        for range_idx, azimuth_idx in zip(*np.nonzero(peak_mask)):
            peaks.append(
                (
                    float(motion_strength[range_idx, azimuth_idx]),
                    float(local_snr_db[range_idx, azimuth_idx]),
                    float(energy_db[range_idx, azimuth_idx]),
                    int(range_idx),
                    int(azimuth_idx),
                )
            )

        selected = []
        for normalized_peak, local_snr, peak_db, range_idx, azimuth_idx in sorted(peaks, reverse=True):
            range_m = float(self.range_bin[range_idx])
            azimuth_deg = float(self.azimuth_bin[azimuth_idx])
            if any(
                abs(range_m - item["range_m"]) < self.min_peak_separation_m
                and abs(azimuth_deg - item["azimuth_deg"]) < self.min_peak_separation_deg
                for item in selected
            ):
                continue
            detection = self._measure_detection(
                energy_db,
                noise_floor_db,
                peak_db,
                range_idx,
                azimuth_idx,
                azimuth_cube,
                elevation_cube,
                local_snr,
                motion_strength,
                normalized_peak,
            )
            if (
                detection["cluster_cells"] >= self.cluster_min_cells
                and detection["cluster_intensity"] >= self.cluster_min_intensity
            ):
                selected.append(detection)
            if len(selected) >= self.max_candidates:
                break
        return self._merge_person_detections(selected)[:self.max_targets]

    @staticmethod
    def _connected_component(mask, seed):
        component = np.zeros_like(mask, dtype=bool)
        stack = [seed]
        while stack:
            row, column = stack.pop()
            if (
                row < 0
                or column < 0
                or row >= mask.shape[0]
                or column >= mask.shape[1]
                or component[row, column]
                or not mask[row, column]
            ):
                continue
            component[row, column] = True
            stack.extend(
                ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1))
            )
        return component

    def _measure_detection(
        self,
        energy_db,
        noise_floor_db,
        peak_db,
        range_idx,
        azimuth_idx,
        azimuth_cube,
        elevation_cube,
        local_snr,
        motion_strength,
        normalized_peak,
    ):
        range_radius = max(2, int(math.ceil(0.55 / (self.range_bin[1] - self.range_bin[0]))))
        azimuth_radius = max(
            2, int(math.ceil(14.0 / (self.azimuth_bin[1] - self.azimuth_bin[0])))
        )
        r0, r1 = max(0, range_idx - range_radius), min(
            len(self.range_bin), range_idx + range_radius + 1
        )
        a0, a1 = max(0, azimuth_idx - azimuth_radius), min(
            self.num_azimuth_beams, azimuth_idx + azimuth_radius + 1
        )
        patch = energy_db[r0:r1, a0:a1]
        footprint_drop_db = min(self.extent_drop_db, max(1.0, local_snr - 1.0))
        footprint_level = peak_db - footprint_drop_db
        component = self._connected_component(
            patch >= footprint_level, (range_idx - r0, azimuth_idx - a0)
        )
        local_r, local_a = np.nonzero(component)
        if local_r.size == 0:
            local_r = np.array([range_idx - r0])
            local_a = np.array([azimuth_idx - a0])
            component[range_idx - r0, azimuth_idx - a0] = True
        ranges = self.range_bin[local_r + r0]
        azimuths = np.deg2rad(self.azimuth_bin[local_a + a0])
        weights = np.maximum(
            np.power(10.0, (patch[component] - peak_db) / 10.0),
            np.finfo(float).tiny,
        )
        range_m = float(np.average(ranges, weights=weights))
        azimuth_deg = float(
            np.rad2deg(np.average(azimuths, weights=weights))
        )

        # Couple elevation to the same range-Doppler cell as the azimuth peak.
        doppler_profile = np.abs(azimuth_cube[range_idx, :, azimuth_idx])
        doppler_idx = int(np.argmax(doppler_profile))
        elevation_profile = np.abs(elevation_cube[range_idx, doppler_idx, :])
        elevation_idx = int(np.argmax(elevation_profile))
        elevation_peak = float(elevation_profile[elevation_idx])
        elevation_mask = elevation_profile >= elevation_peak * 10 ** (-self.extent_drop_db / 20.0)
        left = elevation_idx
        right = elevation_idx
        while left > 0 and elevation_mask[left - 1]:
            left -= 1
        while right + 1 < len(elevation_mask) and elevation_mask[right + 1]:
            right += 1
        elevation_weights = np.maximum(elevation_profile[left:right + 1], np.finfo(float).tiny)
        elevation_deg = float(
            np.average(self.elevation_bin[left:right + 1], weights=elevation_weights)
        )

        azimuth_rad = math.radians(azimuth_deg)
        elevation_rad = math.radians(elevation_deg)
        horizontal_range = range_m * math.cos(elevation_rad)
        position = np.array(
            [
                horizontal_range * math.sin(azimuth_rad),
                horizontal_range * math.cos(azimuth_rad),
                range_m * math.sin(elevation_rad),
            ],
            dtype=float,
        )

        lateral = ranges * np.cos(elevation_rad) * np.sin(azimuths)
        vertical_angles = np.deg2rad(self.elevation_bin[left:right + 1])
        vertical = range_m * np.sin(vertical_angles)
        range_resolution = float(self.range_bin[1] - self.range_bin[0])
        dimensions = np.array(
            [
                max(0.15, float(np.ptp(lateral)) if lateral.size > 1 else 0.15),
                max(range_resolution, float(np.ptp(ranges)) if ranges.size > 1 else range_resolution),
                max(0.2, float(np.ptp(vertical)) if vertical.size > 1 else 0.2),
            ],
            dtype=float,
        )
        dimensions = np.clip(dimensions, [0.2, 0.15, 0.35], [0.95, 0.8, 2.2])

        motion_patch = motion_strength[r0:r1, a0:a1]
        peak_strength = float(motion_strength[range_idx, azimuth_idx])
        cluster_level = min(
            self.cluster_threshold,
            max(0.08, peak_strength * 0.55),
            peak_strength * 0.9,
        )
        if peak_strength <= np.finfo(float).eps:
            motion_component = np.zeros_like(motion_patch, dtype=bool)
        else:
            motion_component = self._connected_component(
                motion_patch >= cluster_level, (range_idx - r0, azimuth_idx - a0)
            )
        cluster_values = motion_patch[motion_component]
        return {
            "position": position,
            "dimensions": dimensions,
            "range_m": range_m,
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "snr_db": float(local_snr),
            "normalized_peak": float(normalized_peak),
            "radial_velocity_mps": float(self.velocity_bin[doppler_idx]),
            "cluster_cells": int(cluster_values.size),
            "cluster_intensity": float(np.sum(cluster_values)),
        }

    def _merge_person_detections(self, detections):
        groups = []
        for detection in sorted(
            detections, key=lambda item: item["normalized_peak"], reverse=True
        ):
            group = next(
                (
                    candidate
                    for candidate in groups
                    if np.linalg.norm(candidate[0]["position"][:2] - detection["position"][:2])
                    <= self.person_merge_distance_m
                ),
                None,
            )
            if group is None:
                groups.append([detection])
            else:
                group.append(detection)

        merged = []
        for group in groups:
            weights = np.array([10 ** (item["snr_db"] / 10.0) for item in group])
            positions = np.array([item["position"] for item in group])
            center = np.average(positions, axis=0, weights=weights)
            lower = np.min(
                [item["position"] - item["dimensions"] / 2 for item in group], axis=0
            )
            upper = np.max(
                [item["position"] + item["dimensions"] / 2 for item in group], axis=0
            )
            dimensions = np.maximum(upper - lower, [0.15, self.range_bin[1] - self.range_bin[0], 0.2])
            dimensions = np.clip(dimensions, [0.2, 0.15, 0.35], [0.95, 0.8, 2.2])
            slant_range = float(np.linalg.norm(center))
            merged.append(
                {
                    "position": center,
                    "dimensions": dimensions,
                    "range_m": slant_range,
                    "azimuth_deg": float(math.degrees(math.atan2(center[0], center[1]))),
                    "elevation_deg": float(
                        math.degrees(math.atan2(center[2], math.hypot(center[0], center[1])))
                    ),
                    "snr_db": max(item["snr_db"] for item in group),
                    "normalized_peak": max(item["normalized_peak"] for item in group),
                    "radial_velocity_mps": float(
                        np.average(
                            [item["radial_velocity_mps"] for item in group], weights=weights
                        )
                    ),
                    "cluster_cells": int(sum(item["cluster_cells"] for item in group)),
                    "cluster_intensity": float(
                        sum(item["cluster_intensity"] for item in group)
                    ),
                }
            )
        return sorted(merged, key=lambda item: item["normalized_peak"], reverse=True)

    def _static_track_support(self, track, motion_strength, static_elevation_cube):
        lateral, forward, vertical = track.position
        range_m = float(np.linalg.norm(track.position))
        if range_m < self.dead_zone or range_m > float(self.processing_config["max_range_m"]):
            return None
        azimuth_deg = math.degrees(math.atan2(lateral, forward))
        range_idx = int(np.argmin(np.abs(self.range_bin - range_m)))
        azimuth_idx = int(np.argmin(np.abs(self.azimuth_bin - azimuth_deg)))
        range_step = float(self.range_bin[1] - self.range_bin[0])
        angle_step = float(self.azimuth_bin[1] - self.azimuth_bin[0])
        range_half = max(2, int(math.ceil((0.2 + 0.045 * range_m) / range_step)))
        angle_half = max(2, int(math.ceil((5.0 + 0.8 * range_m) / angle_step)))
        r0, r1 = max(0, range_idx - range_half), min(
            len(self.range_bin), range_idx + range_half + 1
        )
        a0, a1 = max(0, azimuth_idx - angle_half), min(
            len(self.azimuth_bin), azimuth_idx + angle_half + 1
        )
        patch = motion_strength[r0:r1, a0:a1]
        if patch.size == 0:
            return None
        peak_local = np.unravel_index(int(np.argmax(patch)), patch.shape)
        peak_range_idx = r0 + peak_local[0]
        peak_azimuth_idx = a0 + peak_local[1]
        normalized_peak = float(motion_strength[peak_range_idx, peak_azimuth_idx])
        if normalized_peak < self.normalized_threshold:
            return None

        elevation_profile = np.linalg.norm(
            static_elevation_cube[peak_range_idx], axis=0
        )
        elevation_idx = int(np.argmax(elevation_profile))
        measured_range = float(self.range_bin[peak_range_idx])
        measured_azimuth = math.radians(float(self.azimuth_bin[peak_azimuth_idx]))
        measured_elevation = math.radians(float(self.elevation_bin[elevation_idx]))
        horizontal = measured_range * math.cos(measured_elevation)
        position = np.array(
            [
                horizontal * math.sin(measured_azimuth),
                horizontal * math.cos(measured_azimuth),
                measured_range * math.sin(measured_elevation),
            ]
        )
        return position, normalized_peak

    def _update_tracks(self, detections, motion_strength, static_elevation_cube):
        dt = self.frame_period_s
        for track in self._tracks:
            track.predict(dt)

        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_detections = set(range(len(detections)))
        pairs = []
        for track_idx, track in enumerate(self._tracks):
            for detection_idx, detection in enumerate(detections):
                delta = track.position - detection["position"]
                distance = float(math.sqrt(delta[0] ** 2 + delta[1] ** 2 + 0.25 * delta[2] ** 2))
                if distance <= self.association_distance_m:
                    pairs.append((distance, track_idx, detection_idx))

        for _, track_idx, detection_idx in sorted(pairs):
            if track_idx not in unmatched_tracks or detection_idx not in unmatched_detections:
                continue
            self._tracks[track_idx].update(
                detections[detection_idx], dt, self.track_alpha, self.track_beta
            )
            unmatched_tracks.remove(track_idx)
            unmatched_detections.remove(detection_idx)

        static_holds = set()
        for track_idx in list(unmatched_tracks):
            track = self._tracks[track_idx]
            if not track.confirmed:
                continue
            support = self._static_track_support(
                track, motion_strength, static_elevation_cube
            )
            if support is None:
                continue
            position, normalized_peak = support
            track.hold_static(position, normalized_peak)
            static_holds.add(track_idx)
        unmatched_tracks -= static_holds
        for track_idx in unmatched_tracks:
            self._tracks[track_idx].miss()
        for detection_idx in unmatched_detections:
            self._tracks.append(_Track(self._next_track_id, detections[detection_idx]))
            self._next_track_id += 1
        for track in self._tracks:
            if not track.confirmed and track.hit_streak >= self.confirm_hits and track.confidence >= 0.7:
                track.confirmed = True
        self._tracks = [
            track
            for track in self._tracks
            if (
                (track.confirmed and track.misses <= self.max_misses and track.confidence >= 0.05)
                or (not track.confirmed and track.misses <= self.tentative_max_misses)
            )
        ]

        # A person often creates several separated body reflections. If those
        # reflections survive temporal confirmation, fold the newer track into
        # the oldest nearby identity instead of presenting duplicate people.
        confirmed_tracks = sorted(
            (track for track in self._tracks if track.confirmed), key=lambda track: track.id
        )
        duplicate_ids = set()
        for index, keeper in enumerate(confirmed_tracks):
            if keeper.id in duplicate_ids:
                continue
            for duplicate in confirmed_tracks[index + 1:]:
                if duplicate.id in duplicate_ids:
                    continue
                separation = float(np.linalg.norm(keeper.position[:2] - duplicate.position[:2]))
                if separation > self.track_merge_distance_m:
                    continue
                if duplicate.misses < keeper.misses:
                    keeper.position += 0.45 * (duplicate.position - keeper.position)
                    keeper.velocity += 0.3 * (duplicate.velocity - keeper.velocity)
                keeper.dimension_history.extend(duplicate.dimension_history)
                keeper.dimensions = np.median(np.array(keeper.dimension_history), axis=0)
                keeper.confidence = max(keeper.confidence, duplicate.confidence)
                keeper.misses = min(keeper.misses, duplicate.misses)
                duplicate_ids.add(duplicate.id)
        if duplicate_ids:
            self._tracks = [track for track in self._tracks if track.id not in duplicate_ids]

        confirmed = []
        for track in self._tracks:
            if not track.confirmed:
                continue
            lateral, forward, vertical = track.position
            width, depth, height = track.dimensions
            confirmed.append(
                {
                    "id": track.id,
                    "lateral_m": float(lateral),
                    "forward_m": float(forward),
                    "vertical_m": float(vertical),
                    "range_m": float(np.linalg.norm(track.position)),
                    "azimuth_deg": float(math.degrees(math.atan2(lateral, forward))),
                    "elevation_deg": float(
                        math.degrees(math.atan2(vertical, math.hypot(lateral, forward)))
                    ),
                    "width_m": float(width),
                    "depth_m": float(depth),
                    "height_m": float(height),
                    "speed_mps": float(np.linalg.norm(track.velocity)),
                    "velocity_lateral_mps": float(track.velocity[0]),
                    "velocity_forward_mps": float(track.velocity[1]),
                    "velocity_vertical_mps": float(track.velocity[2]),
                    "radial_velocity_mps": float(track.radial_velocity_mps),
                    "snr_db": float(track.snr_db),
                    "normalized_peak": float(track.normalized_peak),
                    "coasting": bool(track.misses > 0),
                    "presence_mode": track.presence_mode,
                }
            )
        return sorted(confirmed, key=lambda item: item["range_m"])

    def _motion_shadow(self, energy_db, azimuth_cube, elevation_cube):
        start = int(np.searchsorted(self.range_bin, self.dead_zone))
        stop = int(
            np.searchsorted(
                self.range_bin,
                min(float(self.processing_config["max_range_m"]), self.max_range_m),
            )
        )
        valid = energy_db[start:stop]
        floor = float(np.percentile(valid, self.shadow_floor_percentile))
        peak = float(np.percentile(valid, self.shadow_peak_percentile))
        scale = max(peak - floor, 1e-6)
        strength = np.clip((energy_db - floor) / scale, 0.0, 1.0)
        strength = gaussian_filter(strength, sigma=(1.1, 0.9), mode="nearest")
        if self._motion_strength_ema is None:
            self._motion_strength_ema = strength
        else:
            self._motion_strength_ema += 0.42 * (strength - self._motion_strength_ema)
        strength = self._motion_strength_ema.copy()
        mask = np.zeros_like(strength, dtype=bool)
        mask[start:stop] = strength[start:stop] >= self.shadow_threshold
        indices = np.argwhere(mask)
        if len(indices) > self.shadow_max_points:
            values = strength[indices[:, 0], indices[:, 1]]
            keep = np.argpartition(values, -self.shadow_max_points)[-self.shadow_max_points:]
            indices = indices[keep]

        if not len(indices):
            return np.empty((0, 3), dtype=float), np.empty(0, dtype=float), strength

        range_indices = indices[:, 0]
        azimuth_indices = indices[:, 1]
        doppler_profiles = np.abs(azimuth_cube[range_indices, :, azimuth_indices])
        doppler_indices = np.argmax(doppler_profiles, axis=1)
        elevation_profiles = np.abs(
            elevation_cube[range_indices, doppler_indices, :]
        )
        elevation_indices = np.argmax(elevation_profiles, axis=1)
        ranges = self.range_bin[range_indices]
        azimuths = np.deg2rad(self.azimuth_bin[azimuth_indices])
        elevations = np.deg2rad(self.elevation_bin[elevation_indices])
        horizontal = ranges * np.cos(elevations)
        points = np.column_stack((
            horizontal * np.sin(azimuths),
            horizontal * np.cos(azimuths),
            ranges * np.sin(elevations),
        ))
        intensities = strength[range_indices, azimuth_indices]
        return points, intensities, strength

    def update(self, frame):
        (
            azimuth_energy,
            azimuth_cube,
            elevation_cube,
            static_energy,
            static_azimuth_cube,
            static_elevation_cube,
        ) = self.range_angle_products(frame)
        energy_db = 20.0 * np.log10(np.maximum(azimuth_energy, np.finfo(float).tiny))
        shadow_points, shadow_intensity, motion_strength = self._motion_shadow(
            energy_db, azimuth_cube, elevation_cube
        )
        elevation_energy = np.linalg.norm(elevation_cube, axis=1) / math.sqrt(
            self.num_elevation_beams
        )
        self.last_intensity_views = {
            "range_m": self.range_bin.copy(),
            "azimuth_deg": self.azimuth_bin.copy(),
            "elevation_deg": self.elevation_bin.copy(),
            "xy": motion_strength,
            "yz": self._display_intensity(elevation_energy),
        }
        start = int(np.searchsorted(self.range_bin, self.dead_zone))
        noise_floor_db = self._robust_noise_floor_db(energy_db[start:])
        detections = self._candidate_peaks(
            energy_db, noise_floor_db, azimuth_cube, elevation_cube, motion_strength
        )
        targets = self._update_tracks(
            detections, motion_strength, static_elevation_cube
        )
        self.last_detection = {
            "detected": bool(targets),
            "targets": targets,
            "noise_floor_db": noise_floor_db,
            "threshold_normalized": self.normalized_threshold,
            "normalized_peak": max(
                (target["normalized_peak"] for target in targets), default=0.0
            ),
            "candidate_count": len(detections),
            "cfar_peak_db": self.last_cfar_peak_db,
            "motion_points": len(shadow_points),
            "static_targets": sum(
                1 for target in targets if target["presence_mode"] == "static"
            ),
        }
        self.last_motion_shadow = {
            "points": shadow_points,
            "intensity": shadow_intensity,
        }
        return targets
