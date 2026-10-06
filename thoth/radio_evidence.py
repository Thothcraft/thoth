"""Bounded radio summaries on the existing observation/v1 uplink.

Raw IQ remains local. RSSI and CSI are link evidence, never a position fix.
Each physical receiver remains a separate source even on the same host.
"""
from collections import OrderedDict
import math
import re
import time

from .observations import Observation


class RadioEvidenceBridge:
    def __init__(self, node_id, interval_s=5.0, capacity=4096):
        self.node_id = node_id
        self.interval_s = interval_s
        self.capacity = capacity
        self._latest = OrderedDict()
        self._seen = {}
        self._last_flush = 0.0

    def consume(self, window, now=None):
        now = time.time() if now is None else now
        for sensor_id, samples in window.samples.items():
            for sample in samples:
                kind = sample.payload_type
                if kind not in ('ble_scan', 'wifi_scan', 'csi_raw'):
                    continue
                p = sample.payload
                mac = str(p.get('mac') or p.get('addr') or '').lower().replace('-', ':')
                if not re.fullmatch(r'(?:[0-9a-f]{2}:){5}[0-9a-f]{2}', mac):
                    continue
                try:
                    ts, rssi = float(sample.timestamp), float(p['rssi'])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(ts) or not math.isfinite(rssi) or not -127 <= rssi <= 0 or not now - 30 <= ts <= now + 2:
                    continue
                key = (sensor_id, kind, mac)
                if ts <= self._seen.get(key, 0):
                    continue
                self._seen[key] = ts
                radio = 'ble' if kind == 'ble_scan' else 'wifi'
                schema = {'ble_scan': 'radio.ble.v1', 'wifi_scan': 'radio.wifi.v1', 'csi_raw': 'radio.csi.v1'}[kind]
                metadata = p.get('observation') or {}
                value = {
                    'observer': self.node_id, 'component_id': sensor_id,
                    'target': f'{radio}:{mac}', 'mac': mac, 'radio': radio,
                    'measurement': 'csi' if kind == 'csi_raw' else 'rssi',
                    'rssi_dbm': rssi, 'channel': p.get('channel'),
                    'stream': metadata.get('stream'),
                    'quality': metadata.get('quality'),
                    'name': p.get('name') or p.get('ssid'),
                    'position_status': 'unlocated',
                }
                obs = Observation(schema, sensor_id, value, subject=f'{radio}:{mac}',
                    observer=self.node_id, timestamp=ts,
                    units={'rssi_dbm': 'dBm'},
                    provenance={'adapter': 'radio-evidence/v1', 'node_id': self.node_id,
                                'component_id': sensor_id, 'transport': p.get('source', 'usb_serial'),
                                'local_raw_retained': kind == 'csi_raw'})
                self._latest[key] = obs
                # Existing pending keys keep their order so noisy peers cannot starve others.
                while len(self._latest) > self.capacity:
                    self._latest.popitem(last=False)
        self._seen = {key: ts for key, ts in self._seen.items() if now - ts <= 30}
        if len(self._seen) > self.capacity:
            self._seen = dict(sorted(self._seen.items(), key=lambda kv: kv[1])[-self.capacity:])
        if now - self._last_flush < self.interval_s:
            return []
        self._last_flush = now
        out = []
        while self._latest and len(out) < 200:
            _, observation = self._latest.popitem(last=False)
            if now - observation.timestamp <= 30:
                out.append(observation)
        return out
