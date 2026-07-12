import json
import math
import os
import sys
import time
from collections import Counter, defaultdict, deque

import numpy as np
import pyqtgraph.opengl as gl
from PyQt5 import QtCore, QtGui, QtWidgets

from radar_dev import RadarDev
from signal_proc import SigProc

os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"

RADAR_CONFIG = "../radar_config/config_3rx_3m/BGT60TR13C_settings_20241101-104314.json"
PROCESSING_CONFIG = "config/processing_config_advanced.json"
ROOM_CONFIG = "config/room_config.json"
STATE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../config/radar_occupancy.json")
)
SENSOR_FOV_RANGE_M = 15.0
TARGET_COLORS = [
    (91, 224, 207, 210),
    (255, 190, 92, 210),
    (104, 179, 255, 210),
    (240, 116, 140, 210),
]


class InteractiveGLView(gl.GLViewWidget):
    def __init__(self, handles_fn, move_fn, release_fn):
        super().__init__()
        self._handles_fn = handles_fn
        self._move_fn = move_fn
        self._release_fn = release_fn
        self._drag_handle = None
        self._last_mouse = None

    def project_point(self, point):
        viewport = (0, 0, max(1, self.width()), max(1, self.height()))
        projected = (self.projectionMatrix(viewport, viewport) * self.viewMatrix()).map(
            QtGui.QVector3D(*point)
        )
        return np.array(
            [
                (projected.x() + 1.0) * self.width() / 2.0,
                (1.0 - projected.y()) * self.height() / 2.0,
            ]
        )

    def world_amount_for_screen_delta(self, origin, axis, delta):
        start = self.project_point(origin)
        end = self.project_point(np.asarray(origin) + np.asarray(axis))
        pixels_per_meter = end - start
        denominator = float(np.dot(pixels_per_meter, pixels_per_meter))
        if denominator < 1e-6:
            return 0.0
        return float(np.dot(delta, pixels_per_meter) / denominator)

    def mousePressEvent(self, event):
        mouse = np.array([event.pos().x(), event.pos().y()], dtype=float)
        nearest = None
        nearest_distance = 18.0
        for name, point in self._handles_fn().items():
            distance = float(np.linalg.norm(mouse - self.project_point(point)))
            if distance < nearest_distance:
                nearest = name
                nearest_distance = distance
        if nearest is None:
            super().mousePressEvent(event)
            return
        self._drag_handle = nearest
        self._last_mouse = mouse
        self.setCursor(QtCore.Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_handle is None:
            super().mouseMoveEvent(event)
            return
        mouse = np.array([event.pos().x(), event.pos().y()], dtype=float)
        self._move_fn(self._drag_handle, mouse - self._last_mouse)
        self._last_mouse = mouse
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag_handle is None:
            super().mouseReleaseEvent(event)
            return
        self._drag_handle = None
        self._last_mouse = None
        self.unsetCursor()
        self._release_fn()
        event.accept()


class Worker(QtCore.QObject):
    update_signal = QtCore.pyqtSignal(dict)

    def __init__(self, radar, sig_proc):
        super().__init__()
        self.radar = radar
        self.sig_proc = sig_proc
        self._running = True

    @QtCore.pyqtSlot()
    def stop(self):
        self._running = False

    @QtCore.pyqtSlot()
    def run(self):
        while self._running:
            frame = self.radar.get_next_frame()
            if frame is None:
                continue
            targets = self.sig_proc.update(frame)
            self.update_signal.emit(
                {
                    "targets": targets,
                    "detection": dict(self.sig_proc.last_detection),
                    "motion_shadow": {
                        "points": self.sig_proc.last_motion_shadow["points"].copy(),
                        "intensity": self.sig_proc.last_motion_shadow["intensity"].copy(),
                    },
                    "timestamp": time.monotonic(),
                }
            )


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Occupancy Localization")
        self.resize(1500, 900)

        with open(PROCESSING_CONFIG, "r") as file:
            self.processing_config = json.load(file)
        with open(ROOM_CONFIG, "r") as file:
            self.room_config = json.load(file)
        self._room_config_mtime = os.path.getmtime(ROOM_CONFIG)
        self._last_state_write = 0.0

        self._frame_count = 0
        self._first_target_logged = False
        self._last_target_ids = None
        self._fps_times = deque(maxlen=50)
        self._histories = defaultdict(lambda: deque(maxlen=55))
        self._target_items = {}
        self._environment_items = []
        self._latest_targets = {}
        self._display_states = {}
        self._latest_timestamp = 0.0
        self._shadow_frames = deque()
        self._motion_visible = False
        self._pose_history = defaultdict(lambda: deque(maxlen=9))

        radar = RadarDev(9575, RADAR_CONFIG)
        sig_proc = SigProc(PROCESSING_CONFIG, radar.cfg)
        self._build_ui()
        self._build_scene()

        self.render_timer = QtCore.QTimer(self)
        self.render_timer.setInterval(33)
        self.render_timer.timeout.connect(self._render_targets)
        self.render_timer.start()

        radar.open_radar_device()
        self.thread = QtCore.QThread()
        self.worker = Worker(radar, sig_proc)
        self.worker.moveToThread(self.thread)
        self.worker.update_signal.connect(self.update_gui)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def _build_ui(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0a0f11; color: #e6edef; }
            QFrame#panel { background: #11181b; border: 1px solid #263237; border-radius: 10px; }
            QLabel#eyebrow { color: #809399; font-size: 11px; font-weight: 600; }
            QLabel#title { color: #f4f7f8; font-size: 21px; font-weight: 600; }
            QLabel#status { color: #9fb0b6; font-size: 13px; }
            QLabel#value { color: #e8edef; font-size: 18px; font-weight: 600; }
            QLabel#caption { color: #75878e; font-size: 11px; }
            QTreeWidget { background: transparent; border: 0; color: #dce4e7; font-size: 11px; }
            QTreeWidget::item { height: 28px; border-bottom: 1px solid #202a2e; }
            QHeaderView::section { background: #172024; color: #8fa0a6; border: 0; padding: 5px; }
            QDoubleSpinBox, QComboBox { background: #0d1417; border: 1px solid #2b3a40; border-radius: 4px; padding: 4px; }
            QPushButton { background: #1d7774; border: 0; border-radius: 5px; padding: 7px; font-weight: 600; }
            QPushButton:hover { background: #258b87; }
            """
        )
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        self.view = InteractiveGLView(
            self._drag_handles,
            self._drag_handle_moved,
            self._drag_finished,
        )
        self.view.setBackgroundColor("#080d0f")
        root.addWidget(self.view, stretch=5)

        panel = QtWidgets.QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(470)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(20, 20, 20, 18)
        panel_layout.setSpacing(7)

        eyebrow = QtWidgets.QLabel("XENSIV 60 GHz · 3 RX")
        eyebrow.setObjectName("eyebrow")
        panel_layout.addWidget(eyebrow)
        title = QtWidgets.QLabel("3D occupancy")
        title.setObjectName("title")
        panel_layout.addWidget(title)
        self.status_label = QtWidgets.QLabel("Waiting for sensor frames")
        self.status_label.setObjectName("status")
        panel_layout.addWidget(self.status_label)

        summary = QtWidgets.QHBoxLayout()
        self.target_count = self._metric("--", "CONFIRMED OBJECTS")
        self.frame_rate = self._metric("--", "SENSOR / DISPLAY")
        summary.addWidget(self.target_count["widget"])
        summary.addWidget(self.frame_rate["widget"])
        panel_layout.addLayout(summary)

        room_heading = QtWidgets.QLabel("PHYSICAL SPACE")
        room_heading.setObjectName("eyebrow")
        panel_layout.addWidget(room_heading)
        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(5)
        self.room_controls = {}
        for key, label, minimum, maximum in (
            ("width_m", "Width", 1.0, 20.0),
            ("depth_m", "Depth", 1.0, 20.0),
            ("height_m", "Height", 1.5, 8.0),
        ):
            control = self._room_spin(key, minimum, maximum)
            self.room_controls[key] = control
            form.addRow(label, control)
        self.wall_control = QtWidgets.QComboBox()
        self.wall_control.addItems(["Back", "Front", "Left", "Right"])
        self.wall_control.setCurrentText(self.room_config["sensor_wall"])
        form.addRow("Mounting wall", self.wall_control)
        self.room_controls["sensor_position_m"] = self._room_spin(
            "sensor_position_m", 0.0, 20.0
        )
        form.addRow("Position on wall", self.room_controls["sensor_position_m"])
        self.room_controls["sensor_height_m"] = self._room_spin(
            "sensor_height_m", 0.1, 8.0
        )
        form.addRow("Mounting height", self.room_controls["sensor_height_m"])
        self.room_controls["max_object_height_m"] = self._room_spin(
            "max_object_height_m", 0.5, 3.0
        )
        form.addRow("Maximum object height", self.room_controls["max_object_height_m"])
        self.room_controls["max_object_width_m"] = self._room_spin(
            "max_object_width_m", 0.2, 3.0
        )
        form.addRow("Maximum object width", self.room_controls["max_object_width_m"])
        self.room_controls["max_object_depth_m"] = self._room_spin(
            "max_object_depth_m", 0.15, 3.0
        )
        form.addRow("Maximum object depth", self.room_controls["max_object_depth_m"])
        self.room_controls["max_lying_length_m"] = self._room_spin(
            "max_lying_length_m", 0.8, 3.0
        )
        form.addRow("Maximum lying length", self.room_controls["max_lying_length_m"])
        self.floor_anchor_control = QtWidgets.QCheckBox("Floor-anchored standing objects")
        self.floor_anchor_control.setChecked(bool(self.room_config["floor_anchored_targets"]))
        form.addRow("Box model", self.floor_anchor_control)
        panel_layout.addLayout(form)
        apply_button = QtWidgets.QPushButton("Apply room geometry")
        apply_button.clicked.connect(self._apply_room_geometry)
        panel_layout.addWidget(apply_button)

        drag_note = QtWidgets.QLabel(
            "Drag the white handle to move the sensor. Drag amber handles to resize "
            "width, depth, and height. Rotate the view from empty space."
        )
        drag_note.setWordWrap(True)
        drag_note.setObjectName("caption")
        panel_layout.addWidget(drag_note)

        tracks_heading = QtWidgets.QLabel("TRACKED BOXES")
        tracks_heading.setObjectName("eyebrow")
        panel_layout.addWidget(tracks_heading)
        self.target_table = QtWidgets.QTreeWidget()
        self.target_table.setColumnCount(7)
        self.target_table.setHeaderLabels(["ID", "Pose", "Range", "W", "D", "H", "SNR"])
        self.target_table.setRootIsDecorated(False)
        self.target_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.target_table.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.target_table.header().setStretchLastSection(True)
        panel_layout.addWidget(self.target_table, stretch=1)

        note = QtWidgets.QLabel(
            "Dimensions are robust response-envelope estimates. Confirmed tracks coast through "
            "brief dropouts; objects outside the configured room are excluded."
        )
        note.setWordWrap(True)
        note.setObjectName("caption")
        panel_layout.addWidget(note)

        specs = QtWidgets.QLabel(
            "CAPABILITY  58–63.5 GHz · 5 GHz BW · 3 cm · 0.1–15 m\n"
            "FOV  40° H × 65° V · 3 × 12-bit ADC · 1 TX / 3 RX\n"
            "ACTIVE  3 RX · 58–60 GHz · 2 GHz BW · 7.5 cm · 4.8 m · 10 Hz"
        )
        specs.setObjectName("caption")
        panel_layout.addWidget(specs)
        root.addWidget(panel)

    def _room_spin(self, key, minimum, maximum):
        control = QtWidgets.QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(2)
        control.setSingleStep(0.1)
        control.setSuffix(" m")
        control.setValue(float(self.room_config[key]))
        return control

    @staticmethod
    def _metric(value, caption):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 4, 16, 4)
        value_label = QtWidgets.QLabel(value)
        value_label.setObjectName("value")
        caption_label = QtWidgets.QLabel(caption)
        caption_label.setObjectName("caption")
        layout.addWidget(value_label)
        layout.addWidget(caption_label)
        return {"widget": widget, "value": value_label}

    def _build_scene(self):
        self._rebuild_environment(reset_camera=True)
        self.motion_shadow_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=float),
            size=np.empty(0, dtype=float),
            color=np.empty((0, 4), dtype=float),
            pxMode=True,
            glOptions="translucent",
        )
        self.view.addItem(self.motion_shadow_item)

    def _sensor_pose(self):
        width = self.room_config["width_m"]
        depth = self.room_config["depth_m"]
        position = self.room_config["sensor_position_m"]
        height = self.room_config["sensor_height_m"]
        wall = self.room_config["sensor_wall"]
        if wall == "Back":
            return np.array([position, 0.0, height]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        if wall == "Front":
            return np.array([position, depth, height]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])
        if wall == "Left":
            return np.array([0.0, position, height]), np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
        return np.array([width, position, height]), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0])

    def _world_from_local(self, point):
        origin, lateral_axis, forward_axis = self._sensor_pose()
        return origin + lateral_axis * point[0] + forward_axis * point[1] + np.array([0.0, 0.0, point[2]])

    def _world_vector_from_local(self, vector):
        _, lateral_axis, forward_axis = self._sensor_pose()
        return lateral_axis * vector[0] + forward_axis * vector[1] + np.array([0.0, 0.0, vector[2]])

    def _inside_room(self, point):
        return (
            0.0 <= point[0] <= self.room_config["width_m"]
            and 0.0 <= point[1] <= self.room_config["depth_m"]
            and 0.0 <= point[2] <= self.room_config["height_m"]
        )

    def _inside_floorplan(self, point):
        return (
            0.0 <= point[0] <= self.room_config["width_m"]
            and 0.0 <= point[1] <= self.room_config["depth_m"]
        )

    def _clip_ray_to_room(self, origin, direction, maximum_distance):
        limits = np.array(
            [self.room_config["width_m"], self.room_config["depth_m"], self.room_config["height_m"]]
        )
        distance = maximum_distance
        for axis in range(3):
            if direction[axis] > 1e-9:
                distance = min(distance, (limits[axis] - origin[axis]) / direction[axis])
            elif direction[axis] < -1e-9:
                distance = min(distance, -origin[axis] / direction[axis])
        return origin + direction * max(0.0, distance)

    @staticmethod
    def _local_ray(azimuth_deg, elevation_deg):
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)
        horizontal = math.cos(elevation)
        return np.array(
            [horizontal * math.sin(azimuth), horizontal * math.cos(azimuth), math.sin(elevation)]
        )

    def _add_environment_item(self, item):
        self._environment_items.append(item)
        self.view.addItem(item)

    def _drag_handles(self):
        width = self.room_config["width_m"]
        depth = self.room_config["depth_m"]
        height = self.room_config["height_m"]
        sensor, _, _ = self._sensor_pose()
        return {
            "sensor": sensor,
            "room_width": np.array([width, depth * 0.5, 0.08]),
            "room_depth": np.array([width * 0.5, depth, 0.08]),
            "room_height": np.array([width * 0.5, depth * 0.5, height]),
        }

    def _drag_handle_moved(self, name, delta):
        handles = self._drag_handles()
        origin = handles[name]
        if name == "room_width":
            amount = self.view.world_amount_for_screen_delta(origin, [1, 0, 0], delta)
            self.room_config["width_m"] = float(
                np.clip(self.room_config["width_m"] + amount, 1.0, 20.0)
            )
        elif name == "room_depth":
            amount = self.view.world_amount_for_screen_delta(origin, [0, 1, 0], delta)
            self.room_config["depth_m"] = float(
                np.clip(self.room_config["depth_m"] + amount, 1.0, 20.0)
            )
        elif name == "room_height":
            amount = self.view.world_amount_for_screen_delta(origin, [0, 0, 1], delta)
            self.room_config["height_m"] = float(
                np.clip(self.room_config["height_m"] + amount, 1.5, 8.0)
            )
        else:
            wall = self.room_config["sensor_wall"]
            wall_axis = np.array([1.0, 0.0, 0.0]) if wall in ("Back", "Front") else np.array([0.0, 1.0, 0.0])
            position_amount = self.view.world_amount_for_screen_delta(origin, wall_axis, delta)
            height_amount = self.view.world_amount_for_screen_delta(origin, [0, 0, 1], delta)
            wall_length = (
                self.room_config["width_m"]
                if wall in ("Back", "Front")
                else self.room_config["depth_m"]
            )
            self.room_config["sensor_position_m"] = float(
                np.clip(
                    self.room_config["sensor_position_m"] + position_amount,
                    0.0,
                    wall_length,
                )
            )
            self.room_config["sensor_height_m"] = float(
                np.clip(
                    self.room_config["sensor_height_m"] + height_amount,
                    0.1,
                    self.room_config["height_m"],
                )
            )
        self.room_config["sensor_position_m"] = min(
            self.room_config["sensor_position_m"],
            self.room_config["width_m"]
            if self.room_config["sensor_wall"] in ("Back", "Front")
            else self.room_config["depth_m"],
        )
        self.room_config["sensor_height_m"] = min(
            self.room_config["sensor_height_m"], self.room_config["height_m"]
        )
        self._sync_room_controls()
        self._rebuild_environment(reset_camera=False)

    def _drag_finished(self):
        self._save_room_config()

    def _sync_room_controls(self):
        for key, control in self.room_controls.items():
            control.blockSignals(True)
            control.setValue(float(self.room_config[key]))
            control.blockSignals(False)
        self.wall_control.setCurrentText(self.room_config["sensor_wall"])

    def _save_room_config(self):
        with open(ROOM_CONFIG, "w") as file:
            json.dump(self.room_config, file, indent=4)
            file.write("\n")
        self._room_config_mtime = os.path.getmtime(ROOM_CONFIG)

    def _reload_room_config_if_changed(self):
        try:
            modified = os.path.getmtime(ROOM_CONFIG)
            if modified <= self._room_config_mtime:
                return
            with open(ROOM_CONFIG, "r") as file:
                self.room_config = json.load(file)
            self._room_config_mtime = modified
            self._sync_room_controls()
            self._histories.clear()
            self._pose_history.clear()
            self._rebuild_environment(reset_camera=False)
        except (OSError, ValueError, KeyError):
            return

    def _write_occupancy_state(self, targets, shadow_points):
        now = time.time()
        if now - self._last_state_write < 0.09:
            return
        self._last_state_write = now
        serialized_targets = []
        for target in targets:
            local = np.array(
                [target["lateral_m"], target["forward_m"], target["vertical_m"]]
            )
            world = self._world_from_local(local)
            if target["floor_anchored"]:
                world[2] = target["display_height_m"] / 2
            world_size = self._world_box_size(
                [target["width_m"], target["depth_m"], target["display_height_m"]]
            )
            serialized_targets.append(
                {
                    "id": target["id"],
                    "pose": target["pose"],
                    "presence_mode": target["presence_mode"],
                    "range_m": round(float(target["range_m"]), 3),
                    "position": [round(float(value), 3) for value in world],
                    "size": [round(float(value), 3) for value in world_size],
                    "snr_db": round(float(target["snr_db"]), 2),
                }
            )
        state = {
            "updated_at": now,
            "occupied": bool(serialized_targets),
            "target_count": len(serialized_targets),
            "room": self.room_config,
            "fov": {
                "horizontal_deg": 40.0,
                "vertical_deg": 65.0,
                "range_m": SENSOR_FOV_RANGE_M,
            },
            "targets": serialized_targets,
            "shadow": [
                [round(float(value), 3) for value in point]
                for point in shadow_points[:: max(1, len(shadow_points) // 80)]
            ][:80],
        }
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        temporary = f"{STATE_FILE}.tmp"
        with open(temporary, "w") as file:
            json.dump(state, file, separators=(",", ":"))
        os.replace(temporary, STATE_FILE)

    def _rebuild_environment(self, reset_camera=False):
        for item in self._environment_items:
            self.view.removeItem(item)
        self._environment_items.clear()

        width = self.room_config["width_m"]
        depth = self.room_config["depth_m"]
        height = self.room_config["height_m"]
        room = gl.GLBoxItem(color=(112, 134, 141, 105), glOptions="translucent")
        room.setSize(x=width, y=depth, z=height)
        self._add_environment_item(room)

        floor = gl.GLGridItem(color=(71, 89, 95, 50))
        floor.setSize(width, depth, 1)
        floor.setSpacing(1, 1, 1)
        floor.translate(width / 2, depth / 2, 0)
        self._add_environment_item(floor)

        handles = self._drag_handles()
        handle_positions = np.array(list(handles.values()))
        handle_colors = np.array(
            [
                [0.9, 0.96, 0.97, 1.0],
                [0.98, 0.68, 0.25, 0.95],
                [0.98, 0.68, 0.25, 0.95],
                [0.98, 0.68, 0.25, 0.95],
            ]
        )
        marker = gl.GLScatterPlotItem(
            pos=handle_positions,
            size=np.array([13, 11, 11, 11]),
            color=handle_colors,
            pxMode=True,
        )
        self._add_environment_item(marker)

        origin = handles["sensor"]

        max_range = SENSOR_FOV_RANGE_M
        max_azimuth = float(self.processing_config["max_azimuth_deg"])
        max_elevation = float(self.processing_config["max_elevation_deg"])
        corners = []
        for azimuth, elevation in (
            (-max_azimuth, -max_elevation),
            (max_azimuth, -max_elevation),
            (max_azimuth, max_elevation),
            (-max_azimuth, max_elevation),
        ):
            direction = self._world_vector_from_local(self._local_ray(azimuth, elevation))
            corners.append(origin + direction * max_range)
        segments = []
        for corner in corners:
            segments.extend((origin, corner))
        for index in range(4):
            segments.extend((corners[index], corners[(index + 1) % 4]))
        cone = gl.GLLinePlotItem(
            pos=np.array(segments),
            color=(0.31, 0.75, 0.76, 0.48),
            width=1.25,
            mode="lines",
            antialias=True,
        )
        self._add_environment_item(cone)

        if reset_camera:
            self.view.opts["center"] = QtGui.QVector3D(width / 2, depth / 2, height / 2)
            self.view.setCameraPosition(
                distance=max(width, depth, height) * 1.65,
                elevation=25,
                azimuth=-55,
            )

    def _apply_room_geometry(self):
        for key, control in self.room_controls.items():
            self.room_config[key] = float(control.value())
        self.room_config["sensor_wall"] = self.wall_control.currentText()
        self.room_config["floor_anchored_targets"] = self.floor_anchor_control.isChecked()
        wall_length = (
            self.room_config["width_m"]
            if self.room_config["sensor_wall"] in ("Back", "Front")
            else self.room_config["depth_m"]
        )
        self.room_config["sensor_position_m"] = min(
            wall_length, self.room_config["sensor_position_m"]
        )
        self.room_config["sensor_height_m"] = min(
            self.room_config["height_m"], self.room_config["sensor_height_m"]
        )
        self.room_controls["sensor_position_m"].setValue(self.room_config["sensor_position_m"])
        self.room_controls["sensor_height_m"].setValue(self.room_config["sensor_height_m"])
        self._save_room_config()
        self._histories.clear()
        self._pose_history.clear()
        self._rebuild_environment(reset_camera=True)

    def _ensure_target_items(self, target_id):
        if target_id in self._target_items:
            return self._target_items[target_id]
        color = TARGET_COLORS[(target_id - 1) % len(TARGET_COLORS)]
        box = gl.GLBoxItem(color=color, glOptions="translucent")
        trail = gl.GLLinePlotItem(
            color=tuple(channel / 255 for channel in color),
            width=1.6,
            mode="line_strip",
            antialias=True,
        )
        label = gl.GLTextItem(text=f"T{target_id}", color=color)
        self.view.addItem(trail)
        self.view.addItem(box)
        self.view.addItem(label)
        items = {"box": box, "trail": trail, "label": label}
        self._target_items[target_id] = items
        return items

    def _world_box_size(self, dimensions):
        if self.room_config["sensor_wall"] in ("Back", "Front"):
            return dimensions.copy()
        return np.array([dimensions[1], dimensions[0], dimensions[2]])

    def _shape_target(self, target, world_response_center):
        shaped = dict(target)
        raw_width = float(target["width_m"])
        raw_depth = float(target["depth_m"])
        raw_height = float(target["height_m"])
        horizontal_extent = max(raw_width, raw_depth)
        response_top = float(world_response_center[2] + raw_height / 2)

        if world_response_center[2] < 0.72 and horizontal_extent >= 0.55:
            provisional_pose = "lying"
        elif response_top >= 1.35 and world_response_center[2] >= 0.55:
            provisional_pose = "standing"
        else:
            provisional_pose = "sitting"
        history = self._pose_history[target["id"]]
        history.append(provisional_pose)
        pose = Counter(history).most_common(1)[0][0]

        max_width = self.room_config["max_object_width_m"]
        max_depth = self.room_config["max_object_depth_m"]
        max_height = min(
            self.room_config["height_m"], self.room_config["max_object_height_m"]
        )
        if pose == "standing":
            shaped["width_m"] = float(np.clip(raw_width, 0.35, max_width))
            shaped["depth_m"] = float(np.clip(raw_depth, 0.25, max_depth))
            shaped["display_height_m"] = float(np.clip(response_top, 1.3, max_height))
        elif pose == "sitting":
            side = float(
                np.clip(
                    max(raw_width, raw_depth, 0.5),
                    0.5,
                    min(max_width, max_depth),
                )
            )
            shaped["width_m"] = side
            shaped["depth_m"] = side
            shaped["display_height_m"] = float(np.clip(response_top, 0.65, min(1.3, max_height)))
        else:
            length = float(
                np.clip(
                    max(horizontal_extent, 1.15),
                    1.15,
                    self.room_config["max_lying_length_m"],
                )
            )
            cross_section = float(np.clip(min(raw_width, raw_depth), 0.45, 0.8))
            if raw_width >= raw_depth:
                shaped["width_m"], shaped["depth_m"] = length, cross_section
            else:
                shaped["width_m"], shaped["depth_m"] = cross_section, length
            shaped["display_height_m"] = float(np.clip(raw_height, 0.35, min(0.75, max_height)))
        shaped["pose"] = pose
        shaped["floor_anchored"] = bool(self.room_config["floor_anchored_targets"])
        return shaped

    def _render_targets(self):
        now = time.monotonic()
        self._render_motion_shadow(now)
        active_ids = set(self._latest_targets)
        for target_id, items in self._target_items.items():
            for item in items.values():
                item.setVisible(target_id in active_ids)

        for target_id, target in self._latest_targets.items():
            measured = np.array(
                [target["lateral_m"], target["forward_m"], target["vertical_m"]], dtype=float
            )
            velocity = np.array(
                [
                    target["velocity_lateral_mps"],
                    target["velocity_forward_mps"],
                    target["velocity_vertical_mps"],
                ],
                dtype=float,
            )
            prediction_age = min(0.2, max(0.0, now - self._latest_timestamp))
            desired_position = measured + velocity * prediction_age
            desired_dimensions = np.array(
                [target["width_m"], target["depth_m"], target["display_height_m"]], dtype=float
            )
            state = self._display_states.setdefault(
                target_id,
                {"position": desired_position.copy(), "dimensions": desired_dimensions.copy()},
            )
            state["position"] += 0.2 * (desired_position - state["position"])
            state["dimensions"] += 0.12 * (desired_dimensions - state["dimensions"])

            world_center = self._world_from_local(state["position"])
            if target["floor_anchored"]:
                world_center[2] = state["dimensions"][2] / 2
            else:
                world_center[2] = float(np.clip(world_center[2], 0.0, self.room_config["height_m"]))
            if not self._inside_floorplan(world_center):
                if target_id in self._target_items:
                    for item in self._target_items[target_id].values():
                        item.setVisible(False)
                continue
            world_size = self._world_box_size(state["dimensions"])
            items = self._ensure_target_items(target_id)
            items["box"].resetTransform()
            items["box"].setSize(x=world_size[0], y=world_size[1], z=world_size[2])
            items["box"].translate(*(world_center - world_size / 2))
            items["label"].setData(
                pos=tuple(world_center + np.array([world_size[0] / 2, 0, world_size[2] / 2])),
                text=f"T{target_id}",
            )
            history = self._histories[target_id]
            if history:
                items["trail"].setData(pos=np.array(history, dtype=float))

    def _render_motion_shadow(self, now):
        persistence = float(self.processing_config["motion_shadow"]["persistence_s"])
        while self._shadow_frames and now - self._shadow_frames[0][0] > persistence:
            self._shadow_frames.popleft()
        if not self._shadow_frames:
            self.motion_shadow_item.setData(
                pos=np.empty((0, 3), dtype=float),
                size=np.empty(0, dtype=float),
                color=np.empty((0, 4), dtype=float),
            )
            self._motion_visible = False
            return

        positions = []
        colors = []
        sizes = []
        for timestamp, points, intensity in self._shadow_frames:
            age_weight = max(0.0, 1.0 - (now - timestamp) / persistence)
            positions.append(points)
            alpha = np.clip(intensity * 0.24 * age_weight, 0.015, 0.22)
            colors.append(
                np.column_stack(
                    (
                        np.full(len(points), 0.38),
                        np.full(len(points), 0.82),
                        np.full(len(points), 0.78),
                        alpha,
                    )
                )
            )
            sizes.append(4.0 + intensity * 9.0)
        self.motion_shadow_item.setData(
            pos=np.vstack(positions),
            color=np.vstack(colors),
            size=np.concatenate(sizes),
            pxMode=True,
        )
        self._motion_visible = True

    def _update_table(self, targets):
        self.target_table.clear()
        for target in targets:
            state = "~" if target["coasting"] else ""
            self.target_table.addTopLevelItem(
                QtWidgets.QTreeWidgetItem(
                    [
                        f"T{target['id']}{state}",
                        target["pose"],
                        f"{target['range_m']:.2f}",
                        f"{target['width_m']:.2f}",
                        f"{target['depth_m']:.2f}",
                        f"{target['display_height_m']:.2f}",
                        f"{target['snr_db']:.1f}",
                    ]
                )
            )

    @QtCore.pyqtSlot(dict)
    def update_gui(self, payload):
        self._frame_count += 1
        if self._frame_count % 10 == 0:
            self._reload_room_config_if_changed()
        self._fps_times.append(payload["timestamp"])
        self._latest_timestamp = payload["timestamp"]

        shadow = payload["motion_shadow"]
        latest_world_shadow = np.empty((0, 3), dtype=float)
        if len(shadow["points"]):
            world_points = np.array(
                [self._world_from_local(point) for point in shadow["points"]], dtype=float
            )
            inside = (
                (world_points[:, 0] >= 0.0)
                & (world_points[:, 0] <= self.room_config["width_m"])
                & (world_points[:, 1] >= 0.0)
                & (world_points[:, 1] <= self.room_config["depth_m"])
            )
            world_points = world_points[inside]
            world_points[:, 2] = np.clip(
                world_points[:, 2], 0.0, self.room_config["height_m"]
            )
            if len(world_points):
                latest_world_shadow = world_points
                self._shadow_frames.append(
                    (payload["timestamp"], world_points, shadow["intensity"][inside])
                )

        visible_targets = []
        for target in payload["targets"]:
            local = np.array([target["lateral_m"], target["forward_m"], target["vertical_m"]])
            world = self._world_from_local(local)
            if self._inside_floorplan(world):
                target = self._shape_target(target, world)
                visible_targets.append(target)
                self._histories[target["id"]].append(tuple(world))
        self._latest_targets = {target["id"]: target for target in visible_targets}
        self._update_table(visible_targets)
        self._write_occupancy_state(visible_targets, latest_world_shadow)

        target_ids = tuple(target["id"] for target in visible_targets)
        if target_ids != self._last_target_ids:
            print(f"Confirmed target set: count={len(target_ids)} ids={target_ids}", flush=True)
            self._last_target_ids = target_ids
        if visible_targets and not self._first_target_logged:
            target = visible_targets[0]
            print(
                "First confirmed target: "
                f"id={target['id']} range={target['range_m']:.2f}m "
                f"dimensions={target['width_m']:.2f}x{target['depth_m']:.2f}x"
                f"{target['display_height_m']:.2f}m",
                flush=True,
            )
            self._first_target_logged = True

        count = len(visible_targets)
        self.target_count["value"].setText(str(count))
        self.status_label.setText(
            f"Tracking {count} confirmed object{'s' if count != 1 else ''}"
            if count
            else (
                "Motion present · no object-sized cluster"
                if len(shadow["points"])
                else "Sensor live · no motion"
            )
        )
        if len(self._fps_times) > 1:
            elapsed = self._fps_times[-1] - self._fps_times[0]
            fps = (len(self._fps_times) - 1) / elapsed if elapsed > 0 else 0.0
            self.frame_rate["value"].setText(f"{fps:.1f} / 30 Hz")
        if self._frame_count == 1:
            print("First 3D sensor frame received: rx_mask=7 display=30Hz", flush=True)
        if self._frame_count % 100 == 0:
            detection = payload["detection"]
            print(
                f"Detection health: peak={detection['cfar_peak_db']:.1f}dB "
                f"motion={detection['motion_points']} "
                f"clusters={detection['candidate_count']} "
                f"static={detection.get('static_targets', 0)} confirmed={count}",
                flush=True,
            )

    def closeEvent(self, event):
        if hasattr(self, "render_timer"):
            self.render_timer.stop()
        if hasattr(self, "worker"):
            self.worker.stop()
        if hasattr(self, "thread"):
            self.thread.quit()
            self.thread.wait(1500)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
