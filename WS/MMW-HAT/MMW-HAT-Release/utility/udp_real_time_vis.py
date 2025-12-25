import datetime
import json
import os
import socket
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QGridLayout, QWidget
from PyQt5.QtCore import QThread, pyqtSignal

from utility.helper import find_setting_in_directory, parse_full_frame
from utility.mmw_cube_proc_v0 import CubeProcessor

# This order is required to avoid problems on Raspberry Pi.
app = QApplication(sys.argv)
import pyqtgraph as pg


class ImageGrid(QWidget):
    def __init__(self, M, N):
        super().__init__()
        self.M = M
        self.N = N
        self.initUI()

    def initUI(self):
        # Set up grid layout
        grid = QGridLayout()
        self.setLayout(grid)

        # Generate viridis colormap lookup table
        colormap = pg.colormap.get('viridis')
        lut = colormap.getLookupTable(0.0, 1.0, 256)

        # Add M x N PlotItems with ImageItems to the grid without displaying initially
        self.image_items = []
        self.plot_items = []  # Store plot items to access later for titles and labels
        for i in range(self.M):
            row_items = []
            row_plots = []
            for j in range(self.N):
                plot_item = pg.PlotItem()
                plot_item.setAspectLocked(True)
                plot_item.hideAxis('left')
                plot_item.hideAxis('bottom')

                image_item = pg.ImageItem()
                image_item.setLookupTable(lut)
                plot_item.addItem(image_item)

                view = pg.GraphicsLayoutWidget()
                view.addItem(plot_item)

                grid.addWidget(view, i, j)

                row_items.append(image_item)
                row_plots.append(plot_item)
            self.image_items.append(row_items)
            self.plot_items.append(row_plots)

        self.setWindowTitle('Image Grid with Color Map')
        self.resize(1920, 1080)

    def set_plot_title_and_labels(self, row, col, title, x_name, y_name):
        axis_label_dict = {"Range": "Range(m)",
                           "Doppler": "Doppler(Hz)",
                           "Azimuth": "Azimuth(deg)",
                           "Elevation": "Elevation(deg)"}

        """Set the title and axis labels for a specific plot at (row, col)."""
        if 0 <= row < self.M and 0 <= col < self.N:
            plot_item = self.plot_items[row][col]
            plot_item.setTitle(title, size="18pt")
            bottom_axis = plot_item.getAxis('bottom')
            left_axis = plot_item.getAxis('left')
            label_font = pg.QtGui.QFont("Arial", 16)
            tick_font = pg.QtGui.QFont("Arial", 14)
            bottom_axis.setLabel(axis_label_dict[x_name])
            bottom_axis.label.setFont(label_font)
            bottom_axis.setStyle(tickFont=tick_font)
            left_axis.setLabel(axis_label_dict[y_name])
            left_axis.label.setFont(label_font)
            left_axis.setStyle(tickFont=tick_font)
            plot_item.showAxis('bottom')
            plot_item.showAxis('left')

    def set_axis_ticks(self, row, col, x_ticks, y_ticks):
        """Set custom axis ticks for a specific plot at (row, col)."""
        if 0 <= row < self.M and 0 <= col < self.N:
            plot_item = self.plot_items[row][col]
            if x_ticks:
                plot_item.getAxis('bottom').setTicks([x_ticks])
            if y_ticks:
                plot_item.getAxis('left').setTicks([y_ticks])

    def display_image(self, row, col, image_data):
        if 0 <= row < self.M and 0 <= col < self.N:
            self.image_items[row][col].setImage(image_data)


class ImageUpdateThread(QThread):
    update_image_signal = pyqtSignal(int, int, np.ndarray)

    def __init__(self, port, setting, plots, num_angle_bins, grid, parent=None, save_to_file=None):
        super().__init__(parent)
        self.__port = port
        self.__is_running = True
        self.__setting = setting
        self.__plots = plots
        self.__grid = grid  # Pass the ImageGrid instance
        self.__mmw_proc = CubeProcessor(setting, num_azimuth_bin=num_angle_bins, num_elevation_bin=num_angle_bins)
        self.__save_to_file = save_to_file

        # Set axis ticks for each plot
        for plot in self.__plots:
            row, col, axis_0_name, axis_1_name = plot
            axis_0_ticks = self._get_axis_ticks(axis_0_name)
            axis_1_ticks = self._get_axis_ticks(axis_1_name)
            self.__grid.set_axis_ticks(row, col, axis_0_ticks, axis_1_ticks)

    def _get_axis_ticks(self, axis_name, skip=5):
        """Retrieve axis ticks from CubeProcessor's proc_param using the formatted name."""
        search_name = f"{axis_name.lower()}_bin"
        if search_name in self.__mmw_proc.proc_param:
            bins = self.__mmw_proc.proc_param[search_name]
            center = len(bins) // 2
            indices = range(center % skip, len(bins), skip)  # Start near center and increment by skip
            return [(i, f"{bins[i]:.1f}") for i in indices]
        return None

    def run(self):
        file_fd = None
        if self.__save_to_file is not None:
            # Get the current timestamp at millisecond precision
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

            # Add the timestamp to the filename
            filename_with_timestamp = f"{self.__save_to_file}_{timestamp}.bin"

            # Ensure the directory exists
            directory = os.path.dirname(filename_with_timestamp)
            if directory:  # Check if a directory is specified in the path
                os.makedirs(directory, exist_ok=True)

            file_fd = open(filename_with_timestamp, "wb")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', self.__port))
        while self.__is_running:
            raw_bytes, _ = sock.recvfrom(131072)
            if file_fd is not None:
                file_fd.write(raw_bytes)
            (version, seq, data_len, raw_data) = parse_full_frame(raw_bytes)
            raw_data = list(raw_data)
            self.__mmw_proc.process_raw_data(raw_data)
            for plot in self.__plots:
                row, col, axis_0_name, axis_1_name = plot
                img = self.__mmw_proc.vis_2d(axis_0_name, axis_1_name)
                img = np.log10(img)
                self.update_image_signal.emit(row, col, img)

    def stop(self):
        self.__is_running = False
        self.quit()
        self.wait()

def main(port, cfg_dir, num_rows, num_cols, plots, num_angle_bins, fn=None):
    grid = ImageGrid(num_rows, num_cols)
    for plot in plots:
        row, col, axis_0_name, axis_1_name = plot
        grid.set_plot_title_and_labels(row, col, axis_0_name + "-" + axis_1_name, axis_0_name, axis_1_name)
    grid.show()
    setting_fn = find_setting_in_directory(cfg_dir)
    with open(setting_fn, 'r') as file:
        setting_data = json.load(file)
    thread = ImageUpdateThread(port, setting_data, plots, num_angle_bins, grid, save_to_file=fn)
    thread.update_image_signal.connect(grid.display_image)
    thread.start()
    try:
        sys.exit(app.exec_())
    finally:
        thread.stop()


if __name__ == '__main__':
    port = 9575
    num_rows = 1
    num_cols = 3
    plots = [(0, 0, "Range", "Doppler"),
             (0, 1, "Azimuth", "Range"),
             (0, 2, "Azimuth", "Doppler")]
    cfg_dir = "../radar_config/config_3rx_3m"
    fn = "data/mmw_udp"
    num_angle_bins = 16
    main(port, cfg_dir, num_rows, num_cols, plots, num_angle_bins, fn)
