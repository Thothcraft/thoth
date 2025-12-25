import json
import os
import sys
import numpy as np
import time
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
from PyQt5.QtGui import QColor

from radar_dev import RadarDev
from signal_proc import SigProc

os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

radar_config_fn = "config/BGT60TR13C_settings_20241109-215206.json"
processing_config_fn = "config/processing_config.json"


# Worker class for background processing
class Worker(QtCore.QObject):
    update_signal = QtCore.pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    def __init__(self, radar, sig_proc):
        self.radar = radar
        self.sig_proc = sig_proc
        super().__init__()

    def run(self):
        while True:
            frame = self.radar.get_next_frame()
            if frame is not None:
                location, score, gui_plot = self.sig_proc.update(frame)
                self.update_signal.emit(gui_plot["map"], gui_plot["x_axis"], gui_plot["y_axis"], location)


# Main window class
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize device using the config
        radar = RadarDev(9575, radar_config_fn)

        # Initialize signal processing
        sig_proc = SigProc(processing_config_fn, radar.cfg)

        self.setWindowTitle("Millimeter Wave Radar Detection")
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # PyQtGraph Image Plot
        self.plot_widget = pg.PlotWidget()
        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)
        layout.addWidget(self.plot_widget)

        # Create and start the worker in a new thread
        radar.open_radar_device()
        self.thread = QtCore.QThread()
        self.worker = Worker(radar, sig_proc)
        self.worker.moveToThread(self.thread)
        self.worker.update_signal.connect(self.update_gui)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def update_gui(self, image_data, x_ticks, y_ticks, location):

        # Update the image plot, swap x-y axis for better plot
        image_data = image_data.T
        x_ticks, y_ticks = y_ticks, x_ticks

        # Mirror the plot for better visualization
        image_data = image_data[::-1, ::-1]
        x_ticks = x_ticks[::-1]
        y_ticks = y_ticks[::-1]

        self.image_item.setImage(image_data, levels=(0,1))

        colormap = pg.colormap.get('viridis')  # Get the colormap
        lut = colormap.getLookupTable(start=0, stop=1.0, alpha=False)  # Create LUT from the colormap
        self.image_item.setLookupTable(lut)  # Apply the LUT to the ImageView

        # Update x and y tick values
        x_tick_interval = 9  # Change as needed
        y_tick_interval = 9  # Change as needed
        x_tick_values = [(i, format(x, ".1f")) for i, x in enumerate(x_ticks) if i % x_tick_interval == 0]
        y_tick_values = [(i, format(y, ".1f")) for i, y in enumerate(y_ticks) if i % y_tick_interval == 0]

        self.plot_widget.getPlotItem().getAxis('bottom').setTicks([x_tick_values])
        self.plot_widget.getPlotItem().getAxis('left').setTicks([y_tick_values])


# Run the application
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
