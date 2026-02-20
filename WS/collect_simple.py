#!/usr/bin/env python3
"""
Simplified CSI collector with visualization.
- Collects data for specified duration and times
- If vis is enabled, no file saving
- No memory buildup - samples are discarded after window
- Shows 6 plots: amplitude means and stds for 3 window sizes (50, 500, 2000)
"""

import argparse
import sys
import time
import threading
import signal
import math
import numpy as np
from collections import deque

try:
    import serial
    import serial.tools.list_ports as list_ports
except ImportError:
    print("ERROR: pyserial is required. Install with: pip install pyserial", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.cm import get_cmap
except ImportError:
    print("WARNING: matplotlib not available for visualization", file=sys.stderr)

# Global variables
stop_event = threading.Event()
_csi_count = 0
_csi_count_lock = threading.Lock()

# Data buffers for simplified visualization
MAX_BUFFER_SIZE = 2000
_data_buffers = {
    'amps': deque(maxlen=MAX_BUFFER_SIZE),  # Store amplitude arrays for 52 subcarriers
    'timestamps': deque(maxlen=MAX_BUFFER_SIZE)
}

def reset_board(port, baud, label):
    """Reset ESP32 board."""
    try:
        with serial.Serial(port, baudrate=baud) as s:
            s.dtr = False; s.rts = True
            time.sleep(0.05)
            s.dtr = True;  s.rts = False
            time.sleep(0.05)
        print(f"[info] Reset pulsed on {label} at {port}")
    except Exception as e:
        print(f"[warn] Could not reset {label} at {port}: {e}", file=sys.stderr)

def parse_csi_data(text):
    """Parse CSI data from text line."""
    try:
        parts = text.split(',', 14)
        if len(parts) < 15:
            return None
            
        rssi = int(parts[3])
        data_str = parts[14]
        
        if not data_str:
            return None
            
        nums = data_str.strip('"[] ').split(',')
        amps = []
        
        for j in range(0, len(nums) - 1, 2):
            try:
                re_v = float(nums[j])
                im_v = float(nums[j + 1])
                amp = math.sqrt(re_v * re_v + im_v * im_v)
                amps.append(amp)
            except (ValueError, IndexError):
                continue
                
        if amps:
            return {
                'rssi': rssi,
                'amplitudes': np.array(amps),
                'amp_mean': np.mean(amps),
                'amp_std': np.std(amps)
            }
    except Exception:
        pass
    return None

def collect_data(rx_port, baud, duration, times, save_dir=None):
    """Collect CSI data and store in simplified buffer."""
    global _csi_count
    
    for run in range(times):
        if stop_event.is_set():
            break
            
        print(f"\n[run] {run+1}/{times}")
        
        # Setup serial connection
        ser = serial.Serial(rx_port, baudrate=baud, timeout=1)
        ser.flushInput()
        
        start_time = time.time()
        
        while time.time() - start_time < duration and not stop_event.is_set():
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # Parse CSI data
                amps = parse_csi_data(line)
                if amps is not None and len(amps) >= 52:
                    # Store in simplified buffer
                    _data_buffers['amps'].append(amps)
                    _data_buffers['timestamps'].append(time.time())
                    
                    with _csi_count_lock:
                        _csi_count += 1
        
        ser.close()
        print(f"[run] {run+1} complete, collected {_csi_count} CSI packets")
        
        if run < times - 1 and not stop_event.is_set():
            print(f"[run] Waiting 2s before next run...")
            time.sleep(2)

def visualize_data(duration):
    """Visualize running mean and std for 52 subcarriers in real-time."""
    print("[vis] Starting visualization...")
    
    # Set up the figure with 2 subplots (mean and std)
    plt.ion()
    fig, (ax_mean, ax_std) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('CSI Subcarrier Analysis - Real-time Mean & Standard Deviation', 
                 fontsize=16, fontweight='bold')
    
    # Initialize lines for 52 subcarriers
    lines_mean = []
    lines_std = []
    colors = plt.cm.viridis(np.linspace(0, 1, 52))
    
    for i in range(52):
        line_mean, = ax_mean.plot([], [], color=colors[i], linewidth=1.5, 
                                  alpha=0.8, label=f'SC{i}' if i < 5 else '')
        line_std, = ax_std.plot([], [], color=colors[i], linewidth=1.5, 
                                alpha=0.8, label=f'SC{i}' if i < 5 else '')
        lines_mean.append(line_mean)
        lines_std.append(line_std)
    
    # Configure mean plot
    ax_mean.set_title('Running Mean Amplitude per Subcarrier', fontweight='bold')
    ax_mean.set_ylabel('Amplitude Mean')
    ax_mean.grid(True, alpha=0.3)
    ax_mean.set_xlim(0, duration)
    ax_mean.legend(loc='upper right', ncol=5, fontsize=8)
    
    # Configure std plot
    ax_std.set_title('Running Std Amplitude per Subcarrier', fontweight='bold')
    ax_std.set_ylabel('Amplitude Std')
    ax_std.set_xlabel('Time (s)')
    ax_std.grid(True, alpha=0.3)
    ax_std.set_xlim(0, duration)
    
    # Data storage for running statistics
    time_window = deque(maxlen=1000)  # Store last 1000 time points
    subcarrier_data = [deque(maxlen=1000) for _ in range(52)]  # Data for each subcarrier
    
    plt.tight_layout()
    
    # Animation loop
    start_time = time.time()
    
    while not stop_event.is_set() and time.time() - start_time < duration:
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Get latest data
        buffer = list(_data_buffers['amps'])  # Use simplified buffer
        timestamps = list(_data_buffers['timestamps'])
        
        if len(buffer) > 0 and len(timestamps) > 0:
            # Get the most recent data point
            latest_amps = buffer[-1]
            latest_time = timestamps[-1]
            
            if len(latest_amps) >= 52:  # Ensure we have 52 subcarriers
                # Add new data point
                time_window.append(elapsed)
                for i in range(52):
                    subcarrier_data[i].append(latest_amps[i])
                
                # Update plots if we have enough data
                if len(time_window) >= 10:
                    times = list(time_window)
                    
                    # Calculate running statistics for each subcarrier
                    for i in range(52):
                        sc_data = list(subcarrier_data[i])
                        if len(sc_data) >= 5:  # Need minimum points for std
                            # Calculate running mean and std
                            running_mean = []
                            running_std = []
                            
                            for j in range(5, len(sc_data) + 1):
                                window = sc_data[max(0, j-50):j]  # Use last 50 points
                                running_mean.append(np.mean(window))
                                running_std.append(np.std(window))
                            
                            # Align time data
                            plot_times = times[-len(running_mean):]
                            
                            # Update lines
                            lines_mean[i].set_data(plot_times, running_mean)
                            lines_std[i].set_data(plot_times, running_std)
                    
                    # Auto-scale y-axis
                    if len(times) > 10:
                        # Get all data for y-axis limits
                        all_means = []
                        all_stds = []
                        for i in range(52):
                            if lines_mean[i].get_ydata().size > 0:
                                all_means.extend(lines_mean[i].get_ydata())
                                all_stds.extend(lines_std[i].get_ydata())
                        
                        if all_means:
                            y_margin = 0.1 * (np.max(all_means) - np.min(all_means))
                            ax_mean.set_ylim(np.min(all_means) - y_margin, 
                                           np.max(all_means) + y_margin)
                        
                        if all_stds:
                            y_margin = 0.1 * (np.max(all_stds) - np.min(all_stds))
                            ax_std.set_ylim(np.min(all_stds) - y_margin, 
                                          np.max(all_stds) + y_margin)
        
        # Update display
        plt.pause(0.05)
    
    plt.ioff()
    print(f"[vis] Visualization ended after {elapsed:.1f}s")

def rate_monitor():
    """Print collection rate."""
    global _csi_count
    last_count = 0
    last_time = time.time()
    
    while not stop_event.is_set():
        time.sleep(1.0)
        current_time = time.time()
        
        with _csi_count_lock:
            current_count = _csi_count
        
        time_elapsed = current_time - last_time
        if time_elapsed > 0:
            cps = int((current_count - last_count) / time_elapsed)
            print(f"[rate] {cps:5d} CSI/s  (total={current_count})")
        
        last_count = current_count
        last_time = current_time

def main():
    parser = argparse.ArgumentParser(description="Simplified CSI collector")
    parser.add_argument("--rx-port", required=True, help="Receiver serial port")
    parser.add_argument("--tx-port", default=None, help="Sender serial port (optional)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--out", help="Output file (ignored if --vis is used)")
    parser.add_argument("--duration", type=float, required=True, help="Duration in seconds")
    parser.add_argument("--times", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--vis", action="store_true", help="Enable visualization (no file saving)")
    parser.add_argument("--no-reset", action="store_true", help="Skip board reset")
    
    args = parser.parse_args()
    
    print(f"[info] Receiver: {args.rx_port} @ {args.baud}")
    if args.tx_port:
        print(f"[info] Sender: {args.tx_port} @ {args.baud}")
    print(f"[info] Duration: {args.duration}s")
    print(f"[info] Repeats: {args.times}")
    print(f"[info] Visualization: {'ON (no file saving)' if args.vis else 'OFF'}")
    
    # Reset boards
    if not args.no_reset:
        reset_board(args.rx_port, args.baud, "receiver")
        if args.tx_port:
            reset_board(args.tx_port, args.baud, "sender")
    
    for i in range(args.times):
        print(f"\n[info] Collection {i+1}/{args.times}")
        
        # Reset counters and buffers
        global _csi_count
        _csi_count = 0
        for key in _data_buffers:
            _data_buffers[key].clear()
        stop_event.clear()
        
        # Start threads
        if args.vis:
            # Visualization mode - no file saving
            collector_thread = threading.Thread(
                target=collect_data, 
                args=(args.rx_port, args.baud, args.duration, False),
                daemon=True
            )
            monitor_thread = threading.Thread(target=rate_monitor, daemon=True)
            
            collector_thread.start()
            monitor_thread.start()
            
            # Run visualization in main thread
            visualize_data(args.duration)
            
        else:
            # File saving mode
            output_file = f"{args.out}_{i+1}.csv" if args.out else f"csi_data_{i+1}.csv"
            collector_thread = threading.Thread(
                target=collect_data,
                args=(args.rx_port, args.baud, args.duration, True, output_file),
                daemon=True
            )
            monitor_thread = threading.Thread(target=rate_monitor, daemon=True)
            
            collector_thread.start()
            monitor_thread.start()
            
            # Wait for completion
            collector_thread.join()
            time.sleep(0.1)  # Allow final rate print
        
        stop_event.set()
        time.sleep(0.5)  # Brief pause between runs
    
    print("\n[info] All collections completed.")

if __name__ == "__main__":
    def signal_handler(signum, frame):
        stop_event.set()
        print("\n[info] Stopping...")
    
    signal.signal(signal.SIGINT, signal_handler)
    main()
