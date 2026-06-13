# MMW-HAT Radar Sensing Reference

This directory contains the Raspberry Pi radar sensing examples for the Infineon
BGT60TR13C millimeter-wave FMCW radar HAT. The code configures the radar over
SPI, streams raw ADC frames over local UDP, visualizes range/Doppler/angle
products, supports target tracking, and can save raw and processed outputs for
offline analysis.

The active release tree is:

```text
WS/MMW-HAT/
├── example_1_vis.sh
├── example_2_track.sh
├── example_3_vis.sh
└── MMW-HAT-Release/
    ├── example_1_full_vis/
    ├── example_2_track/
    ├── example_3_full_vis/
    ├── example_4_offline_proc/
    ├── radar_config/
    └── utility/
```

## Hardware

The examples target a BGT60TR13C radar sensor connected to a Raspberry Pi by SPI
and GPIO.

Observed/default hardware wiring in the driver:

```text
Radar IC:        Infineon BGT60TR13C
Radar type:      FMCW millimeter-wave radar
Frequency band:  58 GHz to 63 GHz, depending on config
TX used:         TX1
RX used:         RX mask from config, usually RX1/RX2/RX3
SPI bus/dev:     driver default must match the installed HAT
SPI mode:        mode 0
SPI speed:       10 MHz default, 50 MHz in streaming examples
Reset GPIO:      BCM GPIO12
IRQ GPIO:        BCM GPIO25
UDP host:        127.0.0.1
UDP port:        9575
```

Important SPI note:

The radar returns `CHIP_ID: 0xF4000303` when the driver is pointed at the
correct SPI device. A read of `0xFFFFFFFF` means the SPI path is not receiving a
valid response, commonly because the wrong SPI bus is used, the HAT is not
responding, or MISO is pulled high. On the tested Pi, the working tree used SPI
bus `0` and read `0xF4000303`; a stale mirrored tree configured for SPI bus `10`
read `0xFFFFFFFF`.

GPIO resource note:

Only one radar example should run at a time. If a previous run exits poorly,
GPIO12/GPIO25 or UDP port `9575` can remain busy. Stop stale processes before
starting another example:

```bash
pgrep -af 'run_example|run_udp_streaming|run_vis|location_gui'
kill <pid> <pid>
```

## Radar Configurations

The examples use two main radar configurations.

### `radar_config/config_3rx_3m`

Used by examples 1, 3, and 4.

```text
Register file:  BGT60TR13C_export_registers_20241101-104319.txt
Settings file:  BGT60TR13C_settings_20241101-104314.json
Start freq:     58 GHz
End freq:       60 GHz
Bandwidth:      2 GHz
Sample rate:    2 MHz
RX mask:        7, RX1 + RX2 + RX3
TX mask:        1, TX1
Samples/chirp:  128
Chirps/frame:   64
Frame period:   about 0.1000074 s
Frame rate:     about 10 Hz
Chirp period:   about 0.0010007 s
```

Calculated frame payload:

```text
ADC samples/frame = 64 chirps * 128 samples/chirp * 3 RX = 24576 uint12 samples
Payload bytes     = 24576 / 2 * 3 = 36864 bytes
Frame bytes       = 12 byte software header + 36864 byte payload
```

This configuration is better suited to full visualization because it uses all
three receive antennas and enough chirps for range-Doppler and angle products.

### `radar_config/config_track`

Used by example 2.

```text
Register file:  BGT60TR13C_export_registers_20241109-215212.txt
Settings file:  BGT60TR13C_settings_20241109-215206.json
Start freq:     58 GHz
End freq:       63 GHz
Bandwidth:      5 GHz
Sample rate:    1 MHz
RX mask:        5, RX1 + RX3
TX mask:        1, TX1
Samples/chirp:  512
Chirps/frame:   16
Frame period:   about 0.2003145 s
Frame rate:     about 5 Hz
Chirp period:   about 0.0019993 s
```

Calculated frame payload:

```text
ADC samples/frame = 16 chirps * 512 samples/chirp * 2 RX = 16384 uint12 samples
Payload bytes     = 16384 / 2 * 3 = 24576 bytes
Frame bytes       = 12 byte software header + 24576 byte payload
```

This configuration is optimized for tracking with two azimuth receive antennas,
a wider RF sweep, and a lower frame rate.

## Launcher Scripts

### `example_1_vis.sh`

Runs:

```text
MMW-HAT-Release/example_1_full_vis/run_example_vis.py
```

Behavior:

1. Starts `run_vis.py`.
2. Waits 20 seconds.
3. Starts `run_udp_streaming_vis.py`.
4. Joins both child processes.

Output:

```text
Range-Doppler
Azimuth-Range
Azimuth-Doppler
```

This example uses `config_3rx_3m`.

### `example_2_track.sh`

Runs:

```text
MMW-HAT-Release/example_2_track/run_example_track.py
```

Behavior:

1. Starts `location_gui.py`.
2. Waits 5 seconds.
3. Starts `run_udp_streaming_track.py`.
4. Joins both child processes.

Output:

```text
Target heat map on an x-y plane
Latest estimated target location
Detection score from temporal buffer decay
```

This example uses `config_track`.

### `example_3_vis.sh`

Runs:

```text
MMW-HAT-Release/example_3_full_vis/run_example_vis.py
```

Behavior is similar to example 1, but it uses the example 3 visualizer entry
point. It uses `config_3rx_3m` and the same UDP frame protocol.

### `example_4_offline_proc`

This is an offline workflow.

```text
data_collection.py  captures raw frames to data/mmw_spi_<timestamp>.bin
offline_proc.py     reads a saved .bin file and writes processed .jpg products
```

Default processed outputs:

```text
mmw_proc/<seq>_Range_Doppler.jpg
mmw_proc/<seq>_Azimuth_Range.jpg
mmw_proc/<seq>_Azimuth_Doppler.jpg
```

## Runtime Data Flow

The online examples use two cooperating processes:

```text
Radar process:
  BGT60TR13C driver
  SPI register configuration
  FIFO IRQ reads
  frame packing
  UDP send to 127.0.0.1:9575

Visualization/tracking process:
  UDP bind on 0.0.0.0:9575
  frame parsing
  uint12 ADC unpacking
  signal processing
  GUI update and optional file save
```

Only the UDP receiver binds port `9575`. The streamer sends UDP packets from an
ephemeral local port to `127.0.0.1:9575`.

## Raw Frame Format

The driver wraps every radar frame in a small software header before sending it
over UDP or writing it to `.bin`.

Frame layout:

```text
offset  size  type     endian  name
0       4     uint32   little  version
4       4     uint32   little  seq
8       4     uint32   little  data_len
12      N     bytes    n/a     packed ADC payload
```

Current version:

```text
version = 0
```

Payload format:

```text
Two 12-bit ADC samples are packed into every three bytes.

byte0          byte1          byte2
aaaa aaaa      bbbb cccc      cccc cccc

sample A = byte0 << 4 | byte1 >> 4
sample B = (byte1 & 0x0f) << 8 | byte2
```

Python unpacking:

```text
utility.helper.read_uint12(data_chunk)
  input:  bytes/list of uint8
  output: numpy.ndarray dtype float32
```

Frame reshape:

```text
utility.helper.split_samples(adc_data, num_frames, num_chirps, num_samples, num_rx)
  output shape: (num_frames, num_chirps, num_samples, num_rx)
```

Example 2 transposes one frame to:

```text
(num_rx_antennas, num_chirps_per_frame, num_samples_per_chirp)
```

## Data Types

Important runtime data types:

```text
Raw SPI bytes:          list[int] or bytes, uint8 values
Packed ADC payload:    3 bytes per 2 samples
Unpacked ADC samples:  numpy.float32 array
Data cube input:       numpy.float32
FFT output:            complex64 inside FFTW processor
Power maps:            numpy.float32 or numpy.float64 after abs^2/log10
UDP header fields:     uint32 little-endian
UDP payload length:    uint32 little-endian byte count
Sequence number:       uint32 little-endian, increments per full frame
```

## Preprocessing Pipelines

### Shared Hardware Pipeline

All live examples share this hardware path:

1. Open SPI device.
2. Set SPI speed and mode.
3. Claim reset and IRQ GPIOs.
4. Toggle reset GPIO for a hard reset.
5. Read `BGT60TR13C` chip ID.
6. Load one register export file from the selected config directory.
7. Load one JSON settings file from the same config directory.
8. Calculate frame size from settings.
9. Set FIFO parameters.
10. Start the data collection thread.
11. Soft reset the radar FSM.
12. Start frame generation by setting the frame-start bit.
13. On IRQ, read FIFO bursts.
14. Assemble complete frames.
15. Add the 12-byte software header.
16. Send the frame over UDP and optionally save to `.bin`.

FIFO settings used by `utility.udp_streaming`:

```text
num_samples_irq:        4096
num_samples_per_burst:  2048
num_samples_per_frame:  derived from JSON settings
```

### Full Visualization Pipeline

Implemented mainly by:

```text
utility.udp_real_time_vis.ImageUpdateThread
utility.mmw_cube_proc_v0.CubeProcessor
```

Processing stages:

1. Receive one UDP frame.
2. Parse `version`, `seq`, `data_len`, and payload.
3. Unpack packed uint12 ADC data into float32 samples.
4. Reshape to `(1, chirps, samples, antennas)`.
5. Allocate a 4-D cube:

   ```text
   (doppler_bin, range_bin, azimuth_bin, elevation_bin)
   ```

6. Map RX antennas into the virtual antenna grid:

   ```text
   RX1 -> (1, 0)
   RX2 -> (0, 1)
   RX3 -> (0, 0)
   ```

7. Optionally apply MTI if `mti_alpha` is configured.
8. Run FFTW over Doppler, range, azimuth, and elevation axes.
9. FFT-shift Doppler, azimuth, and elevation axes.
10. Convert complex FFT output to power using squared magnitude.
11. Drop near-range bins below `min_range`.
12. Reduce unused axes by mean to generate 2-D visual products.
13. Apply `log10` before display.

Available visual dimensions:

```text
Doppler
Range
Azimuth
Elevation
```

Default displayed products:

```text
Range-Doppler
Azimuth-Range
Azimuth-Doppler
```

### Tracking Pipeline

Implemented mainly by:

```text
example_2_track/radar_dev.py
example_2_track/signal_proc.py
```

Processing stages:

1. Receive a UDP frame on port `9575`.
2. Parse the software frame header.
3. Unpack uint12 ADC samples.
4. Reshape and transpose to:

   ```text
   (num_rx_antennas, num_chirps_per_frame, num_samples_per_chirp)
   ```

5. For every RX antenna, compute a Doppler map.
6. Run digital beamforming over the range-Doppler spectrum.
7. Collapse Doppler energy into a range-angle map.
8. Ignore the configured dead zone.
9. Pick the strongest remaining range-angle bin.
10. Validate detection window size.
11. Convert polar range/angle to x-y:

    ```text
    x = range * cos(angle)
    y = range * sin(angle)
    ```

12. Accumulate a decayed x-y map in a fixed-length buffer.
13. Report location and score based on the latest valid detection.

Default tracking processing config:

```text
num_beams:           55
max_angle_deg:       40
buffer_len:          10
buffer_decay:        0.8
max_x:               5 m
max_y:               +/-5 m
spatial_resolution:  0.05 m
dead_zone:           1.0 m
detection frames:    5
range window:        0.25 m
angle window:        10 deg
```

## Output Files And Extensions

### `.sh`

Launcher scripts.

```text
example_1_vis.sh
example_2_track.sh
example_3_vis.sh
```

### `.py`

Python runtime, driver, visualization, tracking, and offline processing code.

### `.txt`

Radar register exports.

Format:

```text
label address_hex value_hex
```

Example parser behavior:

```text
label, address_str, value_str = line.split()
address = int(address_str, 16)
value = int(value_str, 16)
```

The loader writes every register over SPI. For `SFCTL`, it also toggles the
high-speed MISO read bit depending on SPI speed.

### `.json`

Radar settings and processing settings.

Radar settings define:

```text
frame repetition time
chirp repetition time
start and end frequencies
sample rate
number of ADC samples per chirp
RX mask
TX mask
IF gain
HP/LP filter cutoffs
TX power
```

Processing settings define:

```text
beam count
angle limits
buffer length
buffer decay
x-y grid limits
dead zone
detection window sizes
```

### `.bin`

Raw frame stream files. These contain the same framed data that is sent over
UDP:

```text
uint32 version
uint32 seq
uint32 data_len
data_len bytes of packed uint12 ADC payload
repeat until EOF
```

Common save prefixes:

```text
data/mmw_spi_<timestamp>.bin  saved by radar/SPI-side capture
data/mmw_udp_<timestamp>.bin  saved by UDP/visualizer-side capture
```

### `.jpg`

Offline processed images generated by `example_4_offline_proc/offline_proc.py`.

Default naming:

```text
<seq>_Range_Doppler.jpg
<seq>_Azimuth_Range.jpg
<seq>_Azimuth_Doppler.jpg
```

The image content is a `log10` power visualization of the selected 2-D reduction.

## Synchronized Saving

There are three relevant save points:

```text
1. SPI-side raw capture in utility.BGT60TR13C
2. UDP-side raw capture in utility.udp_real_time_vis.ImageUpdateThread
3. Offline processed images in example_4_offline_proc/offline_proc.py
```

Current behavior:

Each saving path appends its own timestamp when `save_to_file` or `FN` is not
`None`. That is useful for standalone capture but can make cross-file alignment
hard because the SPI file, UDP file, and processed images may have different
timestamps.

Recommended synchronized convention:

```text
session_id = YYYYMMDD_HHMMSS_mmm

SPI raw:       data/<session_id>/mmw_spi_<session_id>.bin
UDP raw:       data/<session_id>/mmw_udp_<session_id>.bin
Processed:     data/<session_id>/mmw_proc/<seq>_<view>.jpg
Metadata:      data/<session_id>/session_<session_id>.json
```

Recommended metadata fields:

```json
{
  "session_id": "YYYYMMDD_HHMMSS_mmm",
  "radar_config_dir": "../radar_config/config_3rx_3m",
  "register_file": "BGT60TR13C_export_registers_20241101-104319.txt",
  "settings_file": "BGT60TR13C_settings_20241101-104314.json",
  "udp_ip": "127.0.0.1",
  "udp_port": 9575,
  "software_frame_version": 0,
  "raw_frame_header": "uint32 version, uint32 seq, uint32 data_len, payload",
  "payload_type": "packed uint12 ADC samples",
  "processed_views": ["Range-Doppler", "Azimuth-Range", "Azimuth-Doppler"]
}
```

Recommended synchronization key:

Use `seq` as the primary alignment key. The SPI-side `.bin`, UDP-side `.bin`,
and processed `.jpg` outputs all preserve or derive from the software frame
sequence number. Do not rely only on wall-clock timestamp matching.

Recommended implementation pattern:

1. Generate one `session_id` in the top-level launcher.
2. Pass it to the streamer and visualizer through an environment variable:

   ```bash
   export MMW_SESSION_ID="$(date +%Y%m%d_%H%M%S_%3N)"
   ```

3. Use the same base directory in all scripts:

   ```text
   data/${MMW_SESSION_ID}/
   ```

4. Save one metadata JSON file before streaming starts.
5. Save raw SPI and UDP files with the same session id.
6. Name processed files with the sequence number first.

This keeps the three outputs synchronized even when the GUI starts before the
streamer or the offline processor runs later.

## Operational Checks

Before running examples:

```bash
cd WS/MMW-HAT
pgrep -af 'run_example|run_udp_streaming|run_vis|location_gui'
```

Expected good startup lines:

```text
CHIP_ID: 0xF4000303
This chip is BGT60TR13C.
Loading radar_config file: ...
Data collection thread started.
```

Common failure modes:

```text
CHIP_ID: 0xFFFFFFFF
  The SPI device is not returning data. Check SPI bus, HAT seating, power, and
  stale mirrored code using the wrong bus.

Soft reset timeout! Hard reset required.
  The driver wrote the reset bit but the chip did not clear it. Usually follows
  bad SPI communication or a wedged hardware state.

OSError: [Errno 98] Address already in use
  UDP port 9575 is already bound by a stale GUI/receiver process.

lgpio.error: 'GPIO busy'
  GPIO12 or GPIO25 is still claimed by a stale radar process.

Could not load Qt platform plugin / could not connect to display
  The GUI was launched without access to the Pi display session.
```

## Development Notes

Use `utility.helper.find_register_config_in_directory()` and
`find_setting_in_directory()` conventions when adding new configs. Each config
directory should contain exactly one matching register export and one matching
settings JSON file.

Use explicit interpreter paths in launchers when running from a virtual
environment. Child scripts should use `sys.executable` rather than hardcoded
`python` or `python3` so GUI and streamer processes share the same packages and
system-site configuration.

Avoid running examples from stale mirrored directories. Keep one canonical
MMW-HAT release tree for the Pi hardware, otherwise SPI bus defaults and venv
selection can diverge and produce misleading hardware errors.
