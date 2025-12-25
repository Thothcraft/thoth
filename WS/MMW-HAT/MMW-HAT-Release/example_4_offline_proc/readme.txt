Overview This program collects and processes millimeter-wave (mmWave) data. It includes two scripts:

data_collection.py: For collecting data.
offline_proc.py: For processing the collected data.

How to Use

Step 1: Collect Data

Run the data_collection.py script: python data_collection.py
The data will be saved in the data/ folder as a file named mmw_spi_xxxxxxxx_xxxxxx_xxx.bin (e.g., mmw_spi_20241119_081114_994.bin).
Stop collection by terminating it.

Step 2: Process Data

Open the offline_proc.py script and replace the example data filename with the newly collected data file.
Run the offline_proc.py script: python offline_proc.py
The processed data will be saved in the mmw_proc/ folder.

Notes

An example data file is included for testing. Replace the filename in offline_proc.py with the new data file when you collect fresh data.
Ensure the offline_proc.py script is updated with the correct filename before running it.

File Structure
data/: Stores raw data files.
mmw_proc/: Stores processed data files.
data_collection.py: Collects mmWave data.
offline_proc.py: Processes collected data.