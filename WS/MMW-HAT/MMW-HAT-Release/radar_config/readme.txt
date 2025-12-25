Overview
The radar_config/config_xxx/ folder contains radar configuration files. Each configuration is stored as a pair of files:

BGT60TR13C_export_registers_YYYYMMDD-HHMMSS.txt: The register map for the radar chip.
BGT60TR13C_settings_YYYYMMDD-HHMMSS.json: The human-readable configuration file.
These files must be generated together using the Radar Fusion GUI 3.5.4.

How to Use

Use Radar Fusion GUI 3.5.4 to create radar configurations. Ensure both files in the pair are generated at the same time.
Place the generated pair of files in the radar_config/config_track/files folder.

Important Notes

Do not modify the files manually, as this may lead to errors or corrupt configurations.

radar_config/config_xxx/: Contains the radar configuration files. Each configuration pair includes:
BGT60TR13C_export_registers_YYYYMMDD-HHMMSS.txt
BGT60TR13C_settings_YYYYMMDD-HHMMSS.json