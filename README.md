# Thoth Raspberry Pi Sensor Platform

Thoth is a Raspberry Pi edge sensor platform for synchronized DreamHat radar,
ESP32 WiFi CSI, USB camera, and Sense HAT capture. It provides a local Flask
dashboard, continuous one-minute collection, radar visualization and tracking,
model inference, and cloud synchronization.

## Fresh Raspberry Pi OS install

1. Flash Raspberry Pi OS 64-bit with Raspberry Pi Imager.
2. Configure the username, WiFi, hostname, and SSH in Imager.
3. Boot the Pi and clone this repository.
4. Run the one-file installer from the repository root:

```bash
cd thoth
sudo bash setup/first-boot.sh
```

The installer is safe to rerun. It installs system and Python dependencies,
enables SPI, grants hardware-device groups to the clone owner, creates `.venv`,
installs and starts `thoth.service` and `thoth-collector.service`, enables
Avahi/SSH, and configures `http://thoth.local:5000`.

If SPI was not previously enabled, reboot once after installation. Both Thoth
services are already enabled and will start automatically.

## Repository layout

```text
src/
  app.py                 Raspberry Pi dashboard entrypoint
  collector.py           Continuous one-minute collector
  backend/               Dashboard, capture, radar, models, and API code
setup/
  first-boot.sh           Complete fresh-OS installer
WS/MMW-HAT/              DreamHat radar driver, configs, and examples
config/                   Runtime settings and credentials
models/                   Deployed sensor models
data/                     Rolling capture folders (newest 300 retained)
```

This repository targets Raspberry Pi OS only. It does not include Windows or
macOS application variants.

## Services

```bash
systemctl status thoth.service
systemctl status thoth-collector.service
journalctl -u thoth-collector.service -f
```

The collector keeps the newest 300 minute folders by default.

## Sensors

- DreamHat BGT60TR13C radar over SPI/GPIO
- ESP32 CSI receiver over USB serial
- USB UVC camera when attached
- Sense HAT when attached

## License

MIT — see [LICENSE](LICENSE).
