# Thoth Raspberry Pi Image Flow

This is the current deployment flow for Thoth on Raspberry Pi.

## Goal

Build a custom Pi image that:

1. Stores WiFi credentials in Raspberry Pi Imager before first boot.
2. Boots straight onto the saved LAN.
3. Starts `thoth-web` and `thoth-collector` automatically.
4. Serves the UI at `http://thoth.local:5000`.
5. Detects supported sensors and starts collecting immediately.
6. Keeps the newest 300 minute folders in the clone's `data` directory.

## What Is Removed

- Nodogsplash
- Hotspot AP mode
- Captive portal WiFi setup
- In-browser WiFi scan/connect forms
- Old `wifi_configured.flag` flow

## Recommended Image Build

Use Raspberry Pi Imager:

1. Choose Raspberry Pi OS Lite 64-bit.
2. Open advanced settings.
3. Set hostname to `thoth`.
4. Set username/password.
5. Enter WiFi SSID and password.
6. Enable SSH if needed.
7. Write the card.

On first boot:

- The Pi joins the WiFi from Imager.
- Avahi advertises `thoth.local`.
- The web app starts on port `5000`.
- The collector starts and writes minute folders.

## First Boot Script

From a fresh clone on Raspberry Pi OS, run:

```bash
cd thoth
sudo bash setup/first-boot.sh
```

The command is safe to rerun. It:

- installs system and Python dependencies
- enables SPI for the DreamHat radar
- grants the clone owner access to SPI, GPIO, serial, and video devices
- installs systemd units using the actual clone owner instead of a hard-coded user
- starts or restarts `thoth.service` and `thoth-collector.service`
- sets the hostname and enables `thoth.local` through Avahi

The script does not alter NetworkManager or hostapd. If SPI was disabled before
installation, reboot once when prompted; the services are already enabled and
will start automatically afterward.

## Access

- Web app: `http://thoth.local:5000`
- If mDNS is not supported by a client, use the Pi's LAN IP instead.

## Reliability Note

`thoth.local` is mDNS, not guaranteed DNS. For the strongest cross-device reliability, use:

- router DHCP reservation for the Pi IP
- router/local DNS entry for `thoth`

The Pi can advertise `thoth.local`, but it cannot force every client network stack to resolve mDNS if the client or network blocks it.
