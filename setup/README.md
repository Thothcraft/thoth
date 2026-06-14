# Thoth Raspberry Pi Image Flow

This is the current deployment flow for Thoth on Raspberry Pi.

## Goal

Build a custom Pi image that:

1. Stores WiFi credentials in Raspberry Pi Imager before first boot.
2. Boots straight onto the saved LAN.
3. Starts `thoth-web` and `thoth-collector` automatically.
4. Serves the UI at `http://thoth.local:5000`.
5. Detects supported sensors and starts collecting immediately.
6. Keeps the newest 100 minute folders in `/home/pi/Desktop/data`.

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

`setup/first-boot.sh` now performs only:

- hostname setup
- cleanup of legacy hotspot/captive files
- start or restart `thoth-web.service`
- start or restart `thoth-collector.service`

## Access

- Web app: `http://thoth.local:5000`
- If mDNS is not supported by a client, use the Pi's LAN IP instead.

## Reliability Note

`thoth.local` is mDNS, not guaranteed DNS. For the strongest cross-device reliability, use:

- router DHCP reservation for the Pi IP
- router/local DNS entry for `thoth`

The Pi can advertise `thoth.local`, but it cannot force every client network stack to resolve mDNS if the client or network blocks it.
