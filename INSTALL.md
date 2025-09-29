# Thoth Installation Guide

This guide provides step-by-step instructions for setting up the Thoth portable research device.

## Prerequisites

### Hardware Requirements
- Raspberry Pi 4 (recommended) or Pi Zero 2 W
- Sense HAT (for IMU sensors)
- PiSugar 2 Pro (5000mAh) for Pi 4, or PiSugar 2 for Pi Zero
- MicroSD card (32GB+ recommended)
- Optional: Case for protection

### Software Requirements
- Raspberry Pi OS (latest version)
- Python 3.8+
- Git

## Step 1: Hardware Assembly

### 1.1 Raspberry Pi Setup
1. Flash Raspberry Pi OS to SD card using Raspberry Pi Imager
2. Enable SSH and configure WiFi during imaging (optional)
3. Insert SD card and boot the Pi

### 1.2 Sense HAT Installation
1. Power off the Pi
2. Carefully align and stack the Sense HAT on the Pi's GPIO pins
3. Ensure all pins are properly connected
4. Power on and verify detection with `i2cdetect -y 1`

### 1.3 PiSugar Installation
1. Power off the Pi
2. Attach PiSugar to the bottom of the Pi using pogo pins
3. Ensure proper alignment and connection
4. Charge PiSugar via micro-USB before first use

## Step 2: Software Installation

### 2.1 System Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Enable required interfaces
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_ssh 0

# Reboot to apply changes
sudo reboot
```

### 2.2 Download Thoth
```bash
# Clone or copy the Thoth project
cd /opt
sudo git clone <repository-url> thoth
sudo chown -R pi:pi /opt/thoth
cd /opt/thoth
```

### 2.3 Run Installation Scripts
```bash
# Make scripts executable
chmod +x setup/*.sh

# Install dependencies
./setup/install_deps.sh

# Configure services
./setup/enable_services.sh

# Configure PiSugar
./setup/configure_pisugar.sh
```

### 2.4 Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

Configure the following variables:
- `WIFI_SSID`: Your WiFi network name
- `WIFI_PASSWORD`: Your WiFi password
- `UPLOAD_URL`: Server URL for data uploads
- `API_KEY`: Authentication key for uploads

## Step 3: Testing and Verification

### 3.1 Service Status Check
```bash
# Check all services
sudo systemctl status thoth-backend
sudo systemctl status thoth-wifi
sudo systemctl status thoth-button

# View logs
journalctl -u thoth-backend -f
```

### 3.2 Hardware Testing
```bash
# Test I2C devices
i2cdetect -y 1

# Test Sense HAT
python3 -c "from sense_hat import SenseHat; sense = SenseHat(); print('Sense HAT OK')"

# Test PiSugar
python3 -c "from sugarpie import PiSugar; ps = PiSugar(); print(f'Battery: {ps.get_battery_level()}%')"
```

### 3.3 Web Interface Test
1. Find Pi's IP address: `hostname -I`
2. Open browser to `http://<pi-ip>:5000`
3. Verify web interface loads and shows device status

## Step 4: Data Collection Testing

### 4.1 Manual Collection Start
```bash
# Start data collection
sudo systemctl start thoth-collector

# Check data file
tail -f /opt/thoth/data/logs/sensors.json
```

### 4.2 Button Testing
1. Single press: Should toggle data collection
2. Double press: Should start AP mode
3. Long press: Should initiate shutdown

### 4.3 WiFi AP Mode Testing
1. Disconnect from WiFi or double-press button
2. Look for "Thoth-AP" network on phone/laptop
3. Connect with password "thoth123"
4. Browser should redirect to configuration portal

## Step 5: Brain Backend Integration

### 5.1 Device Registration
```bash
# Register device with Brain backend
curl -X POST "https://your-brain-server.com/thoth/device/register" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "unique-device-id",
    "device_name": "My Thoth Device",
    "hardware_info": {
      "model": "Pi 4",
      "sense_hat": true,
      "pisugar": "2 Pro"
    }
  }'
```

### 5.2 Data Upload Configuration
Update `.env` file:
```bash
UPLOAD_URL=https://your-brain-server.com/thoth/data/upload
API_KEY=your_jwt_token
```

## Troubleshooting

### Common Issues

#### Sense HAT Not Detected
```bash
# Check I2C
sudo i2cdetect -y 1
# Should show devices at 0x1c, 0x1e, 0x46, 0x5f

# Reinstall Sense HAT
sudo apt install --reinstall python3-sense-hat
```

#### PiSugar Connection Issues
```bash
# Check PiSugar service
sudo systemctl status pisugar-power-manager

# Manual I2C test
sudo i2cget -y 1 0x74 0x00  # Should return battery level
```

#### WiFi AP Mode Not Working
```bash
# Check hostapd
sudo systemctl status hostapd

# Check interface
ip addr show wlan0

# Restart WiFi manager
sudo systemctl restart thoth-wifi
```

#### Services Not Starting
```bash
# Check Python environment
source /opt/thoth/venv/bin/activate
python -c "import flask, sense_hat; print('Dependencies OK')"

# Check permissions
sudo chown -R pi:pi /opt/thoth
sudo chmod +x /opt/thoth/setup/*.sh
```

### Log Locations
- System logs: `journalctl -u thoth-*`
- Application logs: `/opt/thoth/data/logs/thoth.log`
- Sensor data: `/opt/thoth/data/logs/sensors.json`

### Performance Optimization

#### For Pi Zero
```bash
# Reduce collection rate
echo "COLLECTION_RATE=0.5" >> .env

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable avahi-daemon
```

#### For Extended Battery Life
```bash
# Configure power management
echo "dtoverlay=pi3-disable-wifi" >> /boot/config.txt  # If using ethernet
echo "dtoverlay=pi3-disable-bt" >> /boot/config.txt   # Disable Bluetooth
```

## Maintenance

### Regular Tasks
```bash
# Update system monthly
sudo apt update && sudo apt upgrade -y

# Clean old data (keep last 30 days)
find /opt/thoth/data/logs -name "*.json" -mtime +30 -delete

# Check disk usage
df -h /opt/thoth/data
```

### Backup Configuration
```bash
# Backup configuration
tar -czf thoth-backup-$(date +%Y%m%d).tar.gz /opt/thoth/.env /opt/thoth/src/backend/button_config.json

# Backup data
tar -czf thoth-data-$(date +%Y%m%d).tar.gz /opt/thoth/data/logs/
```

## Security Considerations

### Network Security
- Change default AP password in `.env`
- Use HTTPS for data uploads
- Consider VPN for remote access

### System Security
```bash
# Change default password
passwd

# Update SSH configuration
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no (if using keys)

# Enable firewall
sudo ufw enable
sudo ufw allow 22    # SSH
sudo ufw allow 5000  # Thoth web interface
```

## Support

For issues and questions:
1. Check logs: `journalctl -u thoth-* --since "1 hour ago"`
2. Verify hardware connections
3. Test individual components
4. Consult troubleshooting section above

## Next Steps

After successful installation:
1. Configure data upload to your research server
2. Set up automated data collection schedules
3. Develop custom analysis scripts
4. Consider building a research portal frontend
5. Explore machine learning integration for activity recognition
