#!/bin/bash
# PiSugar Configuration Script

set -e

echo "=== PiSugar Configuration ==="

echo "Enabling I2C interface..."
# Enable I2C if not already enabled
sudo raspi-config nonint do_i2c 0

echo "Checking I2C devices..."
# Check if I2C devices are detected
if command -v i2cdetect >/dev/null 2>&1; then
    echo "I2C devices detected:"
    i2cdetect -y 1 || echo "No I2C devices found or permission denied"
else
    echo "i2cdetect not found, installing i2c-tools..."
    sudo apt install -y i2c-tools
fi

echo "Configuring PiSugar power manager..."
# Configure PiSugar power manager
if command -v pisugar-power-manager >/dev/null 2>&1; then
    # Set PiSugar model (adjust as needed)
    sudo pisugar-power-manager --model 'PiSugar 2 Pro' || echo "PiSugar configuration may need manual adjustment"
else
    echo "PiSugar power manager not found. Please ensure PiSugar is properly installed."
fi

echo "Creating PiSugar configuration..."
# Create basic PiSugar config
sudo mkdir -p /etc/pisugar
sudo tee /etc/pisugar/config.json > /dev/null <<EOF
{
    "model": "PiSugar 2 Pro",
    "button": {
        "single_tap": "custom",
        "double_tap": "custom", 
        "long_press": "shutdown"
    },
    "auto_power_on": true,
    "safe_shutdown_level": 5,
    "safe_shutdown_delay": 30
}
EOF

echo "Setting up button event handling..."
# Create button event script
sudo tee /usr/local/bin/pisugar-button-handler > /dev/null <<'EOF'
#!/bin/bash
# PiSugar button event handler

case "$1" in
    "single")
        # Toggle data collection
        if systemctl is-active --quiet thoth-collector; then
            systemctl stop thoth-collector
            echo "Data collection stopped"
        else
            systemctl start thoth-collector
            echo "Data collection started"
        fi
        ;;
    "double")
        # Start AP mode
        systemctl stop thoth-collector 2>/dev/null || true
        python3 /opt/thoth/src/network/wifi_manager.py --start-ap
        echo "AP mode started"
        ;;
    "long")
        # Safe shutdown
        echo "Shutting down..."
        shutdown -h now
        ;;
esac
EOF

sudo chmod +x /usr/local/bin/pisugar-button-handler

echo "Testing PiSugar connection..."
# Test PiSugar connection
python3 -c "
try:
    import sys
    sys.path.append('/opt/thoth/venv/lib/python*/site-packages')
    from sugarpie import PiSugar
    ps = PiSugar()
    print(f'PiSugar battery level: {ps.get_battery_level()}%')
    print('PiSugar connection: OK')
except Exception as e:
    print(f'PiSugar connection failed: {e}')
    print('This is normal if PiSugar is not connected')
" || echo "PiSugar test completed (may show errors if hardware not connected)"

echo "PiSugar configuration complete!"
echo ""
echo "Configuration summary:"
echo "- I2C interface enabled"
echo "- PiSugar power manager configured"
echo "- Button actions: single=toggle collection, double=AP mode, long=shutdown"
echo "- Safe shutdown at 5% battery"
echo ""
echo "To test button functionality:"
echo "  sudo /usr/local/bin/pisugar-button-handler single"
echo "  sudo /usr/local/bin/pisugar-button-handler double"
