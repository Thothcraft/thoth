#!/bin/bash
# Thoth Services Setup Script

set -e

echo "=== Thoth Services Setup ==="

echo "Creating systemd service files..."

# Flask backend service
sudo tee /etc/systemd/system/thoth-backend.service > /dev/null <<EOF
[Unit]
Description=Thoth Flask Backend
After=network.target
Wants=network.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/opt/thoth
Environment=PATH=/opt/thoth/venv/bin
ExecStart=/opt/thoth/venv/bin/python src/backend/app.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Sensor collector service
sudo tee /etc/systemd/system/thoth-collector.service > /dev/null <<EOF
[Unit]
Description=Thoth Sensor Collector
After=network.target thoth-backend.service
Wants=network.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/opt/thoth
Environment=PATH=/opt/thoth/venv/bin
ExecStart=/opt/thoth/venv/bin/python src/sensors/sensehat_collector.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# WiFi manager service
sudo tee /etc/systemd/system/thoth-wifi.service > /dev/null <<EOF
[Unit]
Description=Thoth WiFi Manager
After=network.target
Before=thoth-backend.service

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/opt/thoth
Environment=PATH=/opt/thoth/venv/bin
ExecStart=/opt/thoth/venv/bin/python src/network/wifi_manager.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# PiSugar button handler service
sudo tee /etc/systemd/system/thoth-button.service > /dev/null <<EOF
[Unit]
Description=Thoth PiSugar Button Handler
After=network.target thoth-backend.service
Wants=network.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/opt/thoth
Environment=PATH=/opt/thoth/venv/bin
ExecStart=/opt/thoth/venv/bin/python src/utils/pisugar_handler.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "Enabling services..."
sudo systemctl enable thoth-backend
sudo systemctl enable thoth-collector
sudo systemctl enable thoth-wifi
sudo systemctl enable thoth-button

echo "Starting services..."
sudo systemctl start thoth-backend
sudo systemctl start thoth-wifi
sudo systemctl start thoth-button

echo "Services setup complete!"
echo "Service status:"
echo "Backend: $(sudo systemctl is-active thoth-backend)"
echo "WiFi: $(sudo systemctl is-active thoth-wifi)"
echo "Button: $(sudo systemctl is-active thoth-button)"
echo ""
echo "To start data collection: sudo systemctl start thoth-collector"
echo "To check service logs: journalctl -u thoth-<service-name> -f"
