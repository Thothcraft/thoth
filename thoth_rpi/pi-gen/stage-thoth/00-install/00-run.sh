#!/bin/bash -e

install -d "${ROOTFS_DIR}/home/pi/Desktop/thoth"
rsync -a --delete \
    "${STAGE_DIR}/files/thoth/" \
    "${ROOTFS_DIR}/home/pi/Desktop/thoth/"

on_chroot << 'EOF'
install -d -o pi -g pi /home/pi/Desktop/thoth/data/config
chown -R pi:pi /home/pi/Desktop/thoth
chmod +x /home/pi/Desktop/thoth/capture_dreamhat_minute.py
chmod +x /home/pi/Desktop/thoth/capture_dreamhat_minute.sh
chmod +x /home/pi/Desktop/thoth/thoth_rpi/setup/install.sh
chmod +x /home/pi/Desktop/thoth/thoth_rpi/setup/prepare-image.sh
chmod +x /home/pi/Desktop/thoth/thoth_rpi/setup/first-boot.sh
EOF
