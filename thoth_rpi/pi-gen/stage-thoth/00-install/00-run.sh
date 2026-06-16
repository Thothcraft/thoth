#!/bin/bash -e

install -d "${ROOTFS_DIR}/home/pi/thoth"
rsync -a --delete \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='logs/*' \
    --exclude='data/config/auth.json' \
    "${REPO_DIR}/" \
    "${ROOTFS_DIR}/home/pi/thoth/"

on_chroot << 'EOF'
chown -R pi:pi /home/pi/thoth
chmod +x /home/pi/thoth/thoth_rpi/setup/install.sh
chmod +x /home/pi/thoth/thoth_rpi/setup/prepare-image.sh
chmod +x /home/pi/thoth/thoth_rpi/setup/first-boot.sh
EOF
