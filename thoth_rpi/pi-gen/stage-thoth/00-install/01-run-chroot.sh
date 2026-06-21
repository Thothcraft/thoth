#!/bin/bash -e

cd /home/pi/Desktop/thoth
./thoth_rpi/setup/install.sh
THOTH_SKIP_INSTALL=1 ./thoth_rpi/setup/prepare-image.sh
