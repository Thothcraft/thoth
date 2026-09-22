#!/usr/bin/env bash
# Assign a friendly, unique hostname: thoth-<name>.local
#
# <name> is a random month, person, or city name drawn from THOTH_NAMES.
# Re-running picks a new name. Set THOTH_HOSTNAME to force a specific name:
#   THOTH_HOSTNAME=thoth-kyoto sudo bash setup/device-hostname.sh
#
# Safe to run on any Thoth device to (re)name it. Requires root for
# hostnamectl + /etc/hosts + avahi restart.

set -euo pipefail

THOTH_NAMES=(
    january february march april may june july august september october november december
    alex amina andre aria arjun ben carlos chen clara dario elena emma felix freya
    hana idris ivy jade jonas kai keiko leila liam luca mara mateo maya milo nina
    noah omar oscar priya ravi rosa sara soren theo uma vera yuki zara
    amsterdam athens austin berlin cairo chicago denver dublin geneva hanoi havana
    kyoto lisbon london madrid manila nairobi oslo paris perth prague quito reykjavik
    rome seoul sydney tokyo toronto venice vienna zurich
)

pick_name() {
    local candidate
    for _attempt in 1 2 3 4 5; do
        candidate="thoth-${THOTH_NAMES[$((RANDOM % ${#THOTH_NAMES[@]}))]}"
        # Skip names already advertised on the local network (mDNS collision).
        if ! getent hosts "${candidate}.local" >/dev/null 2>&1 \
            && ! ping -c1 -W1 "${candidate}.local" >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    printf 'thoth-%s' "${THOTH_NAMES[$((RANDOM % ${#THOTH_NAMES[@]}))]}"
}

if [ -z "${THOTH_HOSTNAME:-}" ]; then
    DEVICE_HOSTNAME="$(pick_name)"
else
    DEVICE_HOSTNAME="$THOTH_HOSTNAME"
fi
DEVICE_HOSTNAME="$(echo "$DEVICE_HOSTNAME" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9-')"
[ -n "$DEVICE_HOSTNAME" ] || DEVICE_HOSTNAME="thoth"

hostnamectl set-hostname "$DEVICE_HOSTNAME" || true
if grep -q '^127.0.1.1' /etc/hosts; then
    sed -i "s/^127.0.1.1.*/127.0.1.1\t$DEVICE_HOSTNAME/" /etc/hosts
else
    printf '127.0.1.1\t%s\n' "$DEVICE_HOSTNAME" >> /etc/hosts
fi
systemctl restart avahi-daemon 2>/dev/null || true

echo "Hostname set: http://$DEVICE_HOSTNAME.local:5000"
