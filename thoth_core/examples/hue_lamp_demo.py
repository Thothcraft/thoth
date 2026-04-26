#!/usr/bin/env python3
"""Demo: Thoth → Home Assistant → Philips Hue lamp.

This example shows how Thoth predictions flow to Home Assistant
via MQTT to control a Philips Hue light.

Prerequisites:
  1. Home Assistant running (e.g. on a RPi or Docker)
  2. MQTT broker installed (HA Add-on: Mosquitto)
  3. MQTT integration enabled in HA
  4. Philips Hue integration set up in HA
  5. pip install paho-mqtt

Run:
  python hue_lamp_demo.py

What happens:
  - Thoth publishes occupancy=ON via MQTT
  - HA discovers it as binary_sensor.thoth_occupancy
  - HA automation turns on the Hue lamp when occupancy is detected

The HA automation (add to configuration.yaml or UI):

  automation:
    - alias: "Thoth: Lights on when someone enters"
      trigger:
        - platform: state
          entity_id: binary_sensor.thoth_thoth_occupancy
          to: "on"
      action:
        - service: light.turn_on
          target:
            entity_id: light.living_room    # Your Hue lamp entity
          data:
            brightness_pct: 80
            color_name: warm_white

    - alias: "Thoth: Lights off when room empty"
      trigger:
        - platform: state
          entity_id: binary_sensor.thoth_thoth_occupancy
          to: "off"
          for: "00:05:00"                   # 5 min grace period
      action:
        - service: light.turn_off
          target:
            entity_id: light.living_room

    - alias: "Thoth: Dim lights when sleeping"
      trigger:
        - platform: state
          entity_id: binary_sensor.thoth_thoth_sleeping
          to: "on"
      condition:
        - condition: time
          after: "21:00"
      action:
        - service: light.turn_on
          target:
            entity_id: light.bedroom
          data:
            brightness_pct: 5
            color_name: red
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from thoth_core.ha_integration import HomeAssistantBridge


def main():
    # Connect to MQTT broker (defaults to localhost:1883)
    # Set MQTT_BROKER env var if your HA is on a different host
    bridge = HomeAssistantBridge(
        broker=os.getenv("MQTT_BROKER", "localhost"),
        device_name="Thoth-MacBook",
    )

    if not bridge.connect():
        print("Failed to connect to MQTT broker.")
        print("Make sure Mosquitto is running on your Home Assistant instance.")
        print("Set MQTT_BROKER=<ha-ip> if it's not on localhost.")
        return

    print("Connected to Home Assistant via MQTT!")
    print("Entities will appear in HA within seconds.")
    print()

    # Simulate predictions
    scenarios = [
        {"desc": "Someone enters the room", "prediction": {
            "occupancy": True, "motion": True, "sleeping": False,
            "activity": "walking", "confidence": 94.2, "occupancy_count": 1,
        }},
        {"desc": "Person sits down", "prediction": {
            "occupancy": True, "motion": False, "sleeping": False,
            "activity": "sitting", "confidence": 91.0, "occupancy_count": 1,
        }},
        {"desc": "Person falls asleep", "prediction": {
            "occupancy": True, "motion": False, "sleeping": True,
            "activity": "sleeping", "confidence": 88.5, "occupancy_count": 1,
        }},
        {"desc": "Room becomes empty", "prediction": {
            "occupancy": False, "motion": False, "sleeping": False,
            "activity": "empty", "confidence": 96.0, "occupancy_count": 0,
        }},
    ]

    for scenario in scenarios:
        print(f"→ {scenario['desc']}")
        bridge.publish_prediction(scenario["prediction"])
        print(f"  Published: {scenario['prediction']}")
        print()
        time.sleep(3)

    print("Demo complete! Check Home Assistant for entity updates.")
    print("Entities: binary_sensor.thoth_*, sensor.thoth_*")

    bridge.disconnect()


if __name__ == "__main__":
    main()
