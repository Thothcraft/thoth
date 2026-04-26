"""Home Assistant MQTT Integration for Thoth.

Publishes ML predictions as MQTT discovery messages so they appear
automatically as entities in Home Assistant.

Setup:
  1. Install an MQTT broker (Mosquitto) on your HA host or use the HA add-on
  2. Enable MQTT integration in HA (Settings → Integrations → MQTT)
  3. Set MQTT_BROKER in your .env (defaults to localhost)

Thoth publishes:
  - binary_sensor.thoth_occupancy
  - binary_sensor.thoth_motion
  - binary_sensor.thoth_sleeping
  - binary_sensor.thoth_pet_present
  - sensor.thoth_activity
  - sensor.thoth_confidence
  - sensor.thoth_occupancy_count
"""

import json
import logging
import os
import threading
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logger.warning("paho-mqtt not installed — HA integration disabled")


# MQTT Discovery prefix (HA default)
DISCOVERY_PREFIX = "homeassistant"
DEVICE_ID_PREFIX = "thoth"


class HomeAssistantBridge:
    """Bridges Thoth predictions to Home Assistant via MQTT auto-discovery."""

    def __init__(
        self,
        broker: str = None,
        port: int = None,
        username: str = None,
        password: str = None,
        device_name: str = None,
    ):
        self.broker = broker or os.getenv("MQTT_BROKER", "localhost")
        self.port = port or int(os.getenv("MQTT_PORT", "1883"))
        self.username = username or os.getenv("MQTT_USERNAME", "")
        self.password = password or os.getenv("MQTT_PASSWORD", "")
        self.device_name = device_name or os.getenv("DEVICE_NAME", "Thoth")
        self.device_id = f"{DEVICE_ID_PREFIX}_{self.device_name.lower().replace(' ', '_')}"

        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._lock = threading.Lock()

        # Entity definitions
        self._entities = {
            "occupancy": {
                "component": "binary_sensor",
                "config": {
                    "name": "Occupancy",
                    "device_class": "occupancy",
                    "payload_on": "ON",
                    "payload_off": "OFF",
                },
            },
            "motion": {
                "component": "binary_sensor",
                "config": {
                    "name": "Motion",
                    "device_class": "motion",
                    "payload_on": "ON",
                    "payload_off": "OFF",
                },
            },
            "sleeping": {
                "component": "binary_sensor",
                "config": {
                    "name": "Sleeping",
                    "icon": "mdi:sleep",
                    "payload_on": "ON",
                    "payload_off": "OFF",
                },
            },
            "pet_present": {
                "component": "binary_sensor",
                "config": {
                    "name": "Pet Present",
                    "icon": "mdi:dog",
                    "payload_on": "ON",
                    "payload_off": "OFF",
                },
            },
            "activity": {
                "component": "sensor",
                "config": {
                    "name": "Activity",
                    "icon": "mdi:walk",
                },
            },
            "confidence": {
                "component": "sensor",
                "config": {
                    "name": "Confidence",
                    "unit_of_measurement": "%",
                    "icon": "mdi:percent",
                },
            },
            "occupancy_count": {
                "component": "sensor",
                "config": {
                    "name": "Occupancy Count",
                    "icon": "mdi:counter",
                },
            },
        }

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to the MQTT broker and publish discovery messages."""
        if not MQTT_AVAILABLE:
            logger.error("paho-mqtt not installed")
            return False

        try:
            self._client = mqtt.Client(client_id=self.device_id)
            if self.username:
                self._client.username_pw_set(self.username, self.password)

            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect

            logger.info("Connecting to MQTT broker %s:%s …", self.broker, self.port)
            self._client.connect(self.broker, self.port, keepalive=60)
            self._client.loop_start()

            # Wait briefly for connection
            for _ in range(30):
                if self._connected:
                    break
                time.sleep(0.1)

            if self._connected:
                self._publish_discovery()
                logger.info("Home Assistant MQTT bridge connected")
            return self._connected

        except Exception as e:
            logger.error("MQTT connection failed: %s", e)
            return False

    def disconnect(self):
        """Disconnect from the MQTT broker."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected (rc=%s)", rc)
        else:
            logger.error("MQTT connection refused (rc=%s)", rc)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            logger.warning("MQTT disconnected unexpectedly (rc=%s)", rc)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _device_payload(self) -> Dict[str, Any]:
        """Common device block for HA discovery."""
        return {
            "identifiers": [self.device_id],
            "name": self.device_name,
            "manufacturer": "Thothcraft",
            "model": "Thoth Edge Sensor",
            "sw_version": "1.0.0",
        }

    def _publish_discovery(self):
        """Publish MQTT discovery messages for all entities."""
        for entity_key, entity_def in self._entities.items():
            component = entity_def["component"]
            unique_id = f"{self.device_id}_{entity_key}"
            state_topic = f"thoth/{self.device_id}/{entity_key}"
            discovery_topic = f"{DISCOVERY_PREFIX}/{component}/{unique_id}/config"

            config = {
                **entity_def["config"],
                "unique_id": unique_id,
                "state_topic": state_topic,
                "device": self._device_payload(),
            }

            self._client.publish(
                discovery_topic,
                json.dumps(config),
                retain=True,
            )
            logger.debug("Published discovery: %s", discovery_topic)

    # ------------------------------------------------------------------
    # State publishing
    # ------------------------------------------------------------------

    def publish_prediction(self, prediction: Dict[str, Any]):
        """Publish a model prediction to Home Assistant.

        Args:
            prediction: dict with keys like:
                {
                    "occupancy": True,
                    "motion": True,
                    "sleeping": False,
                    "pet_present": False,
                    "activity": "walking",
                    "confidence": 94.2,
                    "occupancy_count": 2,
                }
        """
        if not self._connected or not self._client:
            return

        for key, value in prediction.items():
            if key not in self._entities:
                continue

            state_topic = f"thoth/{self.device_id}/{key}"
            entity_def = self._entities[key]

            if entity_def["component"] == "binary_sensor":
                payload = "ON" if value else "OFF"
            else:
                payload = str(value)

            self._client.publish(state_topic, payload, retain=True)

        logger.debug("Published prediction: %s", prediction)

    def publish_activity(self, activity: str, confidence: float):
        """Convenience: publish an activity + confidence update."""
        self.publish_prediction({
            "activity": activity,
            "confidence": round(confidence, 1),
            "motion": activity not in ("idle", "empty", "sleeping"),
            "sleeping": activity == "sleeping",
            "occupancy": activity not in ("empty", "room_empty"),
        })
