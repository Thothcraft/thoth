# Bluetooth architecture — BlueZ subsystem on the node

Phases 2–4, 9–10 of the Thothcraft rollout. The subsystem is
`thoth.bluetooth.BluetoothSubsystem`; the daemon owns it as `self._ble`.

## Roles

| Role | Implementation | Used by |
|------|----------------|---------|
| **Observer** | `BleakScanner` on an asyncio loop thread | RSSI + presence observations |
| **Central** | managed `BleakClient` sessions w/ backoff | wearable links, provisioning peers |
| **Peripheral** | `GattApplication` via `dbus_next` + `LEAdvertisingManager1` | commissioning (unprovisioned nodes only) |

All hardware deps are import-guarded — on a host without BlueZ the
subsystem reports `capability().present = False` and stays inert.

## Duty cycle

`ble.duty_on_s` / `ble.duty_off_s` config keys control the scan window;
`0` on-window = continuous scanning. `_sweep_task` (every
`ble.sweep_s`, default 10 s) drives presence edges: devices silent
longer than `ble.gone_after_s` (default 45 s) flip to `seen:false`.

## Identity + privacy (contract §1)

- `ble_salt` (config, generated once) keys HMAC-SHA256 over the
  normalized MAC → `addr_hash` — the node-scoped anonymous id.
- Unknown advertisers → subject `device:ble:<hmac12>` — stable on-node,
  unlinkable off-node. Raw MACs are stored only in
  `~/.thoth/ble_devices.json` and never appear in any emitted field.
- Enrolled devices → subject `device:<uuid>` from `KnownDeviceStore`
  (enroll via `POST /api/v1/ble/enroll`, optionally binding a
  `person:*` entity with a `wears` edge).

## Emitted schemas

- `ble.rssi.v1` — per subject, first sighting + ≥4 dB EMA change +
  `ble.obs_min_interval_s` (default 10 s) heartbeat.
- `ble.presence.v1` — enrolled devices only: `seen:true` on sighting
  edge (and `ble.presence_hb_s` heartbeat), `seen:false` on the gone
  edge.
- `imu.window.v1` — PineTime RawMotion windows via `PinetimeLink`.

## Commissioning (peripheral role)

`GattApplication` (service `a0630100-…`) is registered while the node is
unprovisioned and torn down on success:

| Char | Flags | Payload |
|------|-------|---------|
| `a0630101` | write | `{"ssid","psk","hidden"}` → `ProvisionManager.provision()` |
| `a0630102` | read+notify | provisioning state JSON |
| `a0630103` | read | scanned networks `[{ssid,signal,security}]` |
| `a0630104` | read | node `device_id` |

## AP fallback

`provisioning.ap_after_failures` (default 3) failed credential attempts
broadcast `thoth-setup-<id6>` with a random PSK (visible in
`GET /api/v1/net`). Any credential write — BLE or local API — tears the
AP down and rejoins STA.

## Watch link (InfiniTime)

`PinetimeLink` central-connects enrolled `kind:"watch"` devices:
subscribes `00030002-…` (RawMotion) into `imu.window.v1`, pushes
context transitions to `00030003-…` (custom context characteristic —
requires matching firmware).
