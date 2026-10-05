# Thothcraft platform rollout — deliverables

Multi-device sensor-fusion + physical-context platform: packaged node
daemon, BLE surfaces, provisioning, observation uplink, node-side
context estimators, entity/relationship graph.

## Changed / added files

| File | Change |
|------|--------|
| `setup/first-boot.sh` | Rewritten: packaged `thoth` daemon install, BlueZ/NetworkManager/Polkit deps, `bluetooth`+`netdev` groups, polkit rule, legacy systemd unit migration → single `thoth.service` |
| `thoth/settings/store.py` | `local_token`/`device_id` get-or-generate made atomic (race fix) |
| `thoth/daemon/service.py` | `_start_brain_ws` lock; BLE + provisioning + watches lifecycle; `emit_observation` → estimator consume; `context()` merges `estimates`; conformance/domain detection; `ble_enroll`/`entities`/`net_*`/`calibrate_zone` |
| `thoth/observations.py` | `Observation` envelope + durable `ObservationSpool` + `build_batch` (observation/v1) |
| `thoth/events.py` | `EventHub` — local SSE fanout |
| `thoth/daemon/brain_ws.py` | `observation_batch` uplink + ack-driven spool drain |
| `thoth/local_api/server.py` | `/api/v1/{capabilities,predictions,stream,sources/*,net,ble/*,entities,relations,context/calibrate}` |
| `thoth/bluetooth/` | `subsystem.py` (observer/central/duty cycle), `backend.py` (bleak/Null), `known.py` (device map), `gatt.py` (commissioning peripheral), `pinetime.py` (InfiniTime link) |
| `thoth/provisioning/` | `manager.py` (state machine + AP fallback), `wifi.py` (nmcli) |
| `thoth/estimators.py` | `EstimatorHub`: `occupancy.v1`, `activity.motion.v1`, `location.zone.v1` + RSSI fingerprints |
| `thoth/entities.py` | `EntityStore` — entities + `wears`/`owns`/… edges |
| `thoth/cli/main.py` | `sources tail`, `context-watch`, `entities`, `relationships` |
| `Brain/server/endpoints/node_ws.py` | `observation_batch` → `ContextEvidence` (idempotent on `external_id`) |
| `docs/observation-contract-v1.md`, `docs/bluetooth-architecture.md` | contract + BLE architecture |

## Data / migrations

New files under `~/.thoth/` (auto-created): `spool/observations.jsonl`
(10 MB cap), `ble_devices.json`, `provisioning.json`,
`fingerprints.json`, `entities.json`, plus `ble.salt` in
`config.json`. Legacy `first-boot` systemd units are disabled and
replaced by `thoth.service`; nothing else migrates.

## Schemas (observation/v1)

`ble.rssi.v1`, `ble.presence.v1`, `imu.window.v1`,
`observation.dropped.v1`, `net.link.v1`, `context.state.v1` — see
`docs/observation-contract-v1.md` §5. Privacy: no raw MACs or SSIDs on
the wire (`device:ble:<hmac>`, `device:<uuid>`, `ssid_hash`).

## Back-compat notes

- `BRAIN_AUTH_TOKEN` env still pairs the node; `BRAIN_URL` overrides.
- `local_token` rotates only if `config.json` is deleted.
- All new `/api/v1/*` routes require the bearer token (same as before).
- `context.state.v1` observations appear in the uplink — Brain treats
  unknown schemas as evidence only; additive by contract.

## Install / flash

```
curl -fsSL https://…/install.sh | bash          # → setup/first-boot.sh
thoth daemon                                    # foreground
systemctl status thoth.service                  # packaged service
thoth sources tail ble:hci0 --follow            # live BLE observations
thoth entities add person --name alice
thoth relationships --rel wears
```

## Known lint noise

`thoth/bluetooth/gatt.py` triggers Pyright "unexpected token" warnings
on `dbus_next` signature-string annotations (`-> "a{sv}"`) — required
D-Bus DSL, runtime-correct.
