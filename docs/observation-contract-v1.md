# Observation Contract v1 — node → Brain normalized observations

Version: `observation/v1`. Frozen 2026-10-04. Changes require a new major
schema suffix (`*.v2`) — producers and consumers must tolerate unknown
fields and unknown schemas.

## Canonical flow

```
sources → observations → predictions/evidence → context state
        → context events → applications/agents/automations
```

Drivers may know about radar, cameras, BLE, GPS, IMUs. The observation
envelope and context contracts below are product- and hardware-agnostic.

## 1. Observation envelope

One observation = one normalized fact about the physical world produced by
a source. Raw high-rate payloads (camera frames, radar frames, IMU sample
streams) are NOT observations — they stay on the edge/capture path.
Observations are the curated uplink: low-rate, context-relevant, JSON-safe.

```json
{
  "schema": "ble.rssi.v1",
  "observation_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "batch_id": "1b9d6bcd-bbfd-4b5d-8b1d-2b0d7b3dcb6d",
  "timestamp": 1759612345.012,
  "source_id": "ble:hci0",
  "subject": "device:watch-8f3a",
  "value": {"rssi_dbm": -48, "tx_power_dbm": -59},
  "units": {"rssi_dbm": "dBm", "tx_power_dbm": "dBm"},
  "confidence": 0.9,
  "sequence": 4021,
  "provenance": {"observer": "device:node-42", "pipeline": "ble-observer/1"},
  "sync": {"clock_domain": "epoch", "uncertainty_s": 0.05}
}
```

| Field | Req | Notes |
|-------|-----|-------|
| `schema` | ✓ | Namespaced + versioned, e.g. `ble.rssi.v1`, `imu.window.v1`, `radio.presence.v1`. Consumers ignore unknown schemas. |
| `observation_id` | ✓ | Producer-generated UUID. Idempotency key — a re-sent observation MUST keep the same id. |
| `batch_id` | opt | Present when produced/transported as a batch. |
| `timestamp` | ✓ | Epoch seconds (float). Producer clock; `sync` carries the domain. |
| `source_id` | ✓ | Stable source identifier (`<driver>:<instance>`), e.g. `ble:hci0`, `radar-3f2a`, `camera-01`. |
| `subject` | opt | Target device/entity key (`device:…`, `person:…`, `space:…`). Absent for ambient observations. |
| `observer` | ✓ | Observing device/entity key — inside `provenance` (`provenance.observer`). |
| `value` | ✓ | Schema-defined measurement payload (JSON). |
| `units` | opt | Field → unit map. |
| `confidence` | opt | 0.0–1.0 where the producer has an estimate. |
| `sequence` | opt | Monotonic per source+subject; consumers use gaps as loss markers. |
| `provenance` | ✓ | `observer` required; may carry `pipeline`, `adapter`, `firmware`, etc. |
| `sync` | opt | Synchronization metadata (`clock_domain`, `uncertainty_s`, `offset_s`). |

### Privacy rules (non-negotiable)

- Persistent BLE MAC addresses MUST NOT appear as subject identity. Map
  bonded/enrolled devices to stable internal ids (`device:<uuid>`) on the
  node; unknown advertisers use anonymous rotating ids or none.
- No person names / product semantics inside drivers or schemas.
- Wi-Fi credentials, raw frames, and raw MACs never leave the node.

## 2. Node → Brain WebSocket frame

Observations ride the existing node→Brain socket
(`wss://<brain>/v1/node/ws`) — no new channel, no `ble_obs`-style
hardware-specific frames:

```json
{
  "type": "observation_batch",
  "id": "1b9d6bcd-bbfd-4b5d-8b1d-2b0d7b3dcb6d",
  "ts": 1759612345.9,
  "items": [ {"schema": "ble.rssi.v1", ...}, ... ]
}
```

- `id` = batch idempotency key (same id on re-send after reconnect).
- `items` = up to 200 observation envelopes per frame.
- Brain acknowledges implicitly: re-delivery after reconnect is safe
  because `observation_id` maps to `ContextEvidence.external_id`
  (unique per user) — duplicates collapse.

### Batching / retry

- Node buffers observations in a bounded in-memory queue
  (default 4 096) plus a JSONL spool (`~/.thoth/spool/observations.jsonl`,
  default cap 10 MB) so a Brain outage loses nothing context-relevant.
- Flush cadence: every tick (~0.5–2 s) or at 100 pending items,
  whichever first. Failed sends stay in the spool; flush resumes
  oldest-first on reconnect.
- Spool overflow drops the OLDEST items and emits a
  `observation.dropped.v1` counter observation — silent data loss is
  worse than a loss marker.

## 3. Brain storage mapping

Each observation becomes one `ContextEvidence` row:

| Observation | ContextEvidence |
|-------------|-----------------|
| `schema` | `evidence_key` |
| `observation_id` | `observation_id` + `external_id="obs:<id>"` (idempotent) |
| `timestamp` | `timestamp` |
| `source_id` | `source_id` |
| node `device_id` | `device_id` |
| `value`/`units`/`confidence`/`sequence`/`subject`/`batch_id` | `value` JSON carries `value`, `units`, `sequence`, `subject`, `batch_id`; `confidence` → column |
| `provenance` | `provenance` |

Estimators consume evidence and write `ContextState` rows with
`evidence_ids` back-references; transitions emit `ContextEvent`s.

## 4. Canonical context keys (initial set)

`occupancy.v1` `presence.v1` `location.geo.v1` `location.space.v1`
`location.zone.v1` `activity.posture.v1` `activity.motion.v1`
`activity.sleep.v1` `activity.fall.v1`

Every derived `ContextState` carries `estimator`, `confidence`,
`evidence_ids`, and `entity_id`. Occupancy ("a space is occupied") and
identity ("a specific person is present") are separate keys — radar may
produce `occupancy.v1` without any identity claim.

## 5. Initial observation schemas

| Schema | Emitted by | `value` |
|--------|-----------|---------|
| `ble.rssi.v1` | BLE observer | `{"rssi_dbm": int, "tx_power_dbm": int\|null, "addr_type": "public\|random\|anonymous"}` |
| `ble.presence.v1` | BLE observer | `{"seen": true, "window_s": float}` — batched "still here" for enrolled devices |
| `imu.window.v1` | connected wearable | `{"rate_hz": f, "samples": n, "axes": {"x":[...],"y":[...],"z":[...]}}` — decimated windows only |
| `observation.dropped.v1` | spool | `{"dropped": n, "reason": "spool_overflow"}` |
| `net.link.v1` | provisioning | `{"iface": "wlan0", "state": "connected", "ssid_hash": "…"}` |
| `context.state.v1` | node estimators | `{"key": "occupancy.v1", "value": {...}, "confidence": f, "estimator": "…", "evidence_ids": […], "version": n}` — node-side estimate transition (§4 keys) |

New schemas follow `<domain>.<measure>.v<N>` naming and are additive only.
