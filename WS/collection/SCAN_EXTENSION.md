# Radio-scan extension (BLE + Wi-Fi metadata alongside CSI)

Both firmwares were extended so every node in the CSI pair also produces
radio-environment metadata. Goal: RSSI maps (Wi-Fi + BLE) time-aligned with
CSI, plus self-identification so observers can anchor radios to
people/roles/positions.

## Serial line protocol (all types share the UART/USB-JTAG stream)

| Prefix       | Fields                                                        |
|--------------|---------------------------------------------------------------|
| `CSI_DATA`   | `seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,ch,ts,sig_len,rx_state,len,first_word,"[iq...]"` (unchanged) |
| `WIFI_DATA`  | `ms,sa_mac,kind,rssi,channel,"ssid"` — kind ∈ `bcn,prb_req,prb_rsp,deauth,asoc,data` |
| `BLE_DATA`   | `ms,addr,addr_type,rssi,tx_power,"name","mfg_hex8"`           |
| `SELF_DATA`  | `role,base_mac,"name","owner"` — emitted once at boot          |

Quoted fields are sanitized (no `"` or `,`). `tx_power` is `127` when the
advertiser didn't include it. `BLE_DATA` addr is the MSB-first printed form.

## csi_recv changes

- `main/scan_radio.h/.c` — output queue (`scan_logf`, drops on full), a
  promiscuous-frame harvester (mgmt + data frames, per-MAC dedupe), and a
  channel-sweep task: anchored on ch 6, one 150 ms hop to another channel
  every 2 s → ≈7 % CSI loss, full 13-channel sweep every ~26 s.
- `main/ble_radio.c` — NimBLE observer, duty-cycled 1 s on / 2 s off
  (BLE costs CSI frames while its scan windows run — this is where the
  allowed rate reduction goes), plus a non-connectable identity
  advertisement (`thoth-csi-rx`, mfg `0xFFFF + "gad21"`).
- `app_main.c` — `scan_output_init()` → `scan_emit_self()` →
  `scan_wifi_init()` → `scan_ble_init()` after `wifi_csi_init()`.

Tuning knobs are `#define`s at the top of `scan_radio.h`
(`SCAN_EMIT_MIN_MS`, `SCAN_CYCLE_MS`, `SCAN_DWELL_MS`,
`BLE_SCAN_ON_MS`, `BLE_SCAN_PERIOD_MS`, `DEVICE_NAME`, `OWNER_ID`,
`WIFI_SWEEP_ENABLE`).

## csi_send changes

- `main/ble_adv.h/.c` — NimBLE non-connectable advertiser only:
  `thoth-csi-tx` + owner tag. ~1 ms burst/s, negligible against the 1 kHz
  ESP-NOW duty.

## sdkconfig

`BT_ENABLED`, `BT_NIMBLE_ENABLED` (+ role minimization) and
`PARTITION_TABLE_SINGLE_APP_LARGE` (NimBLE pushes the app past the ~1 MB
`singleapp` partition) were added to `sdkconfig.defaults` in both projects.
**Existing `sdkconfig` files override defaults — delete them or run
`idf.py fullclean`/`set-target` before building**, otherwise BT stays off
and the app may not fit.

## Build / flash

Requires ESP-IDF ≥ 5.3 (sdkconfig was generated with 5.5.0):

```sh
idf.py set-target esp32c6 build flash -p <PORT>
```

Both consoles run at 921600 baud on UART0 **and** mirror onto the native
USB-Serial-JTAG port (`/dev/ttyACM0` on the Pi / `COMx` on Windows) —
that is the port the whispy adapter reads; USB-JTAG throughput is not
baud-limited, so the extra lines cost no link bandwidth. Remote flash via
the Pi works too: `esptool.py --chip esp32c6 -p /dev/ttyACM0 write_flash
0x0 firmware.bin` (auto-reset over DTR/RTS already proven).

## Physical anchoring

- `SELF_DATA` reports the **factory base MAC** (STA MAC is spoofed to the
  shared `1a:00:00:00:00:00` for ESP-NOW, so it can't identify hardware).
- Each receiver's samples are keyed by its `sensor_id` in whispy
  (e.g. `csi-55f7` on chen vs the laptop unit) — attach position/orientation
  to the sensor's room placement in the thoth metadata (`room/v1` devices:
  `pos`/`yaw`), then every `*_scan` sample is physically attributed.
- BLE adv names carry the owner tag; phones/watches should advertise the
  same convention (name + mfg `0xFFFF<owner>`) for person-level attachment.
- A third RSSI vantage point is available for free: the Pi's onboard
  Wi-Fi/BLE radios (host-side `iw`/`bluetoothctl` scan — future
  `whispy-sensor-radio` adapter).

## Not done / later

- **Zigbee (802.15.4)** on the C6: possible, but it shares RF time with
  Wi-Fi+BLE on one frontend — recommend a dedicated role or dongle rather
  than adding it to csi_recv.
- RSSI→position multilateration model consuming the `wifi_scan`/`ble_scan`
  payloads (downstream whispy model work).
