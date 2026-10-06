/* scan_radio.h — radio-environment scanning alongside CSI capture.
 *
 * Adds three metadata producers to csi_recv:
 *  - WIFI_DATA  : 802.11 frames seen in promiscuous mode (beacon, probe
 *                 req/resp, data) with SA/BSSID/SSID/RSSI/channel.
 *  - BLE_DATA   : NimBLE observer reports (addr, rssi, name, mfg data).
 *  - SELF_DATA  : identity line emitted once at boot.
 *
 * All output is serialized through one FreeRTOS queue so lines never
 * interleave inside CSI_DATA bursts. Producers drop when the queue is
 * full rather than blocking the Wi-Fi/BT tasks.
 */
#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- tuning (override via -D or edit) ---- */
#ifndef SCAN_QUEUE_LEN
#define SCAN_QUEUE_LEN          48      /* queued output lines */
#endif
#ifndef SCAN_LINE_MAX
#define SCAN_LINE_MAX           120     /* chars per emitted line */
#endif
#ifndef SCAN_EMIT_MIN_MS
#define SCAN_EMIT_MIN_MS        500     /* per-MAC dedupe window */
#endif
#ifndef WIFI_SWEEP_ENABLE
#define WIFI_SWEEP_ENABLE       1       /* slow channel rotation */
#endif
#ifndef SCAN_CYCLE_MS
#define SCAN_CYCLE_MS           2000    /* one off-channel dwell per cycle */
#endif
#ifndef SCAN_DWELL_MS
#define SCAN_DWELL_MS           150     /* ms spent off ch6 per hop */
#endif
#ifndef BLE_SCAN_ON_MS
#define BLE_SCAN_ON_MS          1000    /* BLE observer on-time */
#endif
#ifndef BLE_SCAN_PERIOD_MS
#define BLE_SCAN_PERIOD_MS      3000    /* BLE observer period */
#endif
#ifndef DEVICE_ROLE
#define DEVICE_ROLE             "rx"
#endif
#ifndef DEVICE_NAME
#define DEVICE_NAME             "thoth-csi-rx"
#endif
#ifndef OWNER_ID
#define OWNER_ID                "gad21"
#endif
#ifndef CSI_HOME_CHANNEL
#define CSI_HOME_CHANNEL        6
#endif

/* Queue a formatted line for serial output. Non-blocking; drops on
 * overflow. Safe from Wi-Fi/BT/task contexts (not from ISR). */
void scan_logf(const char *fmt, ...);

/* Copy a C6 CSI frame into the same bounded output queue as scan records.
 * No serial I/O in the Wi-Fi callback. Oversize/full-queue frames are
 * dropped explicitly and counted. header excludes len/first_word/data. */
bool scan_csi_log(const char *header, const int8_t *iq, uint16_t len,
                  bool first_word_invalid);

/* Start the output drain task. Call once before producers start. */
void scan_output_init(void);

/* Start promiscuous-frame harvesting + channel sweep supervisor.
 * Requires Wi-Fi started and promiscuous mode already enabled by
 * wifi_csi_init(). */
void scan_wifi_init(void);

/* Start NimBLE: duty-cycled observer + identity advertisement.
 * Requires scan_output_init() first. */
void scan_ble_init(void);

/* Emit the SELF_DATA identity line (role, factory MAC, name). */
void scan_emit_self(void);

#ifdef __cplusplus
}
#endif
