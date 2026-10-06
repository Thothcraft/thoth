/* ble_radio.c — NimBLE observer + identity beacon for csi_recv.
 *
 * Observer: duty-cycled passive scan (BLE_SCAN_ON_MS every
 * BLE_SCAN_PERIOD_MS) — the single 2.4 GHz frontend is shared with the
 * CSI receiver, so scan windows cost CSI frames. Advertisers emit one
 * BLE_DATA line each (per-addr dedupe inside BLE_EMIT_MIN_MS).
 *
 * Advertiser: non-connectable ADV_NONCONN carrying the device name and
 * a manufacturer payload (company 0xFFFF + OWNER_ID) so other thoth
 * scanners can anchor "who is this radio" to a person/role. Runs
 * continuously — advertising is ~1 ms bursts and costs almost nothing.
 */

#include "scan_radio.h"

#include <stdio.h>
#include <string.h>

#include "esp_log.h"
#include "esp_timer.h"

#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/ble_hs_adv.h"
#include "host/ble_uuid.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "ble_radio";

#ifndef BLE_EMIT_MIN_MS
#define BLE_EMIT_MIN_MS   1000
#endif

/* ------------------------------------------------------------------ */
/* per-address dedupe                                                  */
/* ------------------------------------------------------------------ */

#define BLE_SEEN_SLOTS 48
static struct { uint8_t addr[6]; int64_t last_us; } s_seen[BLE_SEEN_SLOTS];

static bool ble_seen_recently(const uint8_t *addr)
{
    int64_t now = esp_timer_get_time();
    int slot = addr[5] % BLE_SEEN_SLOTS;
    for (int i = 0; i < 4; i++) {
        int s = (slot + i) % BLE_SEEN_SLOTS;
        if (!memcmp(s_seen[s].addr, addr, 6)) {
            if (now - s_seen[s].last_us < (int64_t)BLE_EMIT_MIN_MS * 1000)
                return true;
            s_seen[s].last_us = now;
            return false;
        }
        if (s_seen[s].last_us == 0) {
            memcpy(s_seen[s].addr, addr, 6);
            s_seen[s].last_us = now;
            return false;
        }
    }
    memcpy(s_seen[slot].addr, addr, 6);
    s_seen[slot].last_us = now;
    return false;
}

/* ------------------------------------------------------------------ */
/* observer                                                            */
/* ------------------------------------------------------------------ */

static void to_hex(const uint8_t *in, int n, char *out)
{
    static const char *h = "0123456789abcdef";
    for (int i = 0; i < n; i++) {
        out[i * 2]     = h[in[i] >> 4];
        out[i * 2 + 1] = h[in[i] & 0xf];
    }
    out[n * 2] = 0;
}

static int ble_gap_scan_cb(struct ble_gap_event *event, void *arg)
{
    if (event->type != BLE_GAP_EVENT_DISC) return 0;
    struct ble_hs_adv_fields f;
    if (ble_hs_adv_parse_fields(&f, event->disc.data,
                                event->disc.length_data) != 0) return 0;

    const uint8_t *a = event->disc.addr.val;
    if (ble_seen_recently(a)) return 0;

    char name[24] = "-";
    if (f.name_len > 0) {
        int n = f.name_len < (int)sizeof(name) - 1 ? f.name_len
                                                 : (int)sizeof(name) - 1;
        for (int i = 0; i < n; i++) {
            uint8_t c = f.name[i];
            name[i] = (c >= 32 && c < 127 && c != '"' && c != ',')
                      ? (char)c : '.';
        }
        name[n] = 0;
    }
    char mfg[17] = "-";                   /* first 8 mfg bytes as hex */
    if (f.mfg_data_len > 0) {
        int n = f.mfg_data_len < 8 ? f.mfg_data_len : 8;
        to_hex(f.mfg_data, n, mfg);
    }
    int txp = (f.tx_pwr_lvl_is_present) ? (int)f.tx_pwr_lvl : 127;

    /* addr reversed for print — BLE addrs are LSB-first on the wire */
    scan_logf("BLE_DATA,%lu,%02x:%02x:%02x:%02x:%02x:%02x,%d,%d,%d,\"%s\",\"%s\"",
              (unsigned long)(esp_timer_get_time() / 1000),
              a[5], a[4], a[3], a[2], a[1], a[0],
              (int)event->disc.addr.type, (int)event->disc.rssi, txp,
              name, mfg);
    return 0;
}

static void ble_start_scan(void)
{
    struct ble_gap_disc_params p = {
        .itvl            = 0x50,          /* 50 ms */
        .window          = 0x30,          /* 30 ms */
        .filter_policy   = BLE_HCI_SCAN_FILT_NO_WL,
        .limited         = 0,
        .passive         = 1,
        .filter_duplicates = 0,
    };
    int rc = ble_gap_disc(BLE_OWN_ADDR_PUBLIC, BLE_HS_FOREVER, &p,
                          ble_gap_scan_cb, NULL);
    if (rc != 0 && rc != BLE_HS_EALREADY)
        ESP_LOGW(TAG, "scan start rc=%d", rc);
}

/* ------------------------------------------------------------------ */
/* identity advertisement                                              */
/* ------------------------------------------------------------------ */

static void ble_start_advertising(void)
{
    /* mfg payload: company id 0xFFFF + owner tag */
    uint8_t mfg[2 + 8] = {0xFF, 0xFF};
    strncpy((char *)mfg + 2, OWNER_ID, 8);

    struct ble_hs_adv_fields fields = {0};
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.name = (uint8_t *)DEVICE_NAME;
    fields.name_len = strlen(DEVICE_NAME);
    fields.name_is_complete = 1;
    fields.mfg_data = mfg;
    fields.mfg_data_len = 2 + strlen(OWNER_ID);

    int rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        ESP_LOGW(TAG, "adv fields rc=%d", rc);
        return;
    }
    struct ble_gap_adv_params ap = {0};
    ap.conn_mode = BLE_GAP_CONN_MODE_NON;
    ap.disc_mode = BLE_GAP_DISC_MODE_GEN;
    ap.itvl_min = BLE_GAP_ADV_ITVL_MS(1000);
    ap.itvl_max = BLE_GAP_ADV_ITVL_MS(1500);
    rc = ble_gap_adv_start(BLE_OWN_ADDR_PUBLIC, NULL, BLE_HS_FOREVER,
                           &ap, NULL, NULL);
    if (rc != 0) ESP_LOGW(TAG, "adv start rc=%d", rc);
}

/* ------------------------------------------------------------------ */
/* host task + duty-cycle supervisor                                   */
/* ------------------------------------------------------------------ */

static void on_sync(void)
{
    uint8_t addr[6];
    ble_hs_id_copy_addr(BLE_OWN_ADDR_PUBLIC, addr, NULL);
    ESP_LOGI(TAG, "ble addr %02x:%02x:%02x:%02x:%02x:%02x",
             addr[5], addr[4], addr[3], addr[2], addr[1], addr[0]);
    ble_start_advertising();
}

static void on_reset(int reason)
{
    ESP_LOGW(TAG, "host reset, reason=%d", reason);
}

static void ble_host_task(void *param)
{
    nimble_port_run();
    nimble_port_freertos_deinit();
}

/* scan on/off supervisor — owns the duty cycle:
 *   [off: PERIOD-ON] cancel -> [on: ON] ... repeat
 * ble_gap_disc() is left running during the ON leg; the next loop
 * iteration cancels it. Starts are retried harmlessly until host sync. */
static void ble_duty_task(void *arg)
{
    for (;;) {
        ble_gap_disc_cancel();            /* end of previous ON leg */
        vTaskDelay(pdMS_TO_TICKS(BLE_SCAN_PERIOD_MS - BLE_SCAN_ON_MS));
        ble_start_scan();
        vTaskDelay(pdMS_TO_TICKS(BLE_SCAN_ON_MS));
    }
}

void scan_ble_init(void)
{
    int rc = nimble_port_init();
    if (rc != 0) {
        ESP_LOGE(TAG, "nimble_port_init rc=%d", rc);
        return;
    }
    ble_hs_cfg.sync_cb = on_sync;
    ble_hs_cfg.reset_cb = on_reset;
    /* no GAP/GATT *services* — non-connectable advertising only */
    nimble_port_freertos_init(ble_host_task);
    xTaskCreate(ble_duty_task, "ble_duty", 2048, NULL, 4, NULL);
}
