/* scan_radio.c — Wi-Fi environment metadata harvested in promiscuous
 * mode, plus a slow channel sweep for an all-channel RSSI map.
 *
 * Design:
 *  - The promiscuous callback runs in the Wi-Fi task; it does a cheap
 *    per-source-MAC dedupe then queues one formatted line. No blocking.
 *  - A sweep task anchors on CSI_HOME_CHANNEL and briefly visits each
 *    2.4 GHz channel in turn (SCAN_DWELL_MS every SCAN_CYCLE_MS) so the
 *    promiscuous harvest covers all channels while CSI loss stays ~7%.
 *  - A printer task is the only writer of WIFI_DATA/BLE_DATA/SELF_DATA
 *    lines, keeping them atomic against CSI_DATA bursts.
 */

#include "scan_radio.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_wifi.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

static const char *TAG = "scan_radio";

#define CSI_OUTPUT_MAX_BYTES 1024
#define CSI_OUTPUT_HEADER_MAX 256
typedef struct {
    bool is_csi;
    bool first_word_invalid;
    uint16_t len;
    char header[CSI_OUTPUT_HEADER_MAX];
    int8_t iq[CSI_OUTPUT_MAX_BYTES];
} output_record_t;

static QueueHandle_t s_line_q;
static uint32_t s_output_dropped;
static uint32_t s_csi_enqueued;

static void count_output_drop(void)
{
    __atomic_fetch_add(&s_output_dropped, 1, __ATOMIC_RELAXED);
}

bool scan_csi_log(const char *header, const int8_t *iq, uint16_t len,
                  bool first_word_invalid)
{
    if (!s_line_q || !header || !iq || !len || len > CSI_OUTPUT_MAX_BYTES
            || strlen(header) >= CSI_OUTPUT_HEADER_MAX) {
        count_output_drop();
        return false;
    }
    output_record_t record = {.is_csi = true, .len = len,
                              .first_word_invalid = first_word_invalid};
    strcpy(record.header, header);
    memcpy(record.iq, iq, len);
    if (xQueueSend(s_line_q, &record, 0) != pdTRUE) {
        count_output_drop();
        return false;
    }
    __atomic_fetch_add(&s_csi_enqueued, 1, __ATOMIC_RELAXED);
    return true;
}

/* ------------------------------------------------------------------ */
/* output path                                                         */
/* ------------------------------------------------------------------ */

void scan_logf(const char *fmt, ...)
{
    if (!s_line_q) { count_output_drop(); return; }
    output_record_t record = {0};
    va_list ap;
    va_start(ap, fmt);
    int size = vsnprintf(record.header, sizeof(record.header), fmt, ap);
    va_end(ap);
    if (size < 0 || size >= sizeof(record.header)
            || xQueueSend(s_line_q, &record, 0) != pdTRUE) {
        count_output_drop();
    }
}

static void scan_output_task(void *arg)
{
    static output_record_t record;
    int64_t last_health_us = esp_timer_get_time();
    for (;;) {
        if (xQueueReceive(s_line_q, &record, pdMS_TO_TICKS(1000)) == pdTRUE) {
            if (record.is_csi) {
                ets_printf("%s,%u,%u,\"[%d", record.header, record.len,
                           record.first_word_invalid, record.iq[0]);
                for (int i = 1; i < record.len; ++i) ets_printf(",%d", record.iq[i]);
                ets_printf("]\"\n");
            } else {
                ets_printf("%s\n", record.header);
            }
        }
        int64_t now = esp_timer_get_time();
        if (now - last_health_us >= 5000000) {
            ets_printf("HEALTH_DATA,%llu,%u,%lu,%lu\n",
                       (unsigned long long)(now / 1000),
                       (unsigned)uxQueueMessagesWaiting(s_line_q),
                       (unsigned long)__atomic_load_n(&s_output_dropped, __ATOMIC_RELAXED),
                       (unsigned long)__atomic_load_n(&s_csi_enqueued, __ATOMIC_RELAXED));
            last_health_us = now;
        }
    }
}

void scan_output_init(void)
{
    if (s_line_q) return;
    s_line_q = xQueueCreate(SCAN_QUEUE_LEN, sizeof(output_record_t));
    configASSERT(s_line_q);
    BaseType_t started = xTaskCreate(scan_output_task, "scan_out", 3072, NULL, 5, NULL);
    configASSERT(started == pdPASS);
}

void scan_emit_self(void)
{
    uint8_t mac[6] = {0};
    esp_read_mac(mac, ESP_MAC_BT);        /* factory base MAC — stable ID
                                           * (STA MAC is spoofed to the
                                           * shared 1a:00:... for ESP-NOW) */
    scan_logf("SELF_DATA,%s,%02x:%02x:%02x:%02x:%02x:%02x,\"%s\",\"%s\"",
              DEVICE_ROLE, mac[0], mac[1], mac[2], mac[3], mac[4], mac[5],
              DEVICE_NAME, OWNER_ID);
}

/* ------------------------------------------------------------------ */
/* per-source dedupe                                                   */
/* ------------------------------------------------------------------ */

#define DEDUP_SLOTS 64
static struct { uint8_t mac[6]; int64_t last_us; } s_seen[DEDUP_SLOTS];

static bool seen_recently(const uint8_t *mac, uint32_t min_ms)
{
    int64_t now = esp_timer_get_time();
    int slot = mac[5] % DEDUP_SLOTS;
    /* small linear probe so collisions don't starve a MAC */
    for (int i = 0; i < 4; i++) {
        int s = (slot + i) % DEDUP_SLOTS;
        if (memcmp(s_seen[s].mac, mac, 6) == 0) {
            if (now - s_seen[s].last_us < (int64_t)min_ms * 1000)
                return true;
            s_seen[s].last_us = now;
            return false;
        }
        if (s_seen[s].last_us == 0) {
            memcpy(s_seen[s].mac, mac, 6);
            s_seen[s].last_us = now;
            return false;
        }
    }
    /* evict oldest-ish slot */
    memcpy(s_seen[slot].mac, mac, 6);
    s_seen[slot].last_us = now;
    return false;
}

/* ------------------------------------------------------------------ */
/* 802.11 parsing                                                      */
/* ------------------------------------------------------------------ */

static int find_ssid(const uint8_t *ies, int len, char *out, int out_sz)
{
    /* returns ssid length or -1; out is NUL-terminated, non-printable
     * bytes replaced with '.' */
    int pos = 0;
    while (pos + 2 <= len) {
        uint8_t id = ies[pos], ilen = ies[pos + 1];
        if (pos + 2 + ilen > len) break;
        if (id == 0) {                    /* SSID */
            int n = ilen < out_sz - 1 ? ilen : out_sz - 1;
            for (int i = 0; i < n; i++) {
                uint8_t c = ies[pos + 2 + i];
                /* keep the CSV quoted field clean: no '"' or ',' */
                out[i] = (c >= 32 && c < 127 && c != '"' && c != ',')
                         ? (char)c : '.';
            }
            out[n] = 0;
            return n;
        }
        pos += 2 + ilen;
    }
    return -1;
}

static const char *mgmt_kind(uint8_t subtype)
{
    switch (subtype) {
    case 4:  return "prb_req";
    case 5:  return "prb_rsp";
    case 8:  return "bcn";
    case 10: return "disas";
    case 11: return "auth";
    case 12: return "deauth";
    case 0:  return "asoc";
    case 2:  return "reasoc";
    default: return "mgmt";
    }
}

static void wifi_promiscuous_cb(void *buf, wifi_promiscuous_pkt_type_t type)
{
    const wifi_promiscuous_pkt_t *pkt = (const wifi_promiscuous_pkt_t *)buf;
    const uint8_t *f = pkt->payload;
    int flen = pkt->rx_ctrl.sig_len;
    if (flen < 24) return;

    uint16_t fc = f[0] | ((uint16_t)f[1] << 8);
    uint8_t ftype = (fc >> 2) & 0x3;
    uint8_t fsub  = (fc >> 4) & 0xF;
    const uint8_t *sa = f + 10;           /* addr2 = transmitter/SA for
                                           * the frames we care about */

    /* skip our own CSI traffic — already reported via CSI_DATA */
    static const uint8_t csi_mac[6] = {0x1a,0,0,0,0,0};
    if (!memcmp(sa, csi_mac, 6)) return;

    if (ftype == 0) {                     /* management */
        /* beacons are the noisiest class; dedupe harder */
        uint32_t min_ms = (fsub == 8) ? (SCAN_EMIT_MIN_MS * 4)
                                    : SCAN_EMIT_MIN_MS;
        if (seen_recently(sa, min_ms)) return;

        char ssid[33] = "-";
        if (fsub == 8 || fsub == 5)       /* beacon / probe-resp */
            find_ssid(f + 36, flen - 36, ssid, sizeof(ssid));
        else if (fsub == 4)               /* probe-req */
            find_ssid(f + 24, flen - 24, ssid, sizeof(ssid));

        scan_logf("WIFI_DATA,%lu,%02x:%02x:%02x:%02x:%02x:%02x,%s,%d,%d,\"%s\"",
                  (unsigned long)(esp_timer_get_time() / 1000),
                  sa[0], sa[1], sa[2], sa[3], sa[4], sa[5],
                  mgmt_kind(fsub), pkt->rx_ctrl.rssi,
                  pkt->rx_ctrl.channel, ssid);
    } else if (ftype == 2) {              /* data — presence signal */
        if (seen_recently(sa, SCAN_EMIT_MIN_MS * 2)) return;
        scan_logf("WIFI_DATA,%lu,%02x:%02x:%02x:%02x:%02x:%02x,data,%d,%d,\"-\"",
                  (unsigned long)(esp_timer_get_time() / 1000),
                  sa[0], sa[1], sa[2], sa[3], sa[4], sa[5],
                  pkt->rx_ctrl.rssi, pkt->rx_ctrl.channel);
    }
    /* ctrl frames: not useful for presence, ignored */
}

/* ------------------------------------------------------------------ */
/* channel sweep supervisor                                            */
/* ------------------------------------------------------------------ */

#if WIFI_SWEEP_ENABLE
static const uint8_t s_channels[] = {1,2,3,4,5,6,7,8,9,10,11,12,13};

static void sweep_task(void *arg)
{
    int idx = 0;
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(SCAN_CYCLE_MS));
        uint8_t ch = s_channels[idx];
        idx = (idx + 1) % (int)(sizeof(s_channels));
        if (ch == CSI_HOME_CHANNEL) continue;   /* already home */
        esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
        vTaskDelay(pdMS_TO_TICKS(SCAN_DWELL_MS));
        esp_wifi_set_channel(CSI_HOME_CHANNEL, WIFI_SECOND_CHAN_NONE);
    }
}
#endif

void scan_wifi_init(void)
{
    /* promiscuous mode is already enabled by wifi_csi_init(); we only
     * attach the frame callback (all frame types; filter keeps mgmt
     * and data). */
    wifi_promiscuous_filter_t filt = {
        .filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT |
                       WIFI_PROMIS_FILTER_MASK_DATA
    };
    esp_wifi_set_promiscuous_filter(&filt);
    esp_wifi_set_promiscuous_rx_cb(wifi_promiscuous_cb);

#if WIFI_SWEEP_ENABLE
    xTaskCreate(sweep_task, "ch_sweep", 2048, NULL, 4, NULL);
#endif
}
