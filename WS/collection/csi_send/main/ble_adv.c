/* ble_adv.c — NimBLE non-connectable advertiser for csi_send. */

#include "ble_adv.h"

#include <string.h>

#include "esp_log.h"

#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/ble_hs_adv.h"

static const char *TAG = "ble_adv";

static void start_advertising(void)
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

static void on_sync(void)
{
    start_advertising();
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

void ble_adv_init(void)
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
}
