/* ble_adv.h — identity BLE beacon for csi_send.
 *
 * Advertises the CSI transmitter as "thoth-csi-tx" (non-connectable,
 * ~1 s interval) with a manufacturer payload (company 0xFFFF + OWNER_ID)
 * so scanning devices can attach the radio to a person/role. Costs a
 * ~1 ms airtime burst per advert — negligible vs the 1 kHz ESP-NOW duty.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DEVICE_NAME
#define DEVICE_NAME  "thoth-csi-tx"
#endif
#ifndef OWNER_ID
#define OWNER_ID     "gad21"
#endif

void ble_adv_init(void);

#ifdef __cplusplus
}
#endif
