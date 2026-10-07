#pragma once
// Copy to clap_local.h and enter your own values. The real file is ignored.
#define CLAP_WIFI_SSID "YOUR_WIFI_SSID"
#define CLAP_WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#define CLAP_TUYA_DEVICE_ID "YOUR_TUYA_DEVICE_ID"
// A Tuya 3.5 local key must contain exactly 16 characters.
#define CLAP_TUYA_LOCAL_KEY "YOUR_TUYA_KEY_16"
// Optional saved-address hint. Zero selects local discovery until an address
// has been found; the successful bulb address is persisted in device NVS.
#define CLAP_TUYA_IP 0, 0, 0, 0
// Empty retains an existing valid Lab password or generates one at boot.
#define CLAP_LAB_PASSWORD ""
