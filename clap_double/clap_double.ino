#include <WiFi.h>
#include <WiFiUdp.h>
#include <Preferences.h>
#include <ESP_I2S.h>
#include "mbedtls/gcm.h"
#include "mbedtls/md.h"
#include "mbedtls/md5.h"
#include "esp_random.h"
#include "esp_task_wdt.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include <errno.h>
#include <stdarg.h>
#include <atomic>
#include "clap_config.h"
#include "clap_detector.h"

#if !defined(CLAP_WIFI_SSID) || !defined(CLAP_WIFI_PASSWORD) || !defined(CLAP_TUYA_DEVICE_ID) || !defined(CLAP_TUYA_LOCAL_KEY) || !defined(CLAP_TUYA_IP)
#error "Copy clap_local.example.h to ignored clap_local.h and configure your own hardware credentials."
#endif
static_assert(sizeof(CLAP_TUYA_LOCAL_KEY) == 17, "Tuya local key must contain exactly 16 characters");

const char *WIFI_SSID = CLAP_WIFI_SSID;
const char *WIFI_PASS = CLAP_WIFI_PASSWORD;

const char *TUYA_ID = CLAP_TUYA_DEVICE_ID;
const char *TUYA_KEY = CLAP_TUYA_LOCAL_KEY;
const uint16_t TUYA_PORT = 6668;
const int WARM_TEMP = 500;

const uint8_t CMD_DOUBLE = 1;
const uint8_t CMD_LAMP = 2;
const unsigned long WIFI_DEAD_MS = 180000;
const unsigned long I2S_STALL_MS = 2000;
const unsigned long HB_MS = 60000;
const unsigned long LED_MS = 200;
const unsigned long SCAN_GAP_MS = 180;
const unsigned long TUYA_CONNECT_MS = 1500;
const unsigned long TUYA_HB_RX_MS = 800;
const unsigned long TUYA_WRITE_MS = 400;
const unsigned long TUYA_REFRESH_MS = 10800000UL;
const unsigned long LOCK_MS = 200;

#define LOG_MAX 120
#define EV_DULL 1
#define EV_HIT 2
#define EV_IGNORE 3
#define EV_BUSY 4
#define EV_SINGLE 5
#define EV_DOUBLE 6
#define EV_SLAM 7

bool labSetup();
void labOnWifi();
void pcmTeeBlock(const int32_t *samples, unsigned count, uint64_t firstSample);
void labEvent(uint8_t kind, uint32_t dur, int32_t hp);
void labEventAt(uint8_t kind, uint32_t dur, int32_t hp, uint64_t sample, uint64_t onsetSample);
void labRequestLamp(bool on, int bri, int temp);
bool aiSetup();
void aiBegin(uint32_t onsetSample);
void aiSample(int32_t raw24, uint32_t sample);
void aiCandidate(const ClapCandidate &candidate);
void aiPoll();
void aiReset();
bool aiPendingOnset(uint32_t &sample);
const char *aiMode();
uint32_t aiDropped();
uint32_t aiErrors();
uint32_t aiLastInferenceMs();
void labAiEvent(const ClapCandidate &candidate, const float *scores, uint32_t frontendMs,
                uint32_t inferenceMs, bool accepted, const char *mode);
void rejectCandidate(const ClapCandidate &candidate);

I2SClass i2s;
Preferences prefs;
WiFiUDP udp6667;
WiFiUDP udp7000;
WiFiClient tuyaSock;
struct LampCommand {
  uint8_t kind;
  bool on;
  int16_t brightness;
  int16_t temperature;
};
QueueHandle_t clapQ = NULL;
QueueHandle_t logQ = NULL;

IPAddress tuyaIp(CLAP_TUYA_IP);
uint8_t udpKey[16];
uint8_t sessionKey[16];
uint32_t tuyaSeq = 1;
bool tuyaLive = false;
bool needHunt = false;
bool udpUp = false;
bool lampOn = false;
bool lampKnown = false;
bool lampNvsOn = false;
bool lampNvsKnown = false;
bool lampNvsInit = false;
int lampBri = 1000;
int lampTemp = WARM_TEMP;
int lampNvsBri = 1000;
int lampNvsTemp = WARM_TEMP;
bool lampHasColour = false;
uint32_t nvsIp = 0;
int scanHost = 2;
std::atomic<int32_t> armLevel{200000};
int32_t floorHpStored = 1000;
volatile int armMul = 8;
unsigned long lastDiscover = 0;
unsigned long lastHbOk = 0;
unsigned long lastHbTry = 0;
unsigned long lastDpsMs = 0;
unsigned long lastTuyaRefresh = 0;
unsigned long lastScanHost = 0;
unsigned long lastReconnectTry = 0;
unsigned long reconnectDelayMs = 1000;
unsigned long wifiDownSince = 0;
unsigned long ledAt = 0;
int hbMiss = 0;
bool ledLit = false;

void applyArm() {
  int32_t mul = armMul;
  if (mul < 2) mul = 2;
  if (mul > 24) mul = 24;
  armMul = mul;
  armLevel = floorHpStored * mul;
  if (armLevel < 150000) armLevel = 150000;
}

void elog(const char *fmt, ...) {
  char buf[LOG_MAX];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (logQ) xQueueSend(logQ, buf, 0);
  else Serial.println(buf);
}

void drainLog() {
  if (!logQ) return;
  char buf[LOG_MAX];
  int n = 0;
  while (n++ < 12 && xQueueReceive(logQ, buf, 0) == pdTRUE) {
    Serial.println(buf);
  }
}

volatile bool wifiGotIp = false;
volatile bool wifiLost = false;
volatile bool i2sOk = false;
volatile bool tuyaBusy = false;
volatile unsigned long lastI2sMs = 0;

static uint8_t tuyaRx[800];
static size_t tuyaRxN = 0;
static uint8_t tuyaDec[700];
static uint8_t tuyaPlain[700];
static uint8_t tuyaCipher[512];
static uint8_t tuyaPkt[512];
static uint8_t tuyaPayload[400];
static uint8_t udpBuf[800];
static uint8_t udpDec[700];
static char udpJs[700];
static int32_t i2sBuf[128];
static int32_t i2sPrev = 0;
static uint64_t audioSamples = 0;
static ClapDetector clapDetector;
static ClapPairer clapPairer;
static std::atomic<uint32_t> commandDrops{0};
static bool slamLocked = false;
static uint32_t slamSample = 0;
static volatile uint32_t i2sDroppedSamples = 0;
static uint32_t i2sDropsSeen = 0;

bool IRAM_ATTR audioOverflow(i2s_chan_handle_t, i2s_event_data_t *event, void *) {
  // ISR: only count loss. Do not log, allocate, or perform DSP here.
  i2sDroppedSamples += event->size / sizeof(int32_t);
  return false;
}

void wdtFeed() {
  esp_task_wdt_reset();
}

void bootFailure(const char *reason) {
  Serial.print("Boot failed; retrying after restart: ");
  Serial.println(reason);
  Serial.flush();
  delay(500);
  ESP.restart();
  for (;;) delay(1000); // Watchdog is a fallback if restart unexpectedly returns.
}

void wdtAddThis(const char *tag) {
  Serial.print(tag);
  if (esp_task_wdt_add(NULL) == ESP_OK) Serial.println(" wdt ok");
  else Serial.println(" wdt add fail");
}

void wdtSetup() {
  esp_task_wdt_config_t cfg = {
    .timeout_ms = 12000,
    .idle_core_mask = 0,
    .trigger_panic = true,
  };
  // Arduino-ESP32 3.x already started TWDT before setup().
  // Calling init() again prints E (19) TWDT already initialized.
  esp_err_t e = esp_task_wdt_reconfigure(&cfg);
  if (e == ESP_ERR_INVALID_STATE) e = esp_task_wdt_init(&cfg);
  if (e != ESP_OK && e != ESP_ERR_INVALID_STATE) {
    Serial.print("wdt init err ");
    Serial.println((int)e);
  }
  wdtAddThis("loop");
}

const char *resetReasonStr() {
  switch (esp_reset_reason()) {
    case ESP_RST_POWERON: return "power";
    case ESP_RST_SW: return "sw";
    case ESP_RST_PANIC: return "panic";
    case ESP_RST_INT_WDT: return "intwdt";
    case ESP_RST_TASK_WDT: return "taskwdt";
    case ESP_RST_WDT: return "wdt";
    case ESP_RST_BROWNOUT: return "brownout";
    case ESP_RST_SDIO: return "sdio";
    default: return "other";
  }
}

int32_t abs32(int32_t v) {
  if (v < 0) {
    if (v == -2147483648) return 2147483647;
    return -v;
  }
  return v;
}

void saveLamp() {
  bool same = lampNvsInit && lampNvsOn == lampOn && lampNvsKnown == lampKnown &&
              lampNvsBri == lampBri && lampNvsTemp == lampTemp;
  if (same) return;
  prefs.putBool("on", lampOn);
  prefs.putBool("known", lampKnown);
  prefs.putInt("bri", lampBri);
  prefs.putInt("temp", lampTemp);
  lampNvsOn = lampOn;
  lampNvsKnown = lampKnown;
  lampNvsBri = lampBri;
  lampNvsTemp = lampTemp;
  lampNvsInit = true;
}

void saveIp(IPAddress ip) {
  uint32_t v = (uint32_t)ip;
  tuyaIp = ip;
  if (nvsIp == v) return;
  prefs.putUInt("ip", v);
  nvsIp = v;
}

bool hmacSha256(const uint8_t *key, size_t keyLen, const uint8_t *msg, size_t msgLen, uint8_t out[32]) {
  const mbedtls_md_info_t *info = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
  return mbedtls_md_hmac(info, key, keyLen, msg, msgLen, out) == 0;
}

bool gcmEncrypt(const uint8_t *key, const uint8_t *iv, const uint8_t *aad, size_t aadLen,
                const uint8_t *plain, size_t plainLen, uint8_t *cipher, uint8_t tag[16]) {
  mbedtls_gcm_context gcm;
  mbedtls_gcm_init(&gcm);
  int rc = mbedtls_gcm_setkey(&gcm, MBEDTLS_CIPHER_ID_AES, key, 128);
  if (rc == 0) {
    rc = mbedtls_gcm_crypt_and_tag(&gcm, MBEDTLS_GCM_ENCRYPT, plainLen, iv, 12, aad, aadLen, plain, cipher, 16, tag);
  }
  mbedtls_gcm_free(&gcm);
  return rc == 0;
}

bool gcmDecrypt(const uint8_t *key, const uint8_t *iv, const uint8_t *aad, size_t aadLen,
                const uint8_t *cipher, size_t cipherLen, const uint8_t tag[16], uint8_t *plain) {
  mbedtls_gcm_context gcm;
  mbedtls_gcm_init(&gcm);
  int rc = mbedtls_gcm_setkey(&gcm, MBEDTLS_CIPHER_ID_AES, key, 128);
  if (rc == 0) {
    rc = mbedtls_gcm_auth_decrypt(&gcm, cipherLen, iv, 12, aad, aadLen, tag, 16, cipher, plain);
  }
  mbedtls_gcm_free(&gcm);
  return rc == 0;
}

void putU16(uint8_t *p, uint16_t v) {
  p[0] = (v >> 8) & 0xFF;
  p[1] = v & 0xFF;
}

void putU32(uint8_t *p, uint32_t v) {
  p[0] = (v >> 24) & 0xFF;
  p[1] = (v >> 16) & 0xFF;
  p[2] = (v >> 8) & 0xFF;
  p[3] = v & 0xFF;
}

uint32_t getU32(const uint8_t *p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

bool tuyaPack(const uint8_t *key, uint32_t seq, uint32_t cmd, const uint8_t *payload, size_t payloadLen,
              uint8_t *out, size_t *outLen) {
  uint8_t hdr[18];
  uint32_t length = (uint32_t)(payloadLen + 28);
  putU32(hdr, 0x00006699);
  putU16(hdr + 4, 0);
  putU32(hdr + 6, seq);
  putU32(hdr + 10, cmd);
  putU32(hdr + 14, length);

  uint8_t iv[12];
  uint8_t tag[16];
  esp_fill_random(iv, 12);

  if (payloadLen > sizeof(tuyaCipher)) return false;
  if (payloadLen + 50 > sizeof(tuyaPkt)) return false;
  if (!gcmEncrypt(key, iv, hdr + 4, 14, payload, payloadLen, tuyaCipher, tag)) return false;

  size_t n = 0;
  memcpy(out + n, hdr, 18); n += 18;
  memcpy(out + n, iv, 12); n += 12;
  memcpy(out + n, tuyaCipher, payloadLen); n += payloadLen;
  memcpy(out + n, tag, 16); n += 16;
  putU32(out + n, 0x00009966); n += 4;
  *outLen = n;
  return true;
}

void tuyaRxClear() {
  tuyaRxN = 0;
}

void tuyaDrop() {
  tuyaLive = false;
  tuyaRxClear();
  tuyaSock.stop();
}

bool tuyaSockAlive() {
  int s = tuyaSock.fd();
  if (s < 0) return false;
  uint8_t dummy;
  int res = recv(s, &dummy, 1, MSG_DONTWAIT | MSG_PEEK);
  if (res == 0) return false;
  if (res < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINPROGRESS) return true;
    return false;
  }
  return true;
}

void tuyaKeepalive() {
  int s = tuyaSock.fd();
  if (s < 0) return;
  int yes = 1;
  setsockopt(s, SOL_SOCKET, SO_KEEPALIVE, &yes, sizeof(yes));
  int idle = 10, intvl = 3, cnt = 3;
  setsockopt(s, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
  setsockopt(s, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
  setsockopt(s, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(cnt));
}

bool tuyaWriteAll(const uint8_t *data, size_t len) {
  int s = tuyaSock.fd();
  if (s < 0) {
    tuyaDrop();
    return false;
  }
  unsigned long start = millis();
  size_t off = 0;
  while (off < len) {
    wdtFeed();
    if (millis() - start > TUYA_WRITE_MS) {
      tuyaDrop();
      return false;
    }
    int n = send(s, data + off, len - off, MSG_DONTWAIT);
    if (n > 0) off += (size_t)n;
    else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
      tuyaDrop();
      return false;
    } else {
      vTaskDelay(1);
    }
  }
  return true;
}

void tuyaRxPull() {
  int s = tuyaSock.fd();
  if (s < 0) return;
  while (tuyaRxN < sizeof(tuyaRx)) {
    int n = recv(s, tuyaRx + tuyaRxN, sizeof(tuyaRx) - tuyaRxN, MSG_DONTWAIT);
    if (n > 0) tuyaRxN += (size_t)n;
    else break;
  }
}

void tuyaApplyDps(uint8_t *plain, size_t n) {
  if (n > 698) n = 698;
  plain[n] = 0;
  char *js = (char *)plain;
  char *p = strstr(js, "\"20\":");
  if (p) {
    p += 5;
    while (*p == ' ') p++;
    if (strncmp(p, "true", 4) == 0) {
      lampOn = true;
      lampKnown = true;
      lastDpsMs = millis();
    } else if (strncmp(p, "false", 5) == 0) {
      lampOn = false;
      lampKnown = true;
      lastDpsMs = millis();
    }
  }
  p = strstr(js, "\"22\":");
  if (p) {
    int v = atoi(p + 5);
    if (v < 10) v = 10;
    if (v > 1000) v = 1000;
    lampBri = v;
  }
  p = strstr(js, "\"23\":");
  if (p) {
    int v = atoi(p + 5);
    if (v < 0) v = 0;
    if (v > 1000) v = 1000;
    lampTemp = v;
  }
  if (strstr(js, "\"24\":")) lampHasColour = true;
}

// 1 = frame, 0 = need more / silence, -1 = bad (caller must drop)
int tuyaParseFrame(const uint8_t *key, uint32_t *cmd, uint8_t *plain, size_t *plainLen, size_t plainMax) {
  if (tuyaRxN < 18) return 0;
  if (getU32(tuyaRx) != 0x00006699) return -1;
  uint32_t length = getU32(tuyaRx + 14);
  if (length < 28 || length > 700) return -1;
  size_t total = 18 + (size_t)length + 4;
  if (total > sizeof(tuyaRx)) return -1;
  if (tuyaRxN < total) return 0;
  if (getU32(tuyaRx + 18 + length) != 0x00009966) return -1;

  *cmd = getU32(tuyaRx + 10);
  const uint8_t *rest = tuyaRx + 18;
  const uint8_t *iv = rest;
  size_t cipherLen = length - 28;
  const uint8_t *cipher = rest + 12;
  const uint8_t *tag = rest + 12 + cipherLen;
  if (cipherLen > sizeof(tuyaDec)) return -1;
  if (!gcmDecrypt(key, iv, tuyaRx + 4, 14, cipher, cipherLen, tag, tuyaDec)) return -1;

  size_t off = 0;
  if (cipherLen >= 5 && tuyaDec[0] == 0 && tuyaDec[1] == 0 && tuyaDec[2] == 0 && tuyaDec[3] == 0) off = 4;
  size_t n = cipherLen - off;
  if (n > plainMax) n = plainMax;
  memcpy(plain, tuyaDec + off, n);
  *plainLen = n;

  size_t remain = tuyaRxN - total;
  if (remain) memmove(tuyaRx, tuyaRx + total, remain);
  tuyaRxN = remain;
  return 1;
}

int tuyaReadMsg(const uint8_t *key, uint32_t *cmd, uint8_t *plain, size_t *plainLen, size_t plainMax, unsigned long timeoutMs) {
  unsigned long start = millis();
  for (;;) {
    wdtFeed();
    tuyaRxPull();
    int r = tuyaParseFrame(key, cmd, plain, plainLen, plainMax);
    if (r != 0) return r;
    if (millis() - start > timeoutMs) return (tuyaRxN == 0) ? 0 : -1;
    if (!tuyaSockAlive()) return -1;
    vTaskDelay(1);
  }
}

int tuyaEatFrames(unsigned long budgetMs) {
  unsigned long start = millis();
  int got = 0;
  while (millis() - start < budgetMs) {
    wdtFeed();
    unsigned long left = budgetMs - (millis() - start);
    if (left < 1) left = 1;
    uint32_t cmd = 0;
    size_t n = 0;
    int rx = tuyaReadMsg(sessionKey, &cmd, tuyaPlain, &n, sizeof(tuyaPlain) - 1, left);
    if (rx == 1) {
      tuyaApplyDps(tuyaPlain, n);
      got++;
      continue;
    }
    if (rx < 0) {
      tuyaDrop();
      return -1;
    }
    break;
  }
  return got;
}

bool tuyaHandshake() {
  uint8_t realKey[16];
  memcpy(realKey, TUYA_KEY, 16);

  uint8_t localNonce[16];
  esp_fill_random(localNonce, 16);

  size_t pktLen = 0;
  tuyaSeq = 1;
  if (!tuyaPack(realKey, tuyaSeq++, 3, localNonce, 16, tuyaPkt, &pktLen)) return false;
  if (!tuyaWriteAll(tuyaPkt, pktLen)) return false;

  uint32_t cmd = 0;
  size_t plainLen = 0;
  if (tuyaReadMsg(realKey, &cmd, tuyaPlain, &plainLen, sizeof(tuyaPlain) - 1, 800) != 1) return false;
  if (cmd != 4 || plainLen < 48) return false;

  uint8_t expect[32];
  hmacSha256(realKey, 16, localNonce, 16, expect);
  if (memcmp(expect, tuyaPlain + 16, 32) != 0) return false;

  uint8_t remoteNonce[16];
  memcpy(remoteNonce, tuyaPlain, 16);

  uint8_t finish[32];
  hmacSha256(realKey, 16, remoteNonce, 16, finish);
  if (!tuyaPack(realKey, tuyaSeq++, 5, finish, 32, tuyaPkt, &pktLen)) return false;
  if (!tuyaWriteAll(tuyaPkt, pktLen)) return false;
  vTaskDelay(pdMS_TO_TICKS(30));
  tuyaRxClear();
  tuyaSock.clear();

  uint8_t xored[16];
  for (int i = 0; i < 16; i++) xored[i] = localNonce[i] ^ remoteNonce[i];

  uint8_t cipher[16];
  uint8_t tag[16];
  if (!gcmEncrypt(realKey, localNonce, nullptr, 0, xored, 16, cipher, tag)) return false;
  memcpy(sessionKey, cipher, 16);
  return true;
}

bool tuyaSendRaw(uint32_t cmd, const uint8_t *payload, size_t n) {
  size_t pktLen = 0;
  if (!tuyaPack(sessionKey, tuyaSeq++, cmd, payload, n, tuyaPkt, &pktLen)) return false;
  if (!tuyaWriteAll(tuyaPkt, pktLen)) return false;
  return true;
}

bool tuyaSendJson(uint32_t cmd, const char *json, bool versionHdr) {
  size_t n = 0;
  if (versionHdr) {
    memcpy(tuyaPayload, "3.5", 3);
    memset(tuyaPayload + 3, 0, 12);
    n = 15;
  }
  size_t jl = strlen(json);
  if (n + jl > sizeof(tuyaPayload)) return false;
  memcpy(tuyaPayload + n, json, jl);
  n += jl;
  return tuyaSendRaw(cmd, tuyaPayload, n);
}

bool tuyaConnectSavedIp() {
  tuyaDrop();
  tuyaSock.setTimeout(TUYA_CONNECT_MS);
  tuyaSock.setNoDelay(true);
  wdtFeed();
  if (!tuyaSock.connect(tuyaIp, TUYA_PORT, TUYA_CONNECT_MS)) return false;
  wdtFeed();
  tuyaKeepalive();
  if (!tuyaHandshake()) {
    tuyaDrop();
    return false;
  }
  tuyaLive = true;
  hbMiss = 0;
  lastHbOk = millis();
  lastHbTry = lastHbOk;
  saveIp(tuyaIp);
  return true;
}

bool tuyaEnsure() {
  if (tuyaLive && tuyaSockAlive()) {
    if (tuyaEatFrames(20) < 0) return tuyaConnectSavedIp();
    return tuyaLive;
  }
  tuyaDrop();
  return tuyaConnectSavedIp();
}

bool tuyaHeartbeat() {
  if (!tuyaLive || !tuyaSockAlive()) {
    tuyaDrop();
    return false;
  }
  if (tuyaEatFrames(20) < 0) return false;
  if (!tuyaSendRaw(9, (const uint8_t *)"", 0)) return false;
  uint32_t cmd = 0;
  size_t n = 0;
  int rx = tuyaReadMsg(sessionKey, &cmd, tuyaPlain, &n, sizeof(tuyaPlain) - 1, TUYA_HB_RX_MS);
  if (rx == 1) {
    tuyaApplyDps(tuyaPlain, n);
    hbMiss = 0;
    lastHbOk = millis();
    return true;
  }
  if (rx < 0) {
    tuyaDrop();
    return false;
  }
  hbMiss++;
  if (hbMiss >= 2) {
    tuyaDrop();
    return false;
  }
  return true;
}

bool tuyaCmd(const char *json) {
  for (int attempt = 0; attempt < 2; attempt++) {
    wdtFeed();
    if (!tuyaEnsure()) continue;
    if (tuyaEatFrames(8) < 0) continue;
    if (!tuyaSendJson(0x0d, json, true)) continue;
    if (tuyaSockAlive()) return true;
    tuyaDrop();
  }
  return false;
}

int tuyaQueryOn() {
  if (!tuyaEnsure()) return -1;
  if (tuyaEatFrames(20) < 0) return -1;
  if (!tuyaSendJson(0x10, "{}", false)) return -1;
  uint32_t cmd = 0;
  size_t n = 0;
  if (tuyaReadMsg(sessionKey, &cmd, tuyaPlain, &n, sizeof(tuyaPlain) - 1, 800) != 1) {
    tuyaDrop();
    return -1;
  }
  tuyaApplyDps(tuyaPlain, n);
  tuyaPlain[n < 699 ? n : 699] = 0;
  char *p = strstr((char *)tuyaPlain, "\"20\":");
  if (!p) {
    tuyaDrop();
    return -1;
  }
  p += 5;
  while (*p == ' ') p++;
  if (strncmp(p, "true", 4) == 0) return 1;
  if (strncmp(p, "false", 5) == 0) return 0;
  tuyaDrop();
  return -1;
}

bool jsonField(const char *js, const char *key, char *out, size_t outMax) {
  char pat[24];
  snprintf(pat, sizeof(pat), "\"%s\":\"", key);
  const char *p = strstr(js, pat);
  if (!p) return false;
  p += strlen(pat);
  size_t n = 0;
  while (*p && *p != '"' && n + 1 < outMax) out[n++] = *p++;
  out[n] = 0;
  return n > 0;
}

bool adoptJson(const char *js) {
  char id[40] = {0};
  char ipStr[20] = {0};
  if (!jsonField(js, "gwId", id, sizeof(id))) jsonField(js, "id", id, sizeof(id));
  if (!id[0] || strcmp(id, TUYA_ID) != 0) return false;
  if (!jsonField(js, "ip", ipStr, sizeof(ipStr))) return false;
  IPAddress ip;
  if (!ip.fromString(ipStr)) return false;
  if (ip != tuyaIp) {
    Serial.print("bulb ip ");
    Serial.println(ip);
    tuyaDrop();
  }
  saveIp(ip);
  needHunt = false;
  return true;
}

bool decryptUdp(const uint8_t *data, int len, char *out, size_t outMax) {
  if (len < 20) return false;
  if (getU32(data) != 0x00006699) return false;
  uint32_t length = getU32(data + 14);
  if (length < 28 || (int)(18 + length + 4) > len) return false;
  const uint8_t *iv = data + 18;
  size_t cipherLen = length - 28;
  const uint8_t *cipher = data + 30;
  const uint8_t *tag = data + 30 + cipherLen;
  if (cipherLen >= outMax || cipherLen > sizeof(udpDec)) return false;
  if (!gcmDecrypt(udpKey, iv, data + 4, 14, cipher, cipherLen, tag, udpDec)) return false;
  size_t off = 0;
  if (cipherLen >= 5 && udpDec[0] == 0 && udpDec[1] == 0 && udpDec[2] == 0 && udpDec[3] == 0) off = 4;
  size_t n = cipherLen - off;
  if (n >= outMax) n = outMax - 1;
  memcpy(out, udpDec + off, n);
  out[n] = 0;
  return true;
}

bool pollUdp() {
  bool found = false;
  for (int sock = 0; sock < 2; sock++) {
    WiFiUDP &u = (sock == 0) ? udp6667 : udp7000;
    int n = u.parsePacket();
    if (n <= 0) continue;
    if (n > (int)sizeof(udpBuf)) n = sizeof(udpBuf);
    int got = u.read(udpBuf, n);
    if (!decryptUdp(udpBuf, got, udpJs, sizeof(udpJs))) continue;
    if (adoptJson(udpJs)) found = true;
  }
  return found;
}

void tuyaBroadcast() {
  size_t pktLen = 0;
  if (!tuyaPack(udpKey, 1, 0x25, (const uint8_t *)"", 0, tuyaPkt, &pktLen)) return;
  IPAddress bcast = WiFi.localIP();
  bcast[3] = 255;
  udp7000.beginPacket(bcast, 7000);
  udp7000.write(tuyaPkt, pktLen);
  udp7000.endPacket();
  udp7000.beginPacket(IPAddress(255, 255, 255, 255), 7000);
  udp7000.write(tuyaPkt, pktLen);
  udp7000.endPacket();
}

bool portOpen(IPAddress ip) {
  WiFiClient c;
  bool ok = c.connect(ip, TUYA_PORT, 80);
  c.stop();
  return ok;
}

bool findBulb(unsigned long budgetMs) {
  unsigned long t0 = millis();
  Serial.println("finding bulb...");
  tuyaBroadcast();
  while (millis() - t0 < 1200 && millis() - t0 < budgetMs) {
    wdtFeed();
    if (pollUdp()) {
      Serial.print("found via udp ");
      Serial.println(tuyaIp);
      return true;
    }
    vTaskDelay(pdMS_TO_TICKS(10));
  }

  IPAddress me = WiFi.localIP();
  int start = tuyaIp[3];
  if (start < 2 || start > 254) start = 50;

  for (int round = 0; round < 2; round++) {
    int from = (round == 0) ? (start - 20) : 2;
    int to = (round == 0) ? (start + 20) : 254;
    if (from < 2) from = 2;
    if (to > 254) to = 254;
    for (int host = from; host <= to; host++) {
      wdtFeed();
      if (millis() - t0 > budgetMs) return false;
      if (host == (int)me[3]) continue;
      if (round == 1 && host >= start - 20 && host <= start + 20) continue;
      IPAddress ip(me[0], me[1], me[2], host);
      if (!portOpen(ip)) continue;
      saveIp(ip);
      Serial.print("found via scan ");
      Serial.println(ip);
      return true;
    }
  }
  Serial.println("bulb not found");
  return false;
}

void ensureUdp() {
  if (udpUp) return;
  uint8_t a = udp6667.begin(6667);
  uint8_t b = udp7000.begin(7000);
  udpUp = a && b;
}

void stopUdp() {
  if (!udpUp) return;
  udp6667.stop();
  udp7000.stop();
  udpUp = false;
}

void idleHuntTick() {
  static bool wasHunting = false;
  static unsigned long lastSavedTry = 0;
  static IPAddress lastSavedIp;
  if (tuyaLive && !tuyaSockAlive()) {
    tuyaDrop();
    needHunt = true;
  }
  if (!udpUp) return;
  pollUdp();
  // Discovery can update the saved IP and clear needHunt without connecting.
  // A known address is not a live session; restore the connection automatically.
  if (!tuyaLive) needHunt = true;

  if (millis() - lastDiscover > 20000) {
    lastDiscover = millis();
    tuyaBroadcast();
  }

  if (tuyaLive && millis() - lastHbTry > 15000) {
    lastHbTry = millis();
    if (!tuyaHeartbeat()) {
      Serial.println("tuya hb fail");
      needHunt = true;
    }
  }

  if (tuyaLive) { wasHunting = false; return; }
  if (!needHunt) { wasHunting = false; return; }
  // Try the last verified address first. Rate-limit unsuccessful handshakes so
  // an absent bulb cannot make every scan step wait for the connect timeout.
  if (!wasHunting || tuyaIp != lastSavedIp || millis() - lastSavedTry >= 15000) {
    wasHunting = true;
    lastSavedTry = millis();
    lastSavedIp = tuyaIp;
    if (tuyaConnectSavedIp()) {
      int st = tuyaQueryOn();
      if (st >= 0) {
        lampOn = (st == 1);
        lampKnown = true;
        saveLamp();
      }
      if (tuyaLive) { needHunt = false; wasHunting = false; return; }
    }
    needHunt = true;
  }
  if (millis() - lastScanHost < SCAN_GAP_MS) return;
  lastScanHost = millis();

  IPAddress me = WiFi.localIP();
  if (scanHost < 2 || scanHost > 254) scanHost = 2;
  if (scanHost == (int)me[3]) {
    scanHost++;
    return;
  }
  IPAddress ip(me[0], me[1], me[2], scanHost);
  scanHost++;
  if (scanHost > 254) scanHost = 2;
  if (!portOpen(ip)) return;
  saveIp(ip);
  Serial.print("idle found ");
  Serial.println(ip);
  if (tuyaEnsure()) {
    int st = tuyaQueryOn();
    if (st >= 0) {
      lampOn = (st == 1);
      lampKnown = true;
      saveLamp();
    }
  }
  needHunt = !tuyaLive;
  if (tuyaLive) wasHunting = false;
}

void onDoubleClap() {
  tuyaBusy = true;

  if (!tuyaEnsure()) {
    elog("tuya connect fail");
    needHunt = true;
    tuyaBusy = false;
    return;
  }

  // After a panic, NVS can say OFF while the chandelier is still lit.
  // Only query when we would otherwise send ON.
  if (!lampKnown || !lampOn) {
    int st = tuyaQueryOn();
    if (st >= 0) {
      lampOn = (st == 1);
      lampKnown = true;
    }
  }

  bool turnOn = lampKnown ? !lampOn : true;
  char json[128];
  unsigned long t = millis() / 1000;
  bool wrote = false;

  if (turnOn) {
    elog("lamp ON");
    snprintf(json, sizeof(json),
             "{\"protocol\":5,\"t\":%lu,\"data\":{\"dps\":{\"20\":true,\"21\":\"white\",\"22\":%d,\"23\":%d}}}",
             t, lampBri, lampTemp);
    wrote = tuyaCmd(json);
  } else {
    elog("lamp OFF");
    snprintf(json, sizeof(json),
             "{\"protocol\":5,\"t\":%lu,\"data\":{\"dps\":{\"20\":false}}}", t);
    wrote = tuyaCmd(json);
  }

  if (wrote) {
    lampOn = turnOn;
    lampKnown = true;
    saveLamp();
  } else {
    elog(turnOn ? "tuya ON fail" : "tuya OFF fail");
    needHunt = true;
  }

  tuyaBusy = false;
}

void applyLampSet(const LampCommand &request) {
  tuyaBusy = true;
  int bri = request.brightness;
  int tmp = request.temperature;
  bool turnOn = request.on;
  if (bri < 10) bri = 10;
  if (bri > 1000) bri = 1000;
  if (tmp < 0) tmp = 0;
  if (tmp > 1000) tmp = 1000;
  lampBri = bri;
  lampTemp = tmp;

  if (!tuyaEnsure()) {
    elog("lamp set: tuya down");
    needHunt = true;
    tuyaBusy = false;
    return;
  }

  char json[160];
  unsigned long t = millis() / 1000;
  bool wrote = false;
  if (turnOn) {
    elog("lamp set ON bri=%d temp=%d", bri, tmp);
    snprintf(json, sizeof(json),
             "{\"protocol\":5,\"t\":%lu,\"data\":{\"dps\":{\"20\":true,\"21\":\"white\",\"22\":%d,\"23\":%d}}}",
             t, bri, tmp);
    wrote = tuyaCmd(json);
  } else {
    elog("lamp set OFF");
    snprintf(json, sizeof(json),
             "{\"protocol\":5,\"t\":%lu,\"data\":{\"dps\":{\"20\":false}}}", t);
    wrote = tuyaCmd(json);
  }

  if (wrote) {
    lampOn = turnOn;
    lampKnown = true;
    saveLamp();
  } else {
    elog("lamp set fail");
    needHunt = true;
  }
  tuyaBusy = false;
}

void labRequestLamp(bool on, int bri, int temp) {
  if (bri < 10) bri = 10;
  if (bri > 1000) bri = 1000;
  if (temp < 0) temp = 0;
  if (temp > 1000) temp = 1000;
  const LampCommand command = {CMD_LAMP, on, (int16_t)bri, (int16_t)temp};
  if (!clapQ || xQueueSend(clapQ, &command, 0) != pdTRUE) {
    ++commandDrops;
    elog("lamp command queue full");
  }
}

void onWifiEvent(WiFiEvent_t event) {
  if (event == ARDUINO_EVENT_WIFI_STA_GOT_IP) {
    wifiLost = false;
    wifiGotIp = true;
  } else if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
    wifiLost = true;
  }
}

void handleWifiEvents() {
  if (wifiGotIp) {
    wifiGotIp = false;
    reconnectDelayMs = 1000;
    wifiDownSince = 0;
    ensureUdp();
    if (!tuyaLive || !tuyaSockAlive()) needHunt = true;
    Serial.print("wifi ok ");
    Serial.println(WiFi.localIP());
    labOnWifi();
  }
  if (wifiLost) {
    wifiLost = false;
    if (WiFi.status() == WL_CONNECTED) return;
    tuyaDrop();
    needHunt = true;
    stopUdp();
    if (wifiDownSince == 0) wifiDownSince = millis();
    Serial.println("wifi lost");
  }
}

void wifiBackoffTick() {
  if (WiFi.status() == WL_CONNECTED) {
    wifiDownSince = 0;
    reconnectDelayMs = 1000;
    return;
  }
  if (wifiDownSince == 0) wifiDownSince = millis();
  if (millis() - wifiDownSince > WIFI_DEAD_MS) {
    Serial.println("wifi down 3min - restart");
    Serial.flush();
    ESP.restart();
  }
  if (millis() - lastReconnectTry < reconnectDelayMs) return;
  lastReconnectTry = millis();
  Serial.print("wifi retry ");
  Serial.println(reconnectDelayMs);
  wdtFeed();
  WiFi.disconnect();
  wdtFeed();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  wdtFeed();
  reconnectDelayMs *= 2;
  if (reconnectDelayMs > 60000) reconnectDelayMs = 60000;
}

void printHeartbeat() {
  elog("%lu.%lus hb heap=%u min=%u wifi=%d tuya=%d i2s=%d",
       millis() / 1000, (millis() % 1000) / 100,
       (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getMinFreeHeap(),
       (int)WiFi.status(), tuyaLive ? 1 : 0, i2sOk ? 1 : 0);
}

void netTask(void *arg) {
  (void)arg;
  wdtAddThis("net");

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("wifi");
  unsigned long w0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - w0 < 20000) {
    vTaskDelay(pdMS_TO_TICKS(250));
    Serial.print('.');
    wdtFeed();
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    wifiLost = false;
    ensureUdp();
    Serial.print("wifi ok ");
    Serial.println(WiFi.localIP());
    Serial.print("last bulb ip ");
    Serial.println(tuyaIp);
    labOnWifi();
    lastTuyaRefresh = millis();
    if (!tuyaEnsure()) {
      needHunt = true;
      if (findBulb(6000)) tuyaEnsure();
    }
    if (tuyaLive) {
      int st = tuyaQueryOn();
      if (st >= 0) {
        lampOn = (st == 1);
        lampKnown = true;
        saveLamp();
        Serial.print("lamp boot ");
        Serial.println(lampOn ? "ON" : "OFF");
      }
    } else {
      Serial.println("tuya boot miss - idle hunt");
      needHunt = true;
    }
  } else {
    wifiDownSince = millis();
    Serial.println("wifi FAIL - clap LED still works");
  }

  unsigned long lastHbLog = millis();
  for (;;) {
    wdtFeed();
    drainLog();
    handleWifiEvents();
    wifiBackoffTick();

    LampCommand command = {};
    if (clapQ && xQueueReceive(clapQ, &command, 0) == pdTRUE) {
      if (command.kind == CMD_DOUBLE) onDoubleClap();
      else if (command.kind == CMD_LAMP) applyLampSet(command);
    }

    if (WiFi.status() == WL_CONNECTED) {
      if (!udpUp) ensureUdp();
      if (!clapQ || uxQueueMessagesWaiting(clapQ) == 0) idleHuntTick();
      if (!tuyaBusy && (millis() - lastTuyaRefresh) >= TUYA_REFRESH_MS) {
        lastTuyaRefresh = millis();
        elog("tuya refresh");
        tuyaDrop();
        if (!tuyaConnectSavedIp()) needHunt = true;
      }
    }

    if (millis() - lastHbLog >= HB_MS) {
      lastHbLog = millis();
      printHeartbeat();
    }

    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

bool i2sStart() {
  // WS is GPIO 15 (strapping). Do not change unless the mic WS wire is moved.
  i2s.setPins(18, 15, -1, 19);
  i2s.setTimeout(20);
  if (!i2s.begin(I2S_MODE_STD, 16000, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO)) return false;
  const i2s_chan_handle_t rx = i2s.rxChan();
  i2s_event_callbacks_t callbacks = {};
  callbacks.on_recv_q_ovf = audioOverflow;
  if (i2s_channel_disable(rx) != ESP_OK) return false;
  const esp_err_t callbackStatus = i2s_channel_register_event_callback(rx, &callbacks, NULL);
  const esp_err_t enableStatus = i2s_channel_enable(rx);
  if (callbackStatus != ESP_OK || enableStatus != ESP_OK) return false;
  i2s.setTimeout(20);
  return true;
}

void i2sReinit() {
  i2s.end();
  delay(20);
  i2sPrev = 0;
  clapDetector.reset();
  clapPairer.reject();
  aiReset();
  audioSamples += CLAP_SAMPLE_RATE; // Mark the recording discontinuity explicitly.
  lastI2sMs = millis();
  if (!i2sStart()) {
    i2sOk = false;
    elog("i2s reinit FAIL");
  } else {
    lastI2sMs = millis();
    i2sOk = true;
    elog("i2s reinit");
  }
}

int readBlock(int32_t &peakRaw, int32_t &peakHp) {
  peakRaw = 0;
  peakHp = 0;
  int n = i2s.readBytes((char *)i2sBuf, sizeof(i2sBuf));
  if (n < 4) return 0;
  const uint32_t dropped = i2sDroppedSamples;
  if (dropped != i2sDropsSeen) {
    audioSamples += (uint32_t)(dropped - i2sDropsSeen);
    i2sDropsSeen = dropped;
    clapDetector.reset();
    clapPairer.reject();
    aiReset();
    // Every missing DMA block is a recording discontinuity, never quiet audio.
    elog("I2S lost samples=%lu", (unsigned long)dropped);
  }
  lastI2sMs = millis();
  i2sOk = true;
  int count = n / 4;
  for (int i = 0; i < count; i++) {
    int32_t s = i2sBuf[i] >> 8;
    int32_t a = abs32(s);
    if (a > peakRaw) peakRaw = a;
    int32_t hp = abs32(s - i2sPrev);
    i2sPrev = s;
    if (hp > peakHp) peakHp = hp;
    i2sBuf[i] = s;
  }
  pcmTeeBlock(i2sBuf, count, audioSamples);
  audioSamples += count;
  return count;
}

void maybeReinitI2s() {
  if (millis() - lastI2sMs < I2S_STALL_MS) return;
  i2sOk = false;
  i2sReinit();
}

void serviceLed() {
  if (!ledLit) return;
  if ((millis() - ledAt) < LED_MS) digitalWrite(2, HIGH);
  else {
    digitalWrite(2, LOW);
    ledLit = false;
  }
}

void queueDouble() {
  ledAt = millis();
  ledLit = true;
  const LampCommand command = {CMD_DOUBLE, false, 0, 0};
  if (tuyaBusy) { ++commandDrops; return; }
  // clapQ length is 4 (clap + lamp). Overwrite is illegal unless length == 1
  // and was panicking on DOUBLE before Tuya ever got the OFF.
  if (!clapQ || xQueueSend(clapQ, &command, 0) != pdTRUE) {
    ++commandDrops;
    elog("DOUBLE command queue full");
  }
}

// Translate the detector's wrapping 32-bit clock to the recording's 64-bit clock.
uint64_t absoluteSample(uint32_t sample) {
  return audioSamples - (uint32_t)((uint32_t)audioSamples - sample);
}

void rejectCandidate(const ClapCandidate &candidate) {
  clapPairer.reject();
  if (candidate.strong) { slamLocked = true; slamSample = candidate.onsetSample; }
}

void acceptCandidate(const ClapCandidate &candidate, bool verified) {
  const uint64_t onset = absoluteSample(candidate.onsetSample);
  const uint64_t decision = absoluteSample(candidate.decisionSample);
  const uint32_t duration = candidate.activeSamples * 1000UL / CLAP_SAMPLE_RATE;
  const bool acceptable = verified ||
      (candidate.legacySafe() && (clapPairer.waiting() || candidate.firstPlausible));
  if (!acceptable) {
    rejectCandidate(candidate);
    const uint8_t kind = candidate.strong ? EV_SLAM : EV_DULL;
    labEventAt(kind, duration, candidate.peakHp, decision, onset);
    elog("sample=%llu dur=%lu hp=%ld -> %s", (unsigned long long)onset,
         (unsigned long)duration, (long)candidate.peakHp, candidate.strong ? "slam" : "dull");
    return;
  }
  if (!verified && slamLocked) {
    if (candidate.onsetSample - slamSample < 11200) { // 700 ms after a slam.
      clapPairer.reject();
      labEventAt(EV_SLAM, duration, candidate.peakHp, decision, onset);
      return;
    }
    slamLocked = false;
  }
  const ClapPairResult result = clapPairer.accept(candidate.onsetSample);
  labEventAt(result == ClapPairResult::Ignored ? EV_IGNORE : EV_HIT,
             duration, candidate.peakHp, decision, onset);
  if (result == ClapPairResult::Double) {
    labEventAt(EV_DOUBLE, duration, candidate.peakHp, decision, onset);
    elog("sample=%llu DOUBLE", (unsigned long long)onset);
    queueDouble();
  }
}

void clapTick() {
  aiPoll();
  int32_t peakRaw, peakHp;
  const int count = readBlock(peakRaw, peakHp);
  const uint64_t firstSample = audioSamples - count;
  clapDetector.setArmThreshold(armLevel);
  for (int i = 0; i < count; ++i) {
    ClapCandidate candidate;
    const bool wasActive = clapDetector.active();
    const uint32_t sample = (uint32_t)(firstSample + i);
    const bool completed = clapDetector.processSample(i2sBuf[i], sample, candidate);
    if (!wasActive && clapDetector.active()) aiBegin(clapDetector.onsetSample());
    aiSample(i2sBuf[i], sample);
    if (completed) {
      aiCandidate(candidate);
#if !CLAP_ENABLE_AI || CLAP_AI_SHADOW
      acceptCandidate(candidate, false);
#endif
    }
  }
  // An event near the pair deadline must finish before the first clap expires.
  uint32_t expireAt = clapDetector.active() ? clapDetector.onsetSample() : (uint32_t)audioSamples;
#if CLAP_ENABLE_AI && !CLAP_AI_SHADOW
  uint32_t pending;
  if (aiPendingOnset(pending) && (uint32_t)((uint32_t)audioSamples - pending) >
      (uint32_t)((uint32_t)audioSamples - expireAt)) expireAt = pending;
#endif
  if (clapPairer.expire(expireAt))
    labEventAt(EV_SINGLE, 0, 0, audioSamples, audioSamples);
}

void setup() {
  Serial.setTxBufferSize(1024);
  Serial.begin(115200);
  pinMode(2, OUTPUT);

  logQ = xQueueCreate(24, LOG_MAX);
  if (!logQ) bootFailure("log queue allocation");

  Serial.print("reset=");
  Serial.println(resetReasonStr());
  Serial.flush();
  wdtSetup();

  mbedtls_md5((const unsigned char *)"yGAdlopoPVldABfn", 16, udpKey);

  prefs.begin("clap", false);
  nvsIp = prefs.getUInt("ip", (uint32_t)IPAddress(CLAP_TUYA_IP));
  tuyaIp = IPAddress(nvsIp);
  lampOn = prefs.getBool("on", false);
  lampKnown = prefs.getBool("known", false);
  lampBri = prefs.getInt("bri", 1000);
  lampTemp = prefs.getInt("temp", WARM_TEMP);
  if (lampBri < 10) lampBri = 10;
  if (lampBri > 1000) lampBri = 1000;
  if (lampTemp < 0) lampTemp = 0;
  if (lampTemp > 1000) lampTemp = 1000;
  lampNvsOn = lampOn;
  lampNvsKnown = lampKnown;
  lampNvsBri = lampBri;
  lampNvsTemp = lampTemp;
  lampNvsInit = true;

  for (int i = 0; i < 3; i++) {
    digitalWrite(2, HIGH);
    delay(120);
    digitalWrite(2, LOW);
    delay(120);
    wdtFeed();
  }

  bool microphoneStarted = false;
  for (unsigned attempt = 0; attempt < 3; ++attempt) {
    if (i2sStart()) { microphoneStarted = true; break; }
    i2s.end();
    Serial.println("I2S startup retry");
    delay(250);
    wdtFeed();
  }
  if (!microphoneStarted) bootFailure("microphone initialization");
  lastI2sMs = millis();
  i2sOk = true;

  int32_t raw, hp;
  for (int i = 0; i < 30; i++) readBlock(raw, hp);

  Serial.println("quiet 1s");
  int32_t noisePeaks[125];
  for (int i = 0; i < 125; i++) {
    readBlock(raw, hp);
    noisePeaks[i] = hp;
    wdtFeed();
  }
  for (int i = 1; i < 125; i++) {
    const int32_t value = noisePeaks[i];
    int j = i;
    while (j > 0 && noisePeaks[j - 1] > value) { noisePeaks[j] = noisePeaks[j - 1]; --j; }
    noisePeaks[j] = value;
  }
  floorHpStored = noisePeaks[62];
  if (floorHpStored < 1000) floorHpStored = 1000;
  applyArm();

  Serial.print("floor=");
  Serial.println(floorHpStored);
  Serial.print("arm=");
  Serial.println(armLevel);

  clapQ = xQueueCreate(4, sizeof(LampCommand));
  if (!clapQ) bootFailure("lamp command queue allocation");
  // tcpip_init() lives in WiFi.mode. streamTask bind()s HTTP/WS before
  // netTask runs, so the lwIP core mutex must exist first or IDF asserts
  // xQueueSemaphoreTake(NULL) and reboot-loops after arm=.
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  if (!labSetup()) bootFailure("Lab queue/task allocation");
  WiFi.onEvent(onWifiEvent);
  if (xTaskCreatePinnedToCore(netTask, "net", 12288, NULL, 1, NULL, 0) != pdPASS)
    bootFailure("network task allocation");
  if (!aiSetup()) bootFailure("AI initialization");

  vTaskPrioritySet(NULL, 4); // Audio preempts the lower-priority inference worker.

  elog("ready  |  double-clap/snap %s  |  continuous sample-clock detector", aiMode());
  elog("Reserve the bulb address in router DHCP for reliable local control");
}

void loop() {
  wdtFeed();
  serviceLed();
  maybeReinitI2s();
  clapTick();
  vTaskDelay(1); // Give inference/idle tasks time even when DMA has a backlog.
}
