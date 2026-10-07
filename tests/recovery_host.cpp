// Compile the actual firmware functions with deterministic socket/Wi-Fi mocks.
// The Python driver extracts those bodies into temporary includes, so these
// tests cannot silently keep testing an obsolete copy of the implementation.
#include <algorithm>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <vector>
#include "clap_ws.h"

namespace Ws {
struct Client {
  bool live = true;
  std::deque<uint8_t> input;
  bool connected() const { return live; }
  int available() const { return static_cast<int>(input.size()); }
  int read() {
    if (input.empty()) return -1;
    const int value = input.front(); input.pop_front(); return value;
  }
  void feed(const std::vector<uint8_t> &bytes) {
    input.insert(input.end(), bytes.begin(), bytes.end());
  }
};
Client wsCli[1];
bool wsOn[1] = {true};
uint8_t wsRx[1][256] = {};
size_t wsRxN[1] = {};
bool failSend = false;
std::vector<std::string> applied;
std::vector<std::string> pongs;
void wsDrop(int i) { wsOn[i] = false; wsCli[i].live = false; wsRxN[i] = 0; }
bool wsSend(int i, uint8_t op, const uint8_t *data, size_t size) {
  assert(op == 0xA);
  if (failSend) { wsDrop(i); return false; }
  pongs.emplace_back(reinterpret_cast<const char *>(data), size);
  return true;
}
void labApplyWsText(const char *text, size_t size) { applied.emplace_back(text, size); }
#include "ws_under_test.inc"

std::vector<uint8_t> frame(uint8_t op, const std::string &payload, bool extended = false) {
  const uint8_t mask[] = {1, 23, 45, 67};
  std::vector<uint8_t> wire{uint8_t(0x80 | op), uint8_t(0x80 | (extended ? 126 : payload.size()))};
  if (extended) { wire.push_back(uint8_t(payload.size() >> 8)); wire.push_back(uint8_t(payload.size())); }
  wire.insert(wire.end(), mask, mask + 4);
  for (size_t i = 0; i < payload.size(); ++i) wire.push_back(uint8_t(payload[i]) ^ mask[i % 4]);
  return wire;
}
void append(std::vector<uint8_t> &out, const std::vector<uint8_t> &bytes) {
  out.insert(out.end(), bytes.begin(), bytes.end());
}
void run(const std::string &name) {
  if (name == "ws_fragmented_ping") {
    const auto wire = frame(9, "ping");
    for (size_t i = 0; i < wire.size(); ++i) {
      wsCli[0].feed({wire[i]}); labReadWs(0);
      assert(wsOn[0]);
      assert(pongs.size() == (i + 1 == wire.size() ? 1U : 0U));
    }
    assert(pongs[0] == "ping" && wsRxN[0] == 0);
  } else if (name == "ws_coalesced_ping_text") {
    auto wire = frame(9, "one"); append(wire, frame(1, "gain:3")); append(wire, frame(9, "two"));
    wsCli[0].feed(wire); labReadWs(0);
    assert(wsOn[0] && wsRxN[0] == 0);
    assert((pongs == std::vector<std::string>{"one", "two"}));
    assert((applied == std::vector<std::string>{"gain:3"}));
  } else if (name == "ws_split_length_mask") {
    const auto wire = frame(1, "arm:8", true);
    for (auto byte : wire) {
      wsCli[0].feed({byte}); labReadWs(0); assert(wsOn[0]);
    }
    assert(wsRxN[0] == 0 && applied.size() == 1 && applied[0] == "arm:8");
  } else if (name == "ws_invalid_after_text") {
    auto wire = frame(1, "gain:2");
    append(wire, {0x81, 0xff, 0, 0, 0, 0, 0, 0, 0, 1});
    append(wire, frame(1, "lamp:1"));
    wsCli[0].feed(wire); labReadWs(0);
    assert(!wsOn[0] && wsRxN[0] == 0);
    assert((applied == std::vector<std::string>{"gain:2"}));
  } else if (name == "ws_oversized_after_text") {
    auto wire = frame(1, "gain:2"); append(wire, frame(1, std::string(81, 'x')));
    append(wire, frame(1, "lamp:1")); wsCli[0].feed(wire); labReadWs(0);
    assert(!wsOn[0] && wsRxN[0] == 0);
    assert((applied == std::vector<std::string>{"gain:2"}));
  } else if (name == "ws_failed_pong_followed_text") {
    failSend = true;
    auto wire = frame(9, "ping"); append(wire, frame(1, "lamp:1"));
    wsCli[0].feed(wire); labReadWs(0);
    assert(!wsOn[0] && wsRxN[0] == 0 && applied.empty());
  } else if (name == "ws_close_followed_text") {
    auto wire = frame(8, ""); append(wire, frame(1, "lamp:1"));
    wsCli[0].feed(wire); labReadWs(0);
    assert(!wsOn[0] && wsRxN[0] == 0 && applied.empty());
  } else if (name == "ws_empty_ping") {
    wsCli[0].feed(frame(9, "")); labReadWs(0);
    assert(wsOn[0] && wsRxN[0] == 0 && pongs.size() == 1 && pongs[0].empty());
  } else if (name == "ws_disconnected") {
    wsCli[0].live = false; wsRxN[0] = 12; labReadWs(0);
    assert(!wsOn[0] && wsRxN[0] == 0);
  } else { assert(false); }
}
}

namespace Http {
struct Wire {
  std::string bytes;
  uint32_t &clock;
  uint32_t period = 0;
  uint32_t next = 0;
  bool live = true;
  bool connected() const { return live; }
  bool available() const { return !bytes.empty() && clock >= next; }
  int read() {
    if (!available()) return -1;
    const uint8_t byte = static_cast<uint8_t>(bytes[0]);
    bytes.erase(0, 1); next = clock + period; return byte;
  }
};
void run(const std::string &name) {
  uint32_t clock = 0;
  char line[513];
  auto now = [&]() { return clock; };
  auto wait = [&]() { ++clock; };
  if (name == "http_boundary") {
    auto ping = Ws::frame(9, "ping");
    Wire wire{"GET / HTTP/1.1\r\nConnection: Upgrade\r\n\r\n", clock};
    wire.bytes.append(reinterpret_cast<const char *>(ping.data()), ping.size());
    assert(clapWsReadHttpLine(wire, line, sizeof(line), 0, 400, now, wait));
    assert(strcmp(line, "GET / HTTP/1.1") == 0);
    assert(clapWsReadHttpLine(wire, line, sizeof(line), 0, 400, now, wait));
    assert(strcmp(line, "Connection: Upgrade") == 0);
    assert(clapWsReadHttpLine(wire, line, sizeof(line), 0, 400, now, wait) && line[0] == 0);
    assert(wire.bytes.size() == ping.size());
    Ws::wsCli[0].feed(std::vector<uint8_t>(wire.bytes.begin(), wire.bytes.end()));
    Ws::labReadWs(0);
    assert(Ws::pongs.size() == 1 && Ws::pongs[0] == "ping");
  } else if (name == "http_limits") {
    Wire exact{std::string(512, 'x') + "\n", clock};
    assert(clapWsReadHttpLine(exact, line, sizeof(line), 0, 400, now, wait));
    assert(strlen(line) == 512);
    Wire tooLong{std::string(513, 'x') + "\n", clock};
    assert(!clapWsReadHttpLine(tooLong, line, sizeof(line), 0, 400, now, wait));
    Wire embeddedNul{std::string("abc\0def\n", 8), clock};
    assert(!clapWsReadHttpLine(embeddedNul, line, sizeof(line), 0, 400, now, wait));
    Wire empty{"\n", clock};
    assert(!clapWsReadHttpLine(empty, nullptr, sizeof(line), 0, 400, now, wait));
    assert(!clapWsReadHttpLine(empty, line, 0, 0, 400, now, wait));
    Wire trim{"  Header: value \t\r\n", clock};
    assert(clapWsReadHttpLine(trim, line, sizeof(line), 0, 400, now, wait));
    assert(strcmp(line, "Header: value") == 0);
    Wire disconnected{"", clock}; disconnected.live = false;
    assert(!clapWsReadHttpLine(disconnected, line, sizeof(line), 0, 400, now, wait));
    assert(clock == 0);
  } else if (name == "http_shared_deadline") {
    Wire wire{"ok\nslow second line\n", clock, 50};
    assert(clapWsReadHttpLine(wire, line, sizeof(line), 0, 400, now, wait));
    assert(clock == 100 && strcmp(line, "ok") == 0);
    assert(!clapWsReadHttpLine(wire, line, sizeof(line), 0, 400, now, wait));
    assert(clock == 400 && !wire.bytes.empty());
  } else if (name == "http_clock_wrap") {
    clock = 0xfffffff0U;
    Wire wire{"", clock}; const uint32_t start = clock;
    assert(!clapWsReadHttpLine(wire, line, sizeof(line), start, 400, now, wait));
    assert(static_cast<uint32_t>(clock - start) == 400);
  } else { assert(false); }
}
}

namespace Recovery {
struct IPAddress {
  uint8_t bytes[4];
  IPAddress() : bytes{0, 0, 0, 0} {}
  IPAddress(int a, int b, int c, int d) : bytes{uint8_t(a), uint8_t(b), uint8_t(c), uint8_t(d)} {}
  uint8_t operator[](int at) const { return bytes[at]; }
  bool operator!=(const IPAddress &other) const { return memcmp(bytes, other.bytes, 4) != 0; }
};
constexpr int WL_CONNECTED = 3;
constexpr unsigned long SCAN_GAP_MS = 180;
unsigned long clock = 1000;
unsigned long millis() { return clock; }
struct Wifi {
  int state = WL_CONNECTED;
  int status() const { return state; }
  IPAddress localIP() const { return IPAddress(192, 0, 2, 42); }
} WiFi;
struct Logger {
  template<class T> void print(const T &) {}
  template<class T> void println(const T &) {}
} Serial;
bool tuyaLive = false, needHunt = false, udpUp = true, lampOn = false, lampKnown = false;
bool socketAlive = true, connectOk = true, queryDrops = false, scanOpen = false, ensureOk = true;
bool heartbeatDrops = false;
bool pollClearsHunt = false, wifiGotIp = false, wifiLost = false;
unsigned long lastDiscover = 0, lastHbTry = 0, lastScanHost = 0;
unsigned long reconnectDelayMs = 60000, wifiDownSince = 123;
int scanHost = 2, queryResult = 1;
int drops = 0, savedAttempts = 0, scans = 0, lampSaves = 0, udpStarts = 0, udpStops = 0, labCalls = 0;
IPAddress tuyaIp(192, 0, 2, 20);
std::vector<std::string> calls;
bool tuyaSockAlive() { return socketAlive; }
void tuyaDrop() { tuyaLive = false; socketAlive = false; ++drops; calls.push_back("drop"); }
void pollUdp() { calls.push_back("poll"); if (pollClearsHunt) needHunt = false; }
void tuyaBroadcast() { calls.push_back("broadcast"); }
bool tuyaHeartbeat() {
  calls.push_back("heartbeat");
  if (heartbeatDrops) { tuyaDrop(); return false; }
  return true;
}
bool tuyaConnectSavedIp() {
  ++savedAttempts; calls.push_back("saved"); tuyaLive = socketAlive = connectOk; return connectOk;
}
int tuyaQueryOn() { calls.push_back("query"); if (queryDrops) tuyaLive = false; return queryResult; }
void saveLamp() { ++lampSaves; }
bool portOpen(const IPAddress &) { ++scans; calls.push_back("scan"); return scanOpen; }
void saveIp(const IPAddress &ip) { tuyaIp = ip; }
bool tuyaEnsure() { calls.push_back("ensure"); tuyaLive = socketAlive = ensureOk; return ensureOk; }
void ensureUdp() { udpUp = true; ++udpStarts; }
void stopUdp() { udpUp = false; ++udpStops; }
void labOnWifi() { ++labCalls; }
#include "recovery_under_test.inc"

void run(const std::string &name) {
  if (name == "recovery_dead_socket") {
    tuyaLive = true; socketAlive = false; idleHuntTick();
    assert(drops == 1 && savedAttempts == 1 && tuyaLive && !needHunt);
    assert((calls == std::vector<std::string>{"drop", "poll", "saved", "query"}));
    assert(lampKnown && lampOn && lampSaves == 1 && scans == 0);
  } else if (name == "recovery_dead_socket_no_udp") {
    tuyaLive = true; socketAlive = false; udpUp = false; idleHuntTick();
    assert(drops == 1 && !tuyaLive && needHunt && savedAttempts == 0 && scans == 0);
  } else if (name == "recovery_heartbeat_disconnect") {
    tuyaLive = true; heartbeatDrops = true; clock = 16001; idleHuntTick();
    assert(drops == 1 && savedAttempts == 1 && tuyaLive && socketAlive && !needHunt);
    assert((calls == std::vector<std::string>{"poll", "heartbeat", "drop", "saved", "query"}));
  } else if (name == "recovery_saved_first") {
    idleHuntTick();
    assert(savedAttempts == 1 && scans == 0 && tuyaLive && !needHunt);
  } else if (name == "recovery_rate_limit") {
    connectOk = false; idleHuntTick();
    assert(savedAttempts == 1 && scans == 1 && needHunt);
    clock += 200; idleHuntTick();
    assert(savedAttempts == 1 && scans == 2 && !tuyaLive && needHunt);
    clock = 16000; idleHuntTick();
    assert(savedAttempts == 2 && scans == 3);
    tuyaIp = IPAddress(192, 0, 2, 21); clock += 1; idleHuntTick();
    assert(savedAttempts == 3); // Newly discovered saved address is tried promptly.
  } else if (name == "recovery_discovery_no_session") {
    pollClearsHunt = true; needHunt = true; idleHuntTick();
    assert(savedAttempts == 1 && tuyaLive && !needHunt);
  } else if (name == "recovery_query_drops") {
    queryDrops = true; queryResult = -1; idleHuntTick();
    assert(savedAttempts == 1 && scans == 1 && !tuyaLive && needHunt);
    assert(!lampKnown && lampSaves == 0);
  } else if (name == "recovery_scan_fallback") {
    connectOk = false; scanOpen = true; idleHuntTick();
    assert(savedAttempts == 1 && scans == 1 && tuyaLive && !needHunt);
    assert(tuyaIp[3] == 2 && lampKnown && lampOn && lampSaves == 1);
    assert((calls == std::vector<std::string>{"poll", "saved", "scan", "ensure", "query"}));
  } else if (name == "recovery_wifi_lost") {
    tuyaLive = true; wifiLost = true; WiFi.state = 0; handleWifiEvents();
    assert(!wifiLost && drops == 1 && !tuyaLive && needHunt && !udpUp && udpStops == 1);
  } else if (name == "recovery_wifi_reconnected") {
    udpUp = false; wifiGotIp = true; handleWifiEvents();
    assert(!wifiGotIp && udpUp && udpStarts == 1 && labCalls == 1 && needHunt);
    assert(wifiDownSince == 0 && reconnectDelayMs == 1000);
    idleHuntTick(); assert(tuyaLive && !needHunt && savedAttempts == 1);
  } else if (name == "recovery_stale_loss_event") {
    tuyaLive = true; wifiLost = true; handleWifiEvents();
    assert(!wifiLost && drops == 0 && udpStops == 0 && tuyaLive && !needHunt);
  } else if (name == "recovery_coalesced_wifi_events") {
    wifiGotIp = true; wifiLost = true; handleWifiEvents();
    assert(!wifiGotIp && !wifiLost && udpStarts == 1 && udpStops == 0 && needHunt);
    idleHuntTick(); assert(tuyaLive && savedAttempts == 1);
  } else { assert(false); }
}
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const std::string name(argv[1]);
  if (name.find("ws_") == 0) Ws::run(name);
  else if (name.find("http_") == 0) Http::run(name);
  else if (name.find("recovery_") == 0) Recovery::run(name);
  else return 3;
  std::printf("%s passed\n", name.c_str());
  return 0;
}
