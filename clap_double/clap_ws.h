#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Every line shares the handshake's original deadline. Stream's timeout is
// restarted for each character, so it cannot bound a slowly arriving line.
// The caller supplies a fixed buffer; no peer-controlled String allocation.
template <class Client, class Clock, class Wait>
bool clapWsReadHttpLine(Client &client, char *line, size_t capacity,
                        uint32_t started, uint32_t timeout,
                        Clock now, Wait wait) {
  if (!line || capacity == 0) return false;
  line[0] = 0;
  size_t size = 0;
  while (static_cast<uint32_t>(now() - started) < timeout) {
    if (!client.available()) {
      if (!client.connected()) return false;
      wait();
      continue;
    }
    const int value = client.read();
    if (value < 0) { wait(); continue; }
    if (value == '\n') {
      // HTTP CRLF is fully consumed before the first WebSocket byte is read.
      while (size && (line[size - 1] == '\r' || line[size - 1] == ' ' ||
                      line[size - 1] == '\t')) --size;
      size_t first = 0;
      while (first < size && (line[first] == ' ' || line[first] == '\t')) ++first;
      if (first) memmove(line, line + first, size - first);
      line[size - first] = 0;
      return true;
    }
    if (value == 0 || size + 1 >= capacity) return false;
    line[size++] = static_cast<char>(value);
    line[size] = 0;
  }
  return false;
}

// RFC6455 server frames are unmasked. Two complete frames may share one TCP
// write; the text metadata still precedes its binary PCM frame on the wire.
inline size_t clapWsFrame(uint8_t *out, size_t capacity, uint8_t opcode,
                          const uint8_t *data, size_t size) {
  if (size > 65535) return 0;
  const size_t header = size < 126 ? 2 : 4;
  if (capacity < header + size) return 0;
  out[0] = 0x80 | opcode;
  out[1] = size < 126 ? (uint8_t)size : 126;
  if (header == 4) { out[2] = (uint8_t)(size >> 8); out[3] = (uint8_t)size; }
  if (size) memcpy(out + header, data, size);
  return header + size;
}
