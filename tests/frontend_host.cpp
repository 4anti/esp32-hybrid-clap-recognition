#include <cstdio>
#include <cstdint>
#include <initializer_list>
#include <string>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include "clap_frontend.h"
#include "clap_pcm.h"
#include "clap_ws.h"

struct HttpWire {
  std::string remaining;
  bool connected() const { return true; }
  int available() const { return !remaining.empty(); }
  int read() {
    if (remaining.empty()) return -1;
    const uint8_t result = static_cast<uint8_t>(remaining[0]);
    remaining.erase(0, 1);
    return result;
  }
};

static void checkHandshakeBoundary() {
  // Exercise the actual line reader with a complete HTTP upgrade immediately
  // followed by a masked control frame. No LF may enter the WS frame parser.
  const uint8_t ping[] = {0x89, 0x84, 1, 2, 3, 4, 'p' ^ 1, 'i' ^ 2, 'n' ^ 3, 'g' ^ 4};
  HttpWire input{"GET / HTTP/1.1\r\nSec-WebSocket-Key: test\r\nConnection: Upgrade\r\n\r\n"};
  input.remaining.append(reinterpret_cast<const char *>(ping), sizeof(ping));
  uint32_t tick = 0;
  char line[513];
  auto readLine = [&]() {
    return clapWsReadHttpLine(input, line, sizeof(line), 0, 400,
        [&]() { return tick; }, [&]() { ++tick; });
  };
  assert(readLine() && strcmp(line, "GET / HTTP/1.1") == 0);
  assert(readLine() && strcmp(line, "Sec-WebSocket-Key: test") == 0);
  assert(readLine() && strcmp(line, "Connection: Upgrade") == 0);
  assert(readLine() && line[0] == 0);
  assert(input.remaining.size() == sizeof(ping));
  assert(memcmp(input.remaining.data(), ping, sizeof(ping)) == 0);
}

static void checkStreamFrames() {
  // Decode independently of the writer: batching must preserve two separate
  // metadata/audio messages, including the 16-bit audio payload length.
  const uint8_t metadata[] = "{\"sample\":4294967303,\"samples\":512}";
  uint8_t pcm[1024], wire[1200];
  for (unsigned i = 0; i < sizeof(pcm); ++i) pcm[i] = uint8_t(i);
  size_t n = clapWsFrame(wire, sizeof(wire), 1, metadata, sizeof(metadata) - 1);
  assert(n == sizeof(metadata) + 1);
  n += clapWsFrame(wire + n, sizeof(wire) - n, 2, pcm, sizeof(pcm));
  size_t at = 0;
  for (int frame = 0; frame < 2; ++frame) {
    assert(wire[at++] == uint8_t(0x81 + frame));
    const uint8_t length = wire[at++];
    assert((length & 0x80) == 0); // server frames are unmasked
    size_t size = length;
    if (length == 126) {
      size = (size_t(wire[at]) << 8) | wire[at + 1];
      at += 2;
    }
    assert(size == (frame == 0 ? sizeof(metadata) - 1 : sizeof(pcm)));
    assert(memcmp(wire + at, frame == 0 ? metadata : pcm, size) == 0);
    at += size;
  }
  assert(at == n);
  assert(clapWsFrame(wire, 1027, 2, pcm, sizeof(pcm)) == 0);
  assert(clapWsFrame(wire, sizeof(wire), 2, pcm, 65536) == 0);
  assert(clapWsFrame(wire, sizeof(wire), 10, nullptr, 0) == 2);
  assert(wire[0] == 0x8a && wire[1] == 0);
  for (size_t size : {size_t(125), size_t(126)}) {
    const size_t header = size == 125 ? 2 : 4;
    assert(clapWsFrame(wire, sizeof(wire), 2, pcm, size) == size + header);
    assert(wire[1] == (size == 125 ? 125 : 126));
  }
}

int main(int argc, char **argv) {
  checkStreamFrames();
  checkHandshakeBoundary();
  // Overflow must retain a contiguous suffix with correct absolute offsets,
  // including clocks beyond uint32_t. Playback gain never changes stored PCM.
  ClapPcmBuffer stream;
  int32_t raw[128];
  uint64_t first = (uint64_t(1) << 32) + 7;
  for (int block = 0; block < 40; ++block) {
    for (int i = 0; i < 128; ++i) raw[i] = (block * 128 + i - 2500) * 256;
    stream.push(raw, 128, first + block * 128);
  }
  assert(stream.available() == 4096);
  int16_t popped[256];
  uint64_t position;
  uint32_t dropped;
  for (int packet = 0; packet < 16; ++packet) {
    assert(stream.pop(popped, 256, position, dropped) == 256);
    assert(dropped == 1024 && position == first + 1024 + packet * 256);
    for (int i = 0; i < 256; ++i) assert(popped[i] == 1024 + packet * 256 + i - 2500);
  }
  assert(stream.pop(popped, 256, position, dropped) == 0);
  assert(stream.available() == 0);
  // A discontinuity must discard stale samples and restart the sequence.
  stream.push(raw, 128, first + 6000);
  stream.push(raw, 128, first + 9000);
  assert(stream.pop(popped, 256, position, dropped) == 128);
  assert(position == first + 9000 && dropped == 1152);
  if (argc != 3 && argc != 4) return 2;
  int16_t pcm[8000];
  FILE *input = fopen(argv[1], "rb");
  if (!input || fread(pcm, sizeof(int16_t), 8000, input) != 8000) return 3;
  fclose(input);
  ClapFrontend frontend;
  float features[64 * 48];
  frontend.compute(pcm, features);
  FILE *output = fopen(argv[2], "wb");
  if (!output || fwrite(features, sizeof(float), 64 * 48, output) != 64 * 48) return 4;
  fclose(output);
  if (argc == 4) {
    float mean[64], deviation[64];
    for (int mel = 0; mel < 64; ++mel) { mean[mel] = -3.0f + mel / 64.0f; deviation[mel] = 1.0f + mel / 128.0f; }
    int8_t quantized[64 * 48];
    assert(!frontend.computeQuantized(pcm, quantized, mean, deviation, 0, -3));
    assert(frontend.computeQuantized(pcm, quantized, mean, deviation, 0.075f, -3));
    int8_t target[64 * 48];
    assert(frontend.computeQuantizedTarget(pcm + ClapFrontend::kTargetOffset,
        target, mean, deviation, 0.075f, -3));
    for (int mel = 0; mel < 64; ++mel)
      for (int frame = 0; frame < 48; ++frame)
        assert(target[mel * 48 + frame] == (frame < 32 ? -3 : quantized[mel * 48 + frame]));
    FILE *quant = fopen(argv[3], "wb");
    assert(quant && fwrite(quantized, 1, sizeof(quantized), quant) == sizeof(quantized));
    fclose(quant);
  }
  return 0;
}
