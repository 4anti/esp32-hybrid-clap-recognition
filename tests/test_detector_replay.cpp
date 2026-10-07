// Host replay of the exact portable firmware detector.
// Usage: detector_replay audio.pcm [raw24_per_pcm16_sample=64]
// Legacy recordings defaulted to listenGain=4: 16-bit PCM * 64 approximates
// raw24. Lost clipping and missing gain metadata prevent exact reconstruction.
#include "../clap_double/clap_detector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--pair") == 0) {
    ClapPairer pairer;
    unsigned onset;
    int verified;
    unsigned accepted = 0, doubles = 0;
    while (scanf("%u %d", &onset, &verified) == 2) {
      if (verified) {
        ++accepted;
        if (pairer.accept(onset) == ClapPairResult::Double) ++doubles;
      } else pairer.reject();
    }
    printf("{\"accepted\":%u,\"doubles\":%u}\n", accepted, doubles);
    return 0;
  }
  if (argc < 2 || argc > 7) {
    fputs("usage: detector_replay audio.pcm [sample_scale=64] [arm=200000] [--events] [--votes votes.tsv]\n", stderr);
    return 2;
  }
  const int scale = argc >= 3 ? atoi(argv[2]) : 64;
  if (scale < 1 || scale > 256) {
    fputs("sample_scale must be 1..256\n", stderr);
    return 2;
  }
  FILE *input = fopen(argv[1], "rb");
  if (!input) {
    perror("audio.pcm");
    return 2;
  }
  ClapDetector detector;
  const int arm = argc >= 4 ? atoi(argv[3]) : 200000;
  if (arm <= 0) return 2;
  detector.setArmThreshold(arm);
  const bool printEvents = argc >= 5 && strcmp(argv[4], "--events") == 0;
  const bool verify = argc == 7 && strcmp(argv[5], "--votes") == 0;
  std::map<uint32_t, bool> votes;
  if (verify) {
    FILE *voteFile = fopen(argv[6], "r");
    if (!voteFile) return 2;
    unsigned onset;
    int vote;
    while (fscanf(voteFile, "%u %d", &onset, &vote) == 2) {
      if ((vote != 0 && vote != 1) || votes.count(onset)) { fclose(voteFile); return 2; }
      votes[onset] = vote != 0;
    }
    const bool badVotes = !feof(voteFile) || ferror(voteFile);
    fclose(voteFile);
    if (badVotes) return 2;
  }
  ClapPairer pairer;
  bool slamLocked = false;
  uint32_t slamSample = 0;
  uint32_t samples = 0, candidates = 0, plausible = 0, strong = 0;
  uint32_t timedOut = 0, accepted = 0, doubles = 0;
  unsigned char bytes[2];
  while (fread(bytes, 1, 2, input) == 2) {
    const uint16_t unsignedSample = static_cast<uint16_t>(bytes[0]) |
        static_cast<uint16_t>(static_cast<uint16_t>(bytes[1]) << 8);
    const int32_t pcm = unsignedSample >= 32768 ?
        static_cast<int32_t>(unsignedSample) - 65536 : unsignedSample;
    ClapCandidate event;
    if (detector.processSample(pcm * scale, samples, event)) {
      if (printEvents) printf("{\"t\":\"candidate\",\"onset\":%u,\"decision\":%u,\"timedOut\":%s,"
          "\"legacySafe\":%s,\"firstPlausible\":%s,\"strong\":%s}\n",
          event.onsetSample, event.decisionSample, event.timedOut ? "true" : "false",
          event.legacySafe() ? "true" : "false", event.firstPlausible ? "true" : "false", event.strong ? "true" : "false");
      ++candidates;
      if (event.plausible) ++plausible;
      if (event.strong) ++strong;
      if (event.timedOut) ++timedOut;
      if (verify && !votes.count(event.onsetSample)) { fclose(input); return 2; }
      bool safe = (!verify || votes[event.onsetSample]) && event.legacySafe() &&
          (pairer.waiting() || event.firstPlausible);
      if (!safe && event.strong) { slamLocked = true; slamSample = event.onsetSample; }
      if (safe && slamLocked) {
        if (event.onsetSample - slamSample < 11200) safe = false;
        else slamLocked = false;
      }
      if (safe) {
        ++accepted;
        if (pairer.accept(event.onsetSample) == ClapPairResult::Double) ++doubles;
      } else {
        pairer.reject();
      }
    }
    ++samples;
    // Firmware expires at each block boundary, holding an active event's onset.
    if (samples % 128 == 0) pairer.expire(detector.active() ? detector.onsetSample() : samples);
  }
  const bool failed = ferror(input) != 0;
  fclose(input);
  if (failed) return 2;
  printf("{\"samples\":%lu,\"scale\":%d,\"candidates\":%lu,"
         "\"plausible\":%lu,\"strong\":%lu,\"timed_out\":%lu,"
         "\"legacy_accepted\":%lu,\"legacy_doubles\":%lu}\n",
         static_cast<unsigned long>(samples), scale,
         static_cast<unsigned long>(candidates), static_cast<unsigned long>(plausible),
         static_cast<unsigned long>(strong), static_cast<unsigned long>(timedOut),
         static_cast<unsigned long>(accepted), static_cast<unsigned long>(doubles));
}
