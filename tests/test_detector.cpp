#include "../clap_double/clap_detector.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
#include <vector>

static std::vector<ClapCandidate> run(const std::vector<int32_t> &audio,
                                    uint32_t origin = 0,
                                    unsigned blockSize = 128) {
  ClapDetector detector;
  std::vector<ClapCandidate> candidates;
  for (unsigned block = 0; block < audio.size(); block += blockSize) {
    for (unsigned i = block; i < audio.size() && i < block + blockSize; ++i) {
      ClapCandidate candidate;
      if (detector.processSample(audio[i], origin + i, candidate))
        candidates.push_back(candidate);
    }
  }
  return candidates;
}

static void testBlockOffsetsAndDuration() {
  for (unsigned offset = 0; offset < 128; ++offset) {
    std::vector<int32_t> audio(3000, 0);
    const uint32_t onset = 512 + offset;
    audio[onset] = 600000;
    const auto events = run(audio);
    assert(events.size() == 1);
    const ClapCandidate &event = events[0];
    assert(event.onsetSample == onset);
    assert(event.lastHotSample == onset + 1);
    assert(event.decisionSample == onset + 1 + 1280);
    assert(event.activeSamples == 2); // Silence tail is never sound duration.
    assert(event.hotSamples == 2);
    assert(event.peakHp == 600000 && event.peakRaw == 600000);
    assert(event.plausible && event.firstPlausible && event.legacySafe());
    const auto singleSampleBlocks = run(audio, 0, 1);
    assert(singleSampleBlocks[0].onsetSample == event.onsetSample);
    assert(singleSampleBlocks[0].decisionSample == event.decisionSample);
  }
}

static void testWeakSnapAndClippedCandidate() {
  std::vector<int32_t> audio(3000, 0);
  audio[100] = 450000;
  auto events = run(audio);
  assert(events.size() == 1);
  assert(events[0].plausible && !events[0].firstPlausible);
  audio[100] = 8388607;
  events = run(audio);
  assert(events[0].plausible && events[0].strong);
  assert(events[0].clippedSamples == 1);
  assert(!events[0].legacySafe()); // Strong means verify, never assume noise.
}

static void testContinuousNoiseAndRecovery() {
  std::vector<int32_t> audio(15000, 0);
  for (unsigned i = 100; i < 10000; ++i)
    audio[i] = i % 2 ? 600000 : -600000;
  audio[12000] = 600000;
  const auto events = run(audio);
  assert(events.size() == 2); // Sustained noise cannot create periodic positives.
  assert(events[0].timedOut && !events[0].plausible);
  assert(events[0].decisionSample == events[0].onsetSample + 3839);
  assert(events[1].onsetSample == 12000 && events[1].plausible);
}

static void testDetectorWrapAndRestart() {
  std::vector<int32_t> audio(3000, 0);
  audio[100] = 600000;
  const uint32_t origin = UINT32_MAX - 200;
  const auto events = run(audio, origin);
  assert(events.size() == 1);
  assert(events[0].onsetSample == origin + 100);
  assert(events[0].activeSamples == 2 && events[0].plausible);
  assert(events[0].decisionSample == origin + 1381);

  ClapDetector detector;
  ClapCandidate out;
  assert(!detector.processSample(INT32_MIN, 0, out));
  assert(!detector.processSample(INT32_MAX, 1, out));
  for (uint32_t i = 2; i <= 1281; ++i)
    detector.processSample(INT32_MAX, i, out);
  assert(out.peakHp == INT32_MAX); // No signed subtraction overflow.
  detector.reset();
  assert(!detector.processSample(600000, 5000, out));
  assert(!detector.active()); // Restart baseline does not fabricate a clap.
}

static void testPairBoundariesAndEchoes() {
  ClapPairer pairer;
  assert(pairer.accept(0) == ClapPairResult::First); // Zero is a valid onset.
  assert(pairer.accept(2399) == ClapPairResult::Ignored);
  assert(pairer.firstOnsetSample() == 0); // Echo does not shift pair timing.
  assert(pairer.accept(2400) == ClapPairResult::Double);
  assert(!pairer.waiting());
  assert(pairer.accept(5599) == ClapPairResult::Ignored);
  assert(pairer.accept(5600) == ClapPairResult::First);
  pairer.reset();
  assert(pairer.accept(10) == ClapPairResult::First);
  assert(pairer.accept(12810) == ClapPairResult::Double);
  pairer.reset();
  assert(pairer.accept(10) == ClapPairResult::First);
  assert(pairer.accept(12811) == ClapPairResult::First);
  assert(pairer.firstOnsetSample() == 12811);
}

static void testPairRejectExpiryAndWrap() {
  ClapPairer pairer;
  pairer.accept(100);
  pairer.reject();
  assert(!pairer.waiting());
  assert(pairer.accept(2500) == ClapPairResult::First);
  assert(!pairer.expire(15300));
  assert(pairer.expire(15301));
  assert(!pairer.expire(20000));
  pairer.reset();
  const uint32_t onset = UINT32_MAX - 1000;
  assert(pairer.accept(onset) == ClapPairResult::First);
  assert(pairer.accept(onset + 2400) == ClapPairResult::Double);
  assert(pairer.accept(onset + 5599) == ClapPairResult::Ignored);
  assert(pairer.accept(onset + 5600) == ClapPairResult::First);
}

static void testDecisionLatencyDoesNotChangePairSpacing() {
  std::vector<int32_t> audio(12000, 0);
  // First sound rings for 100 ms; second sound is a short transient.
  for (unsigned i = 100; i < 1700; ++i)
    audio[i] = i % 2 ? 600000 : -600000;
  audio[4900] = 600000;
  const auto events = run(audio);
  assert(events.size() == 2 && events[0].plausible && events[1].plausible);
  assert(events[1].onsetSample - events[0].onsetSample == 4800);
  assert(events[1].decisionSample - events[0].decisionSample != 4800);
  ClapPairer pairer;
  assert(pairer.accept(events[0].onsetSample) == ClapPairResult::First);
  assert(pairer.accept(events[1].onsetSample) == ClapPairResult::Double);
}

int main() {
  testBlockOffsetsAndDuration();
  testWeakSnapAndClippedCandidate();
  testContinuousNoiseAndRecovery();
  testDetectorWrapAndRestart();
  testPairBoundariesAndEchoes();
  testPairRejectExpiryAndWrap();
  testDecisionLatencyDoesNotChangePairSpacing();
  puts("detector: all sample timing, pairing, noise, overflow and restart tests passed");
}
