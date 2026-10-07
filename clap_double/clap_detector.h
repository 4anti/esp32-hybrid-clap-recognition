#ifndef CLAP_DETECTOR_H
#define CLAP_DETECTOR_H

#include <stdint.h>
#include <limits.h>

// All time values are microphone sample positions, not task/AI completion times.
// Unsigned differences remain valid across uint32_t wrap for these short windows.
struct ClapDetectorConfig {
  int32_t armThreshold = 200000;
  int32_t candidatePeakThreshold = 400000;
  int32_t firstPeakThreshold = 500000;
  int32_t strongPeakThreshold = 4500000;
  int32_t clippingThreshold = 0x7F0000;
  uint32_t quietSamples = 1280;          // 80 ms at 16 kHz.
  uint32_t maximumActiveSamples = 2560; // Old 240 ms event minus 80 ms tail.
  uint32_t maximumEventSamples = 3840;  // End continuous noise after 240 ms.
};

struct ClapCandidate {
  uint32_t onsetSample = 0;
  uint32_t lastHotSample = 0;
  uint32_t decisionSample = 0;
  uint32_t activeSamples = 0;
  uint32_t hotSamples = 0;
  uint32_t clippedSamples = 0;
  int32_t peakHp = 0;
  int32_t peakRaw = 0;
  bool timedOut = false;
  bool strong = false;
  bool plausible = false;
  bool firstPlausible = false;

  // Preserve the previous detector's conservative slam protection until a
  // trained verifier can distinguish close/loud claps from impulsive negatives.
  bool legacySafe() const { return plausible && !strong; }
};

class ClapDetector {
 public:
  explicit ClapDetector(const ClapDetectorConfig &config = ClapDetectorConfig())
      : config_(config) {}

  void setArmThreshold(int32_t threshold) {
    config_.armThreshold = threshold > 0 ? threshold : 1;
  }

  bool active() const { return active_; }
  bool suppressed() const { return suppressed_; }
  uint32_t onsetSample() const { return event_.onsetSample; }

  // Reset after an audio discontinuity. The first new sample seeds the
  // difference filter so a microphone restart cannot invent an impulse.
  void reset() {
    active_ = false;
    suppressed_ = false;
    quiet_ = false;
    havePrevious_ = false;
    event_ = ClapCandidate();
  }

  // Call for every input sample, including while a verifier/network is busy.
  // Emits each finished transient exactly once. Rejected events are emitted as
  // well so telemetry and the verifier can account for every decision.
  bool processSample(int32_t sample, uint32_t sampleIndex, ClapCandidate &out) {
    if (!havePrevious_) {
      previous_ = sample;
      havePrevious_ = true;
      return false;
    }
    const int32_t hp = differenceMagnitude(sample, previous_);
    previous_ = sample;
    const int32_t raw = magnitude(sample);
    const bool hot = hp > config_.armThreshold;

    if (suppressed_) {
      if (hot) {
        quiet_ = false;
      } else if (!quiet_) {
        quiet_ = true;
        quietAt_ = sampleIndex;
      } else if (sampleIndex - quietAt_ + 1 >= config_.quietSamples) {
        suppressed_ = false;
        quiet_ = false;
      }
      return false;
    }

    if (!active_) {
      if (!hot) return false;
      event_ = ClapCandidate();
      event_.onsetSample = sampleIndex;
      event_.lastHotSample = sampleIndex;
      active_ = true;
    }

    if (raw > event_.peakRaw) event_.peakRaw = raw;
    if (hp > event_.peakHp) event_.peakHp = hp;
    if (raw >= config_.clippingThreshold) ++event_.clippedSamples;
    if (hot) {
      event_.lastHotSample = sampleIndex;
      ++event_.hotSamples;
    }

    const bool quietEnough = !hot &&
        sampleIndex - event_.lastHotSample >= config_.quietSamples;
    const bool deadline = sampleIndex - event_.onsetSample + 1 >=
        config_.maximumEventSamples;
    if (!quietEnough && !deadline) return false;

    event_.decisionSample = sampleIndex;
    event_.activeSamples = event_.lastHotSample - event_.onsetSample + 1;
    event_.timedOut = deadline && !quietEnough;
    event_.strong = event_.peakHp >= config_.strongPeakThreshold;
    event_.plausible = !event_.timedOut &&
        event_.activeSamples <= config_.maximumActiveSamples &&
        event_.peakHp >= config_.candidatePeakThreshold;
    event_.firstPlausible = event_.plausible &&
        event_.peakHp >= config_.firstPeakThreshold;
    out = event_;
    active_ = false;
    // A sustained sound must become quiet before it can create another event.
    // This avoids an endless stream of candidates from music or microphone noise.
    suppressed_ = event_.timedOut;
    quiet_ = false;
    return true;
  }

 private:
  static int32_t magnitude(int32_t value) {
    if (value == INT32_MIN) return INT32_MAX;
    return value < 0 ? -value : value;
  }

  static int32_t differenceMagnitude(int32_t current, int32_t previous) {
    int64_t difference = static_cast<int64_t>(current) - previous;
    if (difference < 0) difference = -difference;
    return difference > INT32_MAX ? INT32_MAX : static_cast<int32_t>(difference);
  }

  ClapDetectorConfig config_;
  ClapCandidate event_;
  int32_t previous_ = 0;
  uint32_t quietAt_ = 0;
  bool havePrevious_ = false;
  bool active_ = false;
  bool suppressed_ = false;
  bool quiet_ = false;
};

struct ClapPairConfig {
  uint32_t minimumPairSamples = 2400; // 150 ms at 16 kHz, inclusive.
  uint32_t maximumPairSamples = 12800; // 800 ms, inclusive.
  uint32_t refractorySamples = 3200; // 200 ms after a completed double.
};

enum class ClapPairResult : uint8_t { First, Double, Ignored };

class ClapPairer {
 public:
  explicit ClapPairer(const ClapPairConfig &config = ClapPairConfig())
      : config_(config) {}

  bool waiting() const { return waiting_; }
  uint32_t firstOnsetSample() const { return first_; }

  void reset() { waiting_ = false; locked_ = false; }

  // Submit only verified positives, in onset order. Inference latency must not
  // alter spacing or let a rejected sound become a member of a pair.
  ClapPairResult accept(uint32_t onsetSample) {
    if (locked_) {
      if (onsetSample - lockAt_ < config_.refractorySamples)
        return ClapPairResult::Ignored;
      locked_ = false;
    }
    if (!waiting_ || onsetSample - first_ > config_.maximumPairSamples) {
      first_ = onsetSample;
      waiting_ = true;
      return ClapPairResult::First;
    }
    if (onsetSample - first_ < config_.minimumPairSamples)
      return ClapPairResult::Ignored; // An echo never moves the first onset.
    waiting_ = false;
    locked_ = true;
    lockAt_ = onsetSample;
    return ClapPairResult::Double;
  }

  void reject() { waiting_ = false; }

  // For single-clap telemetry. Do not expire past the oldest unclassified
  // candidate when AI is asynchronous: its later result still owns its onset.
  bool expire(uint32_t currentSample) {
    if (!waiting_ || currentSample - first_ <= config_.maximumPairSamples)
      return false;
    waiting_ = false;
    return true;
  }

 private:
  ClapPairConfig config_;
  uint32_t first_ = 0;
  uint32_t lockAt_ = 0;
  bool waiting_ = false;
  bool locked_ = false;
};

#endif
