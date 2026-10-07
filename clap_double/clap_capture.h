#ifndef CLAP_CAPTURE_H
#define CLAP_CAPTURE_H

#include <atomic>
#include <stdint.h>
#include <string.h>
#include "clap_detector.h"

// One audio producer, one inference consumer. Publication uses acquire/release
// atomics; inference owns immutable PCM until the producer releases its result.
// No operating-system dependency, allocation, waiting, or audio-reader pauses.
template <unsigned PreSamples = 6400, unsigned PostSamples = 1600,
          unsigned SlotCount = 3>
class ClapCapture {
 public:
  static_assert(PreSamples > 0 && PostSamples > 0 && SlotCount > 0,
                "capture dimensions must be positive");
  static constexpr unsigned kWindowSamples = PreSamples + PostSamples;

  struct Token {
    uint32_t generation = 0;
    uint32_t sequence = 0;
    uint32_t onsetSample = 0;
    uint32_t createdTick = 0;
    unsigned slot = SlotCount;
  };

  struct Work {
    Token token;
    const int16_t *pcm = nullptr;
    ClapCandidate candidate;
  };

  uint32_t dropped() const { return dropped_.load(std::memory_order_relaxed); }
  uint32_t generation() const { return generation_.load(std::memory_order_acquire); }

  // Call BEFORE sample() for the onset sample. The history then contains exactly
  // the preceding PreSamples; the onset is the first post-onset sample.
  bool begin(uint32_t onsetSample, uint32_t createdTick = 0) {
    if (!haveClock_ || historyCount_ != PreSamples || onsetSample != nextSample_) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    for (unsigned i = 0; i < SlotCount; ++i)
      if (slots_[i].state.load(std::memory_order_acquire) != Free &&
          slots_[i].generation.load(std::memory_order_relaxed) == generation() &&
          slots_[i].onset.load(std::memory_order_relaxed) == onsetSample) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
    for (unsigned i = 0; i < SlotCount; ++i) {
      uint32_t expected = Free;
      if (!slots_[i].state.compare_exchange_strong(expected, Reserved,
              std::memory_order_acquire, std::memory_order_relaxed)) continue;
      Slot &slot = slots_[i];
      slot.generation.store(generation(), std::memory_order_relaxed);
      slot.sequence.store(nextSequence_++, std::memory_order_relaxed);
      slot.onset.store(onsetSample, std::memory_order_relaxed);
      slot.tick.store(createdTick, std::memory_order_relaxed);
      slot.retired.store(false, std::memory_order_relaxed);
      slot.filled = PreSamples;
      slot.hasCandidate = false;
      const unsigned first = PreSamples - historyWrite_;
      memcpy(slot.pcm, history_ + historyWrite_, first * sizeof(int16_t));
      memcpy(slot.pcm + first, history_, historyWrite_ * sizeof(int16_t));
      slot.state.store(Filling, std::memory_order_release);
      return true;
    }
    dropped_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  void sample(int16_t pcm, uint32_t sampleIndex) {
    if (haveClock_ && sampleIndex != nextSample_) reset();
    haveClock_ = true;
    nextSample_ = sampleIndex + 1;
    history_[historyWrite_] = pcm;
    historyWrite_ = (historyWrite_ + 1) % PreSamples;
    if (historyCount_ < PreSamples) ++historyCount_;
    for (unsigned i = 0; i < SlotCount; ++i) {
      Slot &slot = slots_[i];
      if (slot.state.load(std::memory_order_acquire) != Filling) continue;
      if (slot.filled < kWindowSamples &&
          sampleIndex - slot.onset.load(std::memory_order_relaxed) < PostSamples)
        slot.pcm[slot.filled++] = pcm;
      if (slot.filled == kWindowSamples && slot.hasCandidate)
        slot.state.store(Ready, std::memory_order_release);
    }
  }

  // DSP metadata may finish before or after the 100 ms post-onset audio.
  bool candidate(const ClapCandidate &candidate) {
    for (unsigned i = 0; i < SlotCount; ++i) {
      Slot &slot = slots_[i];
      if (slot.state.load(std::memory_order_acquire) != Filling ||
          slot.generation.load(std::memory_order_relaxed) != generation() ||
          slot.onset.load(std::memory_order_relaxed) != candidate.onsetSample) continue;
      slot.candidate = candidate;
      slot.hasCandidate = true;
      if (slot.filled == kWindowSamples)
        slot.state.store(Ready, std::memory_order_release);
      return true;
    }
    return false;
  }

  // Do not bypass an earlier capture whose DSP metadata/audio is unfinished.
  bool claim(Work &work) {
    unsigned oldest = SlotCount;
    uint32_t oldestSequence = 0;
    for (unsigned i = 0; i < SlotCount; ++i) {
      const uint32_t state = slots_[i].state.load(std::memory_order_acquire);
      if (state != Filling && state != Ready) continue;
      const uint32_t sequence = slots_[i].sequence.load(std::memory_order_relaxed);
      if (oldest == SlotCount || earlier(sequence, oldestSequence)) {
        oldest = i;
        oldestSequence = sequence;
      }
    }
    if (oldest == SlotCount) return false;
    Slot &slot = slots_[oldest];
    uint32_t expected = Ready;
    if (!slot.state.compare_exchange_strong(expected, Processing,
            std::memory_order_acquire, std::memory_order_relaxed)) return false;
    work.token = tokenFor(oldest);
    work.pcm = slot.pcm;
    work.candidate = slot.candidate;
    return true;
  }

  // Consumer signals it no longer reads PCM; the producer releases after
  // consuming the queued result. Old generations still need ownership release.
  bool complete(const Token &token) {
    if (!owns(token)) return false;
    uint32_t expected = Processing;
    return slots_[token.slot].state.compare_exchange_strong(expected, Complete,
        std::memory_order_release, std::memory_order_relaxed);
  }

  bool release(const Token &token) {
    if (!owns(token)) return false;
    uint32_t expected = Complete;
    return slots_[token.slot].state.compare_exchange_strong(expected, Free,
        std::memory_order_release, std::memory_order_relaxed);
  }

  bool current(const Token &token) const {
    return token.generation == generation() && owns(token) &&
        (slots_[token.slot].state.load(std::memory_order_acquire) == Processing ||
         slots_[token.slot].state.load(std::memory_order_acquire) == Complete) &&
        !slots_[token.slot].retired.load(std::memory_order_acquire);
  }

  bool pendingOnset(uint32_t &onset) const {
    unsigned oldest = SlotCount;
    uint32_t oldestSequence = 0;
    for (unsigned i = 0; i < SlotCount; ++i) {
      const Slot &slot = slots_[i];
      const uint32_t state = slot.state.load(std::memory_order_acquire);
      if (state == Free || state == Reserved ||
          slot.generation.load(std::memory_order_relaxed) != generation() ||
          slot.retired.load(std::memory_order_acquire)) continue;
      const uint32_t sequence = slot.sequence.load(std::memory_order_relaxed);
      if (oldest == SlotCount || earlier(sequence, oldestSequence)) {
        oldest = i;
        oldestSequence = sequence;
      }
    }
    if (oldest == SlotCount) return false;
    onset = slots_[oldest].onset.load(std::memory_order_relaxed);
    return true;
  }

  // Producer retires stale work but never reclaims a buffer being read. The
  // consumer/result owner must complete/release it even after a timeout/reset.
  uint32_t expire(uint32_t nowTick, uint32_t maximumAge) {
    uint32_t expired = 0;
    for (unsigned i = 0; i < SlotCount; ++i) {
      Slot &slot = slots_[i];
      uint32_t state = slot.state.load(std::memory_order_acquire);
      if (state == Free || state == Reserved ||
          nowTick - slot.tick.load(std::memory_order_relaxed) < maximumAge) continue;
      if (slot.retired.exchange(true, std::memory_order_acq_rel)) continue;
      ++expired;
      dropped_.fetch_add(1, std::memory_order_relaxed);
      if (state == Filling || state == Ready)
        slot.state.compare_exchange_strong(state, Free,
            std::memory_order_release, std::memory_order_relaxed);
    }
    return expired;
  }

  // An unclassified overload is a decision barrier. Cancel earlier work while
  // retaining uninterrupted history, so late positives cannot cross that gap.
  void invalidatePending() {
    generation_.fetch_add(1, std::memory_order_acq_rel);
    for (unsigned i = 0; i < SlotCount; ++i) {
      Slot &slot = slots_[i];
      uint32_t state = slot.state.load(std::memory_order_acquire);
      if (state == Free || state == Reserved) continue;
      if (!slot.retired.exchange(true, std::memory_order_acq_rel))
        dropped_.fetch_add(1, std::memory_order_relaxed);
      if (state == Filling || state == Ready)
        slot.state.compare_exchange_strong(state, Free,
            std::memory_order_release, std::memory_order_relaxed);
    }
  }

  void reset() {
    invalidatePending();
    haveClock_ = false;
    historyWrite_ = 0;
    historyCount_ = 0;
  }

 private:
  enum State : uint8_t { Free, Reserved, Filling, Ready, Processing, Complete };
  struct Slot {
    // ESP32 has native 32-bit compare-and-swap; avoid narrow atomic helpers.
    std::atomic<uint32_t> state{Free}, retired{0};
    std::atomic<uint32_t> generation{0}, sequence{0}, onset{0}, tick{0};
    unsigned filled = 0;
    bool hasCandidate = false;
    ClapCandidate candidate;
    int16_t pcm[kWindowSamples];
  };

  static bool earlier(uint32_t first, uint32_t second) {
    return static_cast<int32_t>(first - second) < 0;
  }

  bool owns(const Token &token) const {
    return token.slot < SlotCount &&
        slots_[token.slot].generation.load(std::memory_order_relaxed) == token.generation &&
        slots_[token.slot].sequence.load(std::memory_order_relaxed) == token.sequence;
  }

  Token tokenFor(unsigned index) const {
    Token token;
    token.slot = index;
    token.generation = slots_[index].generation.load(std::memory_order_relaxed);
    token.sequence = slots_[index].sequence.load(std::memory_order_relaxed);
    token.onsetSample = slots_[index].onset.load(std::memory_order_relaxed);
    token.createdTick = slots_[index].tick.load(std::memory_order_relaxed);
    return token;
  }

  int16_t history_[PreSamples];
  Slot slots_[SlotCount];
  unsigned historyWrite_ = 0, historyCount_ = 0;
  uint32_t nextSample_ = 0, nextSequence_ = 0;
  bool haveClock_ = false;
  std::atomic<uint32_t> generation_{1}, dropped_{0};
};

#endif
