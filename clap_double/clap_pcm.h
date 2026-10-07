#pragma once
#include <stdint.h>

// Protect calls with the ESP32 cross-core spinlock. Absolute sample positions
// let recordings detect drops; overflow discards old playback, never I2S input.
class ClapPcmBuffer {
 public:
  static constexpr unsigned kCapacity = 4096;

  void push(const int32_t *raw24, unsigned count, uint64_t firstSample) {
    if (!count) return;
    if (!havePosition_ || firstSample != endSample_) {
      dropped_ += count_;
      count_ = 0;
      read_ = write_;
      firstSample_ = firstSample;
      havePosition_ = true;
    }
    for (unsigned i = 0; i < count; ++i) {
      if (count_ == kCapacity) {
        read_ = (read_ + 1) & (kCapacity - 1);
        --count_; ++firstSample_; ++dropped_;
      }
      pcm_[write_] = (int16_t)(raw24[i] >> 8);
      write_ = (write_ + 1) & (kCapacity - 1);
      ++count_;
    }
    endSample_ = firstSample + count;
  }

  unsigned pop(int16_t *out, unsigned maxCount, uint64_t &firstSample, uint32_t &dropped) {
    firstSample = firstSample_;
    dropped = dropped_;
    const unsigned count = count_ < maxCount ? count_ : maxCount;
    for (unsigned i = 0; i < count; ++i) {
      out[i] = pcm_[read_];
      read_ = (read_ + 1) & (kCapacity - 1);
    }
    count_ -= count;
    firstSample_ += count;
    return count;
  }

  uint32_t dropped() const { return dropped_; }
  unsigned available() const { return count_; }

 private:
  int16_t pcm_[kCapacity];
  unsigned read_ = 0, write_ = 0, count_ = 0;
  uint64_t firstSample_ = 0, endSample_ = 0;
  uint32_t dropped_ = 0;
  bool havePosition_ = false;
};
