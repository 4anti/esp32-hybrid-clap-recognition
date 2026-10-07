#pragma once
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "clap_frontend_tables.h"

// No allocation; scratch belongs to the inference worker, never the I2S task.
// Periodic Hann400, hop160, FFT512 magnitude, HTK64 125..7500Hz, ln(x+.001).
// Input is unamplified signed PCM16 /32768; output layout is [mel][time].
class ClapFrontend {
 public:
  static constexpr int kSamples = 8000;
  static constexpr int kMels = 64;
  static constexpr int kFrames = 48;
  static constexpr int kTargetFirstFrame = 32;
  static constexpr int kTargetOffset = kTargetFirstFrame * 160;
  static constexpr int kTargetSamples = kSamples - kTargetOffset;

  void compute(const int16_t *pcm, float *features) {
    transform(pcm, features, nullptr, nullptr, nullptr, 1.0f, 0, 0);
  }

  bool computeQuantized(const int16_t *pcm, int8_t *features,
                        const float *mean, const float *std,
                        float scale, int zeroPoint) {
    if (!(scale > 0) || !isfinite(scale) || zeroPoint < -128 || zeroPoint > 127) return false;
    for (int mel = 0; mel < kMels; ++mel) {
      if (!isfinite(mean[mel]) || !(std[mel] > 0) || !isfinite(std[mel])) return false;
    }
    return transform(pcm, nullptr, features, mean, std, scale, zeroPoint, 0);
  }

  // Version-3 models crop frames 0..31 before any convolution. Store only the
  // 2,880 samples used by frames 32..47; fill ignored inputs with quantized zero.
  // The full frontend above remains the golden reference for this optimization.
  bool computeQuantizedTarget(const int16_t *pcm, int8_t *features,
                             const float *mean, const float *std,
                             float scale, int zeroPoint) {
    if (!(scale > 0) || !isfinite(scale) || zeroPoint < -128 || zeroPoint > 127) return false;
    for (int mel = 0; mel < kMels; ++mel) {
      if (!isfinite(mean[mel]) || !(std[mel] > 0) || !isfinite(std[mel])) return false;
    }
    memset(features, (unsigned char)(int8_t)zeroPoint, kMels * kFrames);
    return transform(pcm, nullptr, features, mean, std, scale, zeroPoint, kTargetFirstFrame);
  }

 private:
  float re_[512], im_[512], magnitude_[257];

  void fft() {
    unsigned j = 0;
    for (unsigned i = 1; i < 512; ++i) {
      unsigned bit = 256;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        float tmp = re_[i]; re_[i] = re_[j]; re_[j] = tmp;
        tmp = im_[i]; im_[i] = im_[j]; im_[j] = tmp;
      }
    }
    for (int length = 2; length <= 512; length *= 2) {
      for (int base = 0; base < 512; base += length) {
        for (int k = 0; k < length / 2; ++k) {
          const int twiddle = k * (512 / length);
          const float wRe = clap_frontend_tables::fftReal[twiddle];
          const float wIm = clap_frontend_tables::fftImag[twiddle];
          const int a = base + k, b = a + length / 2;
          const float bRe = re_[b] * wRe - im_[b] * wIm;
          const float bIm = re_[b] * wIm + im_[b] * wRe;
          re_[b] = re_[a] - bRe; im_[b] = im_[a] - bIm;
          re_[a] += bRe; im_[a] += bIm;
        }
      }
    }
    for (int bin = 0; bin <= 256; ++bin)
      magnitude_[bin] = sqrtf(re_[bin] * re_[bin] + im_[bin] * im_[bin]);
  }

  bool transform(const int16_t *pcm, float *floating, int8_t *quantized,
                 const float *mean, const float *std, float scale, int zeroPoint,
                 int firstFrame) {
    using namespace clap_frontend_tables;
    for (int frame = firstFrame; frame < kFrames; ++frame) {
      for (int i = 0; i < 512; ++i) {
        re_[i] = i < 400 ? (pcm[(frame - firstFrame) * 160 + i] / 32768.0f) * hann[i] : 0;
        im_[i] = 0;
      }
      fft();
      for (int mel = 0; mel < kMels; ++mel) {
        float energy = 0;
        const int offset = melOffset[mel];
        for (int k = offset; k < melOffset[mel + 1]; ++k)
          energy += magnitude_[melStart[mel] + k - offset] * melWeight[k];
        const float feature = logf(energy + 0.001f);
        const int index = mel * kFrames + frame;
        if (floating) floating[index] = feature;
        else {
          const float normalized = (feature - mean[mel]) / std[mel];
          if (!isfinite(normalized)) return false;
          const float value = roundf(normalized / scale) + zeroPoint;
          // Saturate before conversion: integer casts of huge floats are undefined.
          quantized[index] = value <= -128 ? -128 : value >= 127 ? 127 : (int8_t)value;
        }
      }
    }
    return true;
  }
};
