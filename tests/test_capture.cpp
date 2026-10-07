#include "../clap_double/clap_capture.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>

using SmallCapture = ClapCapture<8, 4, 3>;

static int16_t valueAt(uint32_t sample) {
  return static_cast<int16_t>(sample % 30000);
}

template <typename Capture>
static void feed(Capture &capture, uint32_t &next, unsigned count) {
  for (unsigned i = 0; i < count; ++i, ++next)
    capture.sample(valueAt(next), next);
}

static ClapCandidate metadata(uint32_t onset) {
  ClapCandidate candidate;
  candidate.onsetSample = onset;
  candidate.lastHotSample = onset + 1;
  candidate.activeSamples = 2;
  candidate.peakHp = 600000;
  candidate.plausible = true;
  return candidate;
}

static void testExact500msAlignment() {
  ClapCapture<> capture;
  uint32_t next = 0;
  feed(capture, next, 6900);
  assert(capture.begin(next, 50));
  assert(capture.candidate(metadata(next)));
  ClapCapture<>::Work work;
  feed(capture, next, 1599);
  assert(!capture.claim(work));
  feed(capture, next, 1);
  assert(capture.claim(work));
  assert(work.token.onsetSample == 6900);
  assert(work.token.createdTick == 50);
  for (unsigned i = 0; i < capture.kWindowSamples; ++i)
    assert(work.pcm[i] == valueAt(500 + i));
  assert(work.pcm[6400] == valueAt(work.token.onsetSample));
  assert(work.candidate.peakHp == 600000);
  assert(!capture.release(work.token)); // Worker is still reading.
  assert(capture.complete(work.token));
  assert(capture.release(work.token));
  assert(!capture.current(work.token));
}

static void testAudioAndMetadataCanFinishInEitherOrder() {
  SmallCapture capture;
  uint32_t next = 0;
  assert(!capture.begin(0));
  feed(capture, next, 8);
  assert(capture.begin(8));
  assert(!capture.begin(8)); // Duplicate onset does not occupy another slot.
  feed(capture, next, 4);
  SmallCapture::Work work;
  assert(!capture.claim(work)); // Complete audio still needs DSP metadata.
  assert(capture.candidate(metadata(8)));
  assert(capture.claim(work));
  assert(capture.complete(work.token));
  assert(capture.release(work.token));
  assert(capture.begin(12));
  assert(capture.candidate(metadata(12)));
  assert(!capture.claim(work)); // Complete metadata still needs post-onset audio.
  feed(capture, next, 4);
  assert(capture.claim(work));
  for (unsigned i = 0; i < capture.kWindowSamples; ++i)
    assert(work.pcm[i] == valueAt(4 + i));
}

static void testFifoWaitsForEarlierMetadata() {
  SmallCapture capture;
  uint32_t next = 0;
  feed(capture, next, 8);
  assert(capture.begin(8));
  feed(capture, next, 4);
  assert(capture.begin(12));
  assert(capture.candidate(metadata(12)));
  feed(capture, next, 4);
  SmallCapture::Work first, second;
  assert(!capture.claim(first)); // Later ready window cannot overtake onset 8.
  assert(capture.candidate(metadata(8)));
  assert(capture.claim(first) && first.token.onsetSample == 8);
  assert(capture.complete(first.token));
  assert(capture.claim(second) && second.token.onsetSample == 12);
  assert(first.token.sequence < second.token.sequence);
  assert(capture.complete(second.token));
  assert(capture.release(first.token));
  assert(capture.release(second.token));
}

static void testOwnershipOverflowAndOldToken() {
  SmallCapture capture;
  uint32_t next = 0;
  feed(capture, next, 8);
  SmallCapture::Work work[3];
  std::vector<int16_t> snapshots[3];
  for (unsigned i = 0; i < 3; ++i) {
    const uint32_t onset = next;
    assert(capture.begin(onset));
    assert(capture.candidate(metadata(onset)));
    feed(capture, next, 4);
    assert(capture.claim(work[i]));
    snapshots[i].assign(work[i].pcm, work[i].pcm + capture.kWindowSamples);
  }
  assert(!capture.begin(next));
  assert(capture.dropped() == 1);
  feed(capture, next, 1000);
  for (unsigned i = 0; i < 3; ++i)
    for (unsigned j = 0; j < capture.kWindowSamples; ++j)
      assert(work[i].pcm[j] == snapshots[i][j]);
  assert(capture.complete(work[0].token));
  assert(capture.release(work[0].token));
  const uint32_t newOnset = next;
  assert(capture.begin(newOnset));
  assert(capture.candidate(metadata(newOnset)));
  feed(capture, next, 4);
  SmallCapture::Work replacement;
  assert(capture.claim(replacement));
  assert(replacement.token.slot == work[0].token.slot);
  assert(!capture.current(work[0].token));
  assert(!capture.release(work[0].token)); // Old result cannot free reused PCM.
  assert(capture.current(replacement.token));
}

static void testResetAndTimeoutPreserveWorkerPcm() {
  SmallCapture capture;
  uint32_t next = 0;
  feed(capture, next, 8);
  assert(capture.begin(8, UINT32_MAX - 500));
  assert(capture.candidate(metadata(8)));
  feed(capture, next, 4);
  SmallCapture::Work work;
  assert(capture.claim(work));
  const std::vector<int16_t> original(work.pcm, work.pcm + capture.kWindowSamples);
  assert(capture.expire(498, 1000) == 0);
  assert(capture.expire(499, 1000) == 1); // Tick wrapping, exactly one second.
  assert(capture.expire(500, 1000) == 0); // Timeout counted once.
  assert(!capture.current(work.token));
  uint32_t oldest;
  assert(!capture.pendingOnset(oldest));
  const uint32_t generation = capture.generation();
  capture.reset();
  assert(capture.generation() == generation + 1);
  feed(capture, next, 1000);
  for (unsigned i = 0; i < capture.kWindowSamples; ++i)
    assert(work.pcm[i] == original[i]);
  assert(capture.complete(work.token));
  assert(capture.release(work.token));
  assert(capture.begin(next)); // History warmed while old immutable work survived.
}

static void testDecisionBarrierAndAudioDiscontinuity() {
  SmallCapture capture;
  uint32_t next = UINT32_MAX - 10;
  feed(capture, next, 8);
  assert(capture.begin(next));
  const uint32_t onset = next;
  assert(capture.candidate(metadata(onset)));
  feed(capture, next, 4); // Microphone sample clock wraps inside window.
  SmallCapture::Work work;
  assert(capture.claim(work));
  assert(work.token.onsetSample == onset && capture.current(work.token));
  capture.invalidatePending();
  assert(!capture.current(work.token));
  assert(capture.begin(next)); // Decision barrier keeps uninterrupted history.
  const uint32_t generation = capture.generation();
  capture.sample(0, next + 10); // A missing chunk invalidates pending work.
  assert(capture.generation() == generation + 1);
  assert(!capture.begin(next + 11)); // New stream needs complete prehistory.
  assert(capture.complete(work.token));
  assert(capture.release(work.token));
}

static void testConcurrentProducerAndConsumer() {
  using Capture = ClapCapture<32, 16, 3>;
  Capture capture;
  std::atomic<bool> producerDone{false};
  std::atomic<unsigned> completed{0};
  std::mutex resultsLock;
  std::queue<Capture::Token> results;
  std::thread consumer([&]() {
    uint32_t previousOnset = 0;
    bool havePrevious = false;
    for (;;) {
      Capture::Work work;
      if (!capture.claim(work)) {
        if (producerDone.load(std::memory_order_acquire)) {
          uint32_t pending;
          if (!capture.pendingOnset(pending)) break;
        }
        std::this_thread::yield();
        continue;
      }
      if (havePrevious) assert(work.token.onsetSample > previousOnset);
      previousOnset = work.token.onsetSample;
      havePrevious = true;
      int16_t snapshot[Capture::kWindowSamples];
      memcpy(snapshot, work.pcm, sizeof(snapshot));
      for (unsigned i = 0; i < 20; ++i) std::this_thread::yield();
      assert(memcmp(snapshot, work.pcm, sizeof(snapshot)) == 0);
      assert(capture.complete(work.token));
      {
        std::lock_guard<std::mutex> lock(resultsLock);
        results.push(work.token);
      }
      completed.fetch_add(1, std::memory_order_relaxed);
    }
  });
  for (uint32_t sample = 0; sample < 20000; ++sample) {
    if (sample >= 64 && sample % 100 == 64) capture.begin(sample);
    capture.sample(valueAt(sample), sample);
    if (sample >= 79 && sample % 100 == 79) capture.candidate(metadata(sample - 15));
    {
      std::lock_guard<std::mutex> lock(resultsLock);
      while (!results.empty()) {
        assert(capture.release(results.front()));
        results.pop();
      }
    }
    if (sample % 100 == 0) std::this_thread::yield();
  }
  // COMPLETE slots awaiting producer release remain visible as pending; drain
  // results while the consumer processes its last ready work.
  producerDone.store(true, std::memory_order_release);
  for (;;) {
    {
      std::lock_guard<std::mutex> lock(resultsLock);
      while (!results.empty()) {
        assert(capture.release(results.front()));
        results.pop();
      }
    }
    uint32_t pending;
    if (!capture.pendingOnset(pending)) break;
    std::this_thread::yield();
  }
  consumer.join();
  assert(completed.load(std::memory_order_relaxed) > 0);
}

int main() {
  testExact500msAlignment();
  testAudioAndMetadataCanFinishInEitherOrder();
  testFifoWaitsForEarlierMetadata();
  testOwnershipOverflowAndOldToken();
  testResetAndTimeoutPreserveWorkerPcm();
  testDecisionBarrierAndAudioDiscontinuity();
  testConcurrentProducerAndConsumer();
  printf("capture: all alignment, FIFO, ownership, overload, reset, timeout and concurrency tests passed; storage=%lu bytes\n",
         static_cast<unsigned long>(sizeof(ClapCapture<>)));
}
