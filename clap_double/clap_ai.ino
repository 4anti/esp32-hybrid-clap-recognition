#include "clap_config.h"

#if CLAP_ENABLE_AI
#include <atomic>
#include <new>
#include <math.h>
#include "esp_heap_caps.h"
#include "clap_capture.h"
#include "clap_frontend.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

using AiCapture = ClapCapture<CLAP_AI_PRE_SAMPLES, CLAP_AI_POST_SAMPLES, CLAP_AI_SLOTS>;

static_assert(AiCapture::kWindowSamples == ClapFrontend::kTargetSamples,
              "capture must cover the model target region");
static_assert(CLAP_AI_WINDOW_SAMPLES == ClapFrontend::kSamples,
              "frontend and model window sizes must agree");

struct ClapInferenceResult {
  AiCapture::Token token;
  ClapCandidate candidate;
  float scores[3];
  uint32_t frontendMs;
  uint32_t inferenceMs;
  bool ok;
  bool accepted;
};

static AiCapture *aiCapture = nullptr;
static ClapFrontend *aiFrontend = nullptr;
static tflite::MicroMutableOpResolver<8> *aiResolver = nullptr;
static tflite::MicroInterpreter *aiInterpreter = nullptr;
static uint8_t *aiArena = nullptr;
static size_t aiArenaSize = 0;
static QueueHandle_t aiResultQueue = nullptr;
static TaskHandle_t aiWorkerTask = nullptr;
static std::atomic<bool> aiReady{false};
static std::atomic<uint32_t> aiErrorCount{0}, aiUnavailableDrops{0}, aiInferenceTime{0};
static const uint32_t AI_MAX_AGE_MS = 1000;
static const size_t AI_HEADROOM_BYTES = 32 * 1024;
static const uint32_t AI_WORKER_STACK_BYTES = 8192;
static const uint32_t AI_MEMORY_CAPS = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;

static void aiReleaseInterpreter() {
  if (aiInterpreter) {
    aiInterpreter->~MicroInterpreter();
    heap_caps_free(aiInterpreter);
    aiInterpreter = nullptr;
  }
  if (aiArena) {
    heap_caps_free(aiArena);
    aiArena = nullptr;
  }
  aiArenaSize = 0;
}

static void aiReleaseMemory() {
  aiReleaseInterpreter();
  if (aiResolver) {
    aiResolver->~MicroMutableOpResolver();
    heap_caps_free(aiResolver);
    aiResolver = nullptr;
  }
  if (aiFrontend) {
    aiFrontend->~ClapFrontend();
    heap_caps_free(aiFrontend);
    aiFrontend = nullptr;
  }
  if (aiCapture) {
    aiCapture->~ClapCapture();
    heap_caps_free(aiCapture);
    aiCapture = nullptr;
  }
  if (aiResultQueue) {
    vQueueDelete(aiResultQueue);
    aiResultQueue = nullptr;
  }
}

static bool aiAllocateArena() {
  const size_t sizes[] = {32 * 1024, 48 * 1024, 64 * 1024, CLAP_AI_ARENA_BYTES};
  const tflite::Model *model = tflite::GetModel(clap_model::data);
  size_t previousSize = 0;
  for (size_t size : sizes) {
    if (size > CLAP_AI_ARENA_BYTES || size == previousSize) continue;
    previousSize = size;
    const size_t reserve = AI_HEADROOM_BYTES + AI_WORKER_STACK_BYTES +
        sizeof(tflite::MicroInterpreter) + 1024;
    if (heap_caps_get_free_size(AI_MEMORY_CAPS) < size + reserve ||
        heap_caps_get_largest_free_block(AI_MEMORY_CAPS) < size) continue;
    aiArena = static_cast<uint8_t *>(heap_caps_aligned_alloc(16, size, AI_MEMORY_CAPS));
    void *storage = heap_caps_malloc(sizeof(tflite::MicroInterpreter), AI_MEMORY_CAPS);
    if (!aiArena || !storage) {
      if (storage) heap_caps_free(storage);
      aiReleaseInterpreter();
      continue;
    }
    aiInterpreter = new (storage) tflite::MicroInterpreter(
        model, *aiResolver, aiArena, size);
    if (aiInterpreter->AllocateTensors() == kTfLiteOk) {
      aiArenaSize = size;
      return true;
    }
    elog("AI arena %u bytes insufficient", (unsigned)size);
    aiReleaseInterpreter();
  }
  return false;
}

static bool aiValidateTensors() {
  if (aiInterpreter->inputs_size() != 1 || aiInterpreter->outputs_size() != 1)
    return false;
  TfLiteTensor *input = aiInterpreter->input(0);
  TfLiteTensor *output = aiInterpreter->output(0);
  if (!input || !output || input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
      !input->dims || input->dims->size != 4 || input->dims->data[0] != 1 ||
      input->dims->data[1] != 64 || input->dims->data[2] != 48 || input->dims->data[3] != 1 ||
      input->bytes != 64 * 48 || output->bytes != 3 ||
      !output->dims || output->dims->size != 2 || output->dims->data[0] != 1 ||
      output->dims->data[1] != 3 || !(input->params.scale > 0) ||
      !(output->params.scale > 0) || !isfinite(input->params.scale) ||
      !isfinite(output->params.scale)) return false;
  for (unsigned i = 0; i < 64; ++i)
    if (!isfinite(clap_model::mean[i]) || !(clap_model::std[i] > 0) ||
        !isfinite(clap_model::std[i])) return false;
  return true;
}

static bool aiReadScores(float *scores) {
  const TfLiteTensor *output = aiInterpreter->output(0);
  bool ok = true;
  float sum = 0;
  for (unsigned i = 0; i < 3; ++i) {
    scores[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
    if (!isfinite(scores[i]) || scores[i] < 0 || scores[i] > 1.01f) ok = false;
    sum += scores[i];
  }
  return ok && fabsf(sum - 1.0f) <= 0.02f;
}

static void aiSelfTest() {
  // Let setup finish and the higher-priority audio task begin servicing DMA.
  // This worker alone owns the frontend/interpreter, so the diagnostic uses a
  // temporary buffer without borrowing capture slots or generating decisions.
  vTaskDelay(pdMS_TO_TICKS(100));
  const size_t bytes = AiCapture::kWindowSamples * sizeof(int16_t);
  const size_t reserve = 24 * 1024;
  if (heap_caps_get_free_size(AI_MEMORY_CAPS) < bytes + reserve ||
      heap_caps_get_largest_free_block(AI_MEMORY_CAPS) < bytes) {
    elog("AI selftest skipped: internal heap=%u", (unsigned)heap_caps_get_free_size(AI_MEMORY_CAPS));
    return;
  }
  int16_t *pcm = static_cast<int16_t *>(heap_caps_malloc(bytes, AI_MEMORY_CAPS));
  if (!pcm || heap_caps_get_free_size(AI_MEMORY_CAPS) < reserve) {
    if (pcm) heap_caps_free(pcm);
    elog("AI selftest skipped: temporary PCM allocation");
    return;
  }
  memset(pcm, 0, bytes);
  for (unsigned test = 0; test < 2; ++test) {
    wdtFeed();
    if (test) {
      pcm[CLAP_AI_PRE_SAMPLES] = 24000;
      pcm[CLAP_AI_PRE_SAMPLES + 1] = -24000;
    }
    TfLiteTensor *input = aiInterpreter->input(0);
    const uint32_t frontendStart = micros();
    bool ok = aiFrontend->computeQuantizedTarget(pcm, input->data.int8,
        clap_model::mean, clap_model::std, input->params.scale, input->params.zero_point);
    const uint32_t frontendMs = (micros() - frontendStart + 999) / 1000;
    uint32_t inferenceMs = 0;
    wdtFeed();
    if (ok) {
      const uint32_t inferenceStart = micros();
      ok = aiInterpreter->Invoke() == kTfLiteOk;
      inferenceMs = (micros() - inferenceStart + 999) / 1000;
      aiInferenceTime.store(inferenceMs, std::memory_order_relaxed);
    }
    float scores[3] = {0, 0, 0};
    if (ok) ok = aiReadScores(scores);
    if (!ok) aiErrorCount.fetch_add(1, std::memory_order_relaxed);
    elog("AI selftest %s ok=%d front=%u infer=%u scores=%.3f/%.3f/%.3f heap=%u",
         test ? "impulse" : "silence", ok ? 1 : 0,
         (unsigned)frontendMs, (unsigned)inferenceMs,
         scores[0], scores[1], scores[2], (unsigned)heap_caps_get_free_size(AI_MEMORY_CAPS));
    wdtFeed();
    vTaskDelay(pdMS_TO_TICKS(2));
  }
  heap_caps_free(pcm);
}

static void aiWorker(void *arg) {
  (void)arg;
  esp_task_wdt_add(NULL);
  while (!aiReady.load(std::memory_order_acquire)) {
    wdtFeed();
    vTaskDelay(pdMS_TO_TICKS(2));
  }
  aiSelfTest();
  for (;;) {
    wdtFeed();
    AiCapture::Work work;
    if (!aiCapture->claim(work)) {
      vTaskDelay(pdMS_TO_TICKS(2));
      continue;
    }
    ClapInferenceResult result{};
    result.token = work.token;
    result.candidate = work.candidate;
    result.ok = aiCapture->current(work.token);
    TfLiteTensor *input = aiInterpreter->input(0);
    const uint32_t frontendStart = micros();
    if (result.ok) result.ok = aiFrontend->computeQuantizedTarget(
        work.pcm, input->data.int8, clap_model::mean, clap_model::std,
        input->params.scale, input->params.zero_point);
    result.frontendMs = (micros() - frontendStart + 999) / 1000;
    if (result.ok) {
      const uint32_t inferenceStart = micros();
      result.ok = aiInterpreter->Invoke() == kTfLiteOk;
      result.inferenceMs = (micros() - inferenceStart + 999) / 1000;
      aiInferenceTime.store(result.inferenceMs, std::memory_order_relaxed);
    }
    if (result.ok) {
      result.ok = aiReadScores(result.scores);
      // Clap and snap are both valid gestures; their combined posterior avoids
      // rejecting a clear hand transient solely for ambiguity between them.
      const float target = 1.0f - result.scores[0];
      const float threshold = fmaxf(CLAP_AI_MIN_TARGET, clap_model::positive_threshold);
      const bool gestureWins = result.scores[1] > result.scores[0] ||
          result.scores[2] > result.scores[0];
      result.accepted = result.ok && !work.candidate.timedOut && gestureWins && target >= threshold &&
          result.scores[0] <= CLAP_AI_MAX_NOISE &&
          target - result.scores[0] >= CLAP_AI_MIN_MARGIN;
    }
    if (!result.ok && aiCapture->current(work.token))
      aiErrorCount.fetch_add(1, std::memory_order_relaxed);
    if (!aiCapture->complete(work.token))
      aiErrorCount.fetch_add(1, std::memory_order_relaxed);
    // Queue capacity equals slot count, and each owned slot creates one result.
    // Only this worker can wait here; capture never waits for inference/network.
    while (xQueueSend(aiResultQueue, &result, pdMS_TO_TICKS(20)) != pdTRUE) wdtFeed();
  }
}

bool aiSetup() {
  if (aiReady.load(std::memory_order_acquire)) return true;
  if (clap_model::data_len < 8 || clap_model::class_count != 3 || clap_model::model_version != 3 ||
      clap_model::target_frame_start != ClapFrontend::kTargetFirstFrame ||
      strcmp(clap_model::frontend_version, "logmag_htk64_16k_500ms_v1") != 0 ||
      strcmp(clap_model::labels[0], "noise") != 0 || strcmp(clap_model::labels[1], "clap") != 0 ||
      strcmp(clap_model::labels[2], "finger_snap") != 0 ||
      !isfinite(clap_model::positive_threshold) || clap_model::positive_threshold < 0 ||
      clap_model::positive_threshold > 1 ||
      tflite::GetModel(clap_model::data)->version() != TFLITE_SCHEMA_VERSION) {
    aiErrorCount.fetch_add(1, std::memory_order_relaxed);
    elog("AI model schema/class mismatch");
    return false;
  }
  void *captureStorage = heap_caps_malloc(sizeof(AiCapture), AI_MEMORY_CAPS);
  void *frontendStorage = heap_caps_malloc(sizeof(ClapFrontend), AI_MEMORY_CAPS);
  void *resolverStorage = heap_caps_malloc(sizeof(tflite::MicroMutableOpResolver<8>), AI_MEMORY_CAPS);
  if (captureStorage) aiCapture = new (captureStorage) AiCapture();
  if (frontendStorage) aiFrontend = new (frontendStorage) ClapFrontend();
  if (resolverStorage) aiResolver = new (resolverStorage) tflite::MicroMutableOpResolver<8>();
  aiResultQueue = xQueueCreate(CLAP_AI_SLOTS, sizeof(ClapInferenceResult));
  bool ok = aiCapture && aiFrontend && aiResolver && aiResultQueue;
  if (ok) ok = aiResolver->AddConv2D() == kTfLiteOk &&
      aiResolver->AddMaxPool2D() == kTfLiteOk && aiResolver->AddAveragePool2D() == kTfLiteOk &&
      aiResolver->AddReshape() == kTfLiteOk && aiResolver->AddConcatenation() == kTfLiteOk &&
      aiResolver->AddFullyConnected() == kTfLiteOk && aiResolver->AddSoftmax() == kTfLiteOk &&
      aiResolver->AddStridedSlice() == kTfLiteOk;
  if (ok) ok = aiAllocateArena() && aiValidateTensors();
  if (ok) ok = heap_caps_get_free_size(AI_MEMORY_CAPS) >=
      AI_HEADROOM_BYTES + AI_WORKER_STACK_BYTES + 1024;
  if (ok) ok = xTaskCreatePinnedToCore(aiWorker, "clap_ai", AI_WORKER_STACK_BYTES,
      NULL, 2, &aiWorkerTask, 1) == pdPASS;
  if (!ok) {
    aiErrorCount.fetch_add(1, std::memory_order_relaxed);
    aiReleaseMemory();
    elog("AI unavailable: internal heap=%u largest=%u", (unsigned)heap_caps_get_free_size(AI_MEMORY_CAPS),
         (unsigned)heap_caps_get_largest_free_block(AI_MEMORY_CAPS));
    return false;
  }
  aiReady.store(true, std::memory_order_release);
  elog("AI model=%s threshold=%.6f", clap_model::sha256, clap_model::positive_threshold);
  elog("AI %s: capture=%u arena=%u used=%u internal heap=%u", CLAP_AI_SHADOW ? "shadow" : "active",
       (unsigned)sizeof(AiCapture), (unsigned)aiArenaSize,
       (unsigned)aiInterpreter->arena_used_bytes(), (unsigned)heap_caps_get_free_size(AI_MEMORY_CAPS));
  return true;
}

void aiBegin(uint32_t onsetSample) {
  if (!aiReady.load(std::memory_order_acquire)) {
    aiUnavailableDrops.fetch_add(1, std::memory_order_relaxed);
    if (!CLAP_AI_SHADOW) clapPairer.reject();
    return;
  }
  if (!aiCapture->begin(onsetSample, millis()) && !CLAP_AI_SHADOW) {
    aiCapture->invalidatePending();
    clapPairer.reject();
  }
}

void aiSample(int32_t raw24, uint32_t sampleIndex) {
  if (!aiReady.load(std::memory_order_acquire)) return;
  int32_t pcm = raw24 >> 8;
  if (pcm > 32767) pcm = 32767;
  if (pcm < -32768) pcm = -32768;
  aiCapture->sample(static_cast<int16_t>(pcm), sampleIndex);
}

void aiCandidate(const ClapCandidate &candidate) {
  if ((!aiReady.load(std::memory_order_acquire) || !aiCapture->candidate(candidate)) &&
      !CLAP_AI_SHADOW) clapPairer.reject();
}

void aiPoll() {
  if (!aiReady.load(std::memory_order_acquire)) return;
  if (aiCapture->expire(millis(), AI_MAX_AGE_MS) && !CLAP_AI_SHADOW) clapPairer.reject();
  ClapInferenceResult result;
  while (xQueueReceive(aiResultQueue, &result, 0) == pdTRUE) {
    const bool current = aiCapture->current(result.token);
    const bool accepted = current && result.ok && result.accepted;
    const char *mode = !current ? (CLAP_AI_SHADOW ? "shadow_stale" : "active_stale") :
        !result.ok ? (CLAP_AI_SHADOW ? "shadow_error" : "active_error") :
        (CLAP_AI_SHADOW ? "shadow" : "active");
    labAiEvent(result.candidate, result.scores, result.frontendMs, result.inferenceMs, accepted, mode);
    if (!CLAP_AI_SHADOW && current) {
      // With limited room data, AI adds a veto to the working DSP policy.
      // It must not remove existing slam, echo and first-impulse protections.
      if (accepted) acceptCandidate(result.candidate, false);
      else rejectCandidate(result.candidate);
    }
    if (!aiCapture->release(result.token)) aiErrorCount.fetch_add(1, std::memory_order_relaxed);
  }
}

void aiReset() {
  if (aiCapture) aiCapture->reset();
  if (!CLAP_AI_SHADOW) clapPairer.reject();
}

bool aiPendingOnset(uint32_t &onset) {
  return aiReady.load(std::memory_order_acquire) && aiCapture->pendingOnset(onset);
}

const char *aiMode() {
  if (!aiReady.load(std::memory_order_acquire))
    return CLAP_AI_SHADOW ? "shadow_unavailable" : "active_unavailable";
  return CLAP_AI_SHADOW ? "shadow" : "active";
}

uint32_t aiDropped() {
  return aiUnavailableDrops.load(std::memory_order_relaxed) +
      (aiReady.load(std::memory_order_acquire) ? aiCapture->dropped() : 0);
}
uint32_t aiErrors() { return aiErrorCount.load(std::memory_order_relaxed); }
uint32_t aiLastInferenceMs() { return aiInferenceTime.load(std::memory_order_relaxed); }

#else

// With AI disabled these functions allocate no history, slots, model or arena.
bool aiSetup() { return true; }
void aiBegin(uint32_t onsetSample) { (void)onsetSample; }
void aiSample(int32_t sample, uint32_t sampleIndex) { (void)sample; (void)sampleIndex; }
void aiCandidate(const ClapCandidate &candidate) { (void)candidate; }
void aiPoll() {}
void aiReset() {}
bool aiPendingOnset(uint32_t &onset) { (void)onset; return false; }
const char *aiMode() { return "dsp"; }
uint32_t aiDropped() { return 0; }
uint32_t aiErrors() { return 0; }
uint32_t aiLastInferenceMs() { return 0; }

#endif
