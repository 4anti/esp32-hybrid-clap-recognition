#pragma once

#if __has_include("clap_local.h")
#include "clap_local.h"
#endif
#ifndef CLAP_LAB_PASSWORD
#define CLAP_LAB_PASSWORD ""
#endif

// Experimental room-adapted verifier adds a veto to the existing DSP policy.
// Set to 0 for a DSP-only build with no AI allocations.
#ifndef CLAP_ENABLE_AI
#define CLAP_ENABLE_AI 1
#endif

// Local test default: active veto after recorded pipeline/INT8 checks. Live
// gesture and sustained room-noise testing remain required. Set 1 for comparison
// mode, or use tools/build_firmware.ps1 -Mode shadow to bypass the AI veto.
#ifndef CLAP_AI_SHADOW
#define CLAP_AI_SHADOW 0
#endif

#define CLAP_SAMPLE_RATE 16000
#define CLAP_AI_WINDOW_SAMPLES 8000
// The model consumes only frames 32..47 of the logical 500 ms window.
// Keeping its required 80 ms history saves 40 KiB across history and slots.
#define CLAP_AI_PRE_SAMPLES 1280
#define CLAP_AI_POST_SAMPLES 1600
#define CLAP_AI_SLOTS 3
#define CLAP_AI_ARENA_BYTES (96 * 1024)

// Starting points, not calibrated guarantees. Tune using held-out recordings.
#define CLAP_AI_MIN_TARGET 0.85f
#define CLAP_AI_MAX_NOISE 0.15f
#define CLAP_AI_MIN_MARGIN 0.35f
