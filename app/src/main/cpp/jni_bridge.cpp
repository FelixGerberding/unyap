#include <jni.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <mutex>
#include <thread>
#include <vector>
#include <unistd.h>
#include <android/log.h>
#include "whisper.h"
#include "ggml-backend.h"
#include <opus.h>

#define TAG "WhisperJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// Whisper processes audio in 30s windows.
static constexpr int kWindowSamples = 30 * WHISPER_SAMPLE_RATE;

// Silence inserted between the speech segments that VAD keeps, so words from
// neighbouring segments don't run into each other.
static constexpr int kVadGapSamples = WHISPER_SAMPLE_RATE / 5;
static constexpr size_t kMinSamples = WHISPER_SAMPLE_RATE * 3 / 2;

// Tunable parameters. The defaults are what the app uses; the benchmark build
// overrides them through setTuning() to compare alternatives.
struct tuning {
    // Longer audio is transcribed in chunks of this length, cut at the quietest
    // point. Nothing can stream while a chunk is being encoded, so shorter
    // chunks mean shorter pauses between bursts of text. Max must be at least
    // twice min, so that whatever remains after a cut is never shorter than min.
    int chunk_min_samples = 6 * WHISPER_SAMPLE_RATE;
    int chunk_max_samples = 12 * WHISPER_SAMPLE_RATE;

    // Encoder context for audio shorter than one window: 50 frames per second
    // of audio, plus padding, never below a floor. Very small contexts make the
    // model more likely to hallucinate, hence the floor.
    int audio_ctx_pad_frames = 200;  // 4s; 2s measurably increased WER
    int audio_ctx_min_frames = 300;  // 6s

    // Decode chunks with timestamps. Without them, whisper treats a chunk as
    // done when the decoder ends early (it sometimes stops at a sentence end
    // followed by a pause), dropping the rest; with them it resumes from the
    // last timestamp, at the cost of more tokens and an extra encoder pass then.
    bool chunk_timestamps = false;

    int n_threads = 0;          // 0 = pick_n_threads()
    bool use_vad = true;
    // VAD: speech probability threshold, padding kept around speech, and
    // whether to drop non-speech (false = keep all audio, use VAD only to pick
    // chunk cut points).
    float vad_threshold = 0.5f;
    int vad_pad_ms = 200;
    bool vad_drop = true;
    bool vad_agc = true;  // level-normalize the audio VAD sees (see normalize_for_vad)
    std::string cpu_variant;    // e.g. "android_armv8.6_1"; "auto" = ggml's pick;
                                // empty = kPreferredCpuVariants
};
static tuning g_tuning;

// CPU backend variants to try, best first; the first one the CPU supports wins.
// This deliberately skips the SVE/SME variants (armv9.x) that ggml itself scores
// highest: on a Pixel 8 (128-bit SVE2) the armv9.0 variant ran the encoder about
// 3x slower than armv8.6 (NEON + dotprod + i8mm), with identical output.
static const char *kPreferredCpuVariants[] = {
    "android_armv8.6_1", "android_armv8.2_2", "android_armv8.2_1", "android_armv8.0_1",
};

struct engine {
    whisper_context *ctx;
    whisper_vad_context *vad;  // null if the VAD model failed to load
};

struct callback_data {
    JNIEnv *env;
    jobject listener;
    jmethodID onPartial;
    jmethodID onSegment;
    std::thread::id tid;  // the JVM-attached thread whisper_full runs on
    whisper_token eot;
    int last_n_tokens;
    bool partial_enabled;
    std::string last_partial;

    // Timing, all in steady_clock milliseconds since transcription start.
    std::chrono::steady_clock::time_point t_start;
    double t_chunk_start = -1;    // of the current chunk, until its first token
    double prefill_ms = 0;        // summed chunk start -> first token (mel, language
                                  // detection, encoder, prompt)
    double first_text_ms = -1;
};

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

static std::string escape_json(const std::string &s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

// A multi-byte UTF-8 character can be split across tokens. Drop a trailing
// incomplete sequence so partial text never ends in half a character.
static void trim_incomplete_utf8(std::string &s) {
    size_t i = s.size();
    for (int back = 1; back <= 4 && i > 0; back++) {
        unsigned char c = s[--i];
        if ((c & 0xC0) == 0x80) continue;  // continuation byte
        int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
        if (len > back) s.resize(i);
        return;
    }
}

// Token text is raw bytes and not guaranteed to be valid (modified) UTF-8, which
// NewStringUTF requires. Hand bytes to Kotlin and decode there instead.
static void send_text(callback_data *cb, jmethodID method, const std::string &text) {
    jbyteArray bytes = cb->env->NewByteArray((jsize) text.size());
    cb->env->SetByteArrayRegion(bytes, 0, (jsize) text.size(), reinterpret_cast<const jbyte *>(text.data()));
    cb->env->CallVoidMethod(cb->listener, method, bytes);
    cb->env->DeleteLocalRef(bytes);
}

// Fires before each window is encoded: re-enable partial streaming.
static bool encoder_begin_callback(whisper_context *, whisper_state *, void *user_data) {
    auto *cb = static_cast<callback_data *>(user_data);
    cb->last_n_tokens = -1;
    cb->partial_enabled = true;
    return true;
}

// Fires before each token is sampled, with the tokens decoded so far in the
// current window, which we stream as provisional text. Only plain greedy
// decoding is streamed. On temperature fallback whisper restarts the window
// with several decoders, some on worker threads that aren't attached to the
// JVM, so we clear the partial and wait for the window's final segments.
static void logits_filter_callback(
    struct whisper_context *ctx,
    struct whisper_state *,
    const whisper_token_data *tokens,
    int n_tokens,
    float *,
    void *user_data) {

    auto *cb = static_cast<callback_data *>(user_data);
    if (std::this_thread::get_id() != cb->tid) return;
    if (cb->t_chunk_start >= 0) {
        cb->prefill_ms += ms_since(cb->t_start) - cb->t_chunk_start;
        cb->t_chunk_start = -1;
    }
    if (!cb->partial_enabled) return;

    if (n_tokens <= cb->last_n_tokens) {
        cb->partial_enabled = false;
        cb->last_partial.clear();
        send_text(cb, cb->onPartial, "");
        return;
    }
    cb->last_n_tokens = n_tokens;

    std::string text;
    for (int i = 0; i < n_tokens; i++) {
        if (tokens[i].id >= cb->eot) continue;  // skip special tokens
        text += whisper_token_to_str(ctx, tokens[i].id);
    }
    trim_incomplete_utf8(text);
    if (text == cb->last_partial) return;
    cb->last_partial = text;
    if (cb->first_text_ms < 0 && !text.empty()) cb->first_text_ms = ms_since(cb->t_start);
    send_text(cb, cb->onPartial, text);
}

// Fires once a window's segments are final.
static void new_segment_callback(whisper_context *, whisper_state *state, int n_new, void *user_data) {
    auto *cb = static_cast<callback_data *>(user_data);
    const int n_segments = whisper_full_n_segments_from_state(state);
    std::string text;
    for (int i = n_segments - n_new; i < n_segments; i++) {
        text += whisper_full_get_segment_text_from_state(state, i);
    }
    cb->last_partial.clear();
    send_text(cb, cb->onSegment, text);
}

static void log_callback(enum ggml_log_level level, const char *text, void *) {
    if (level == GGML_LOG_LEVEL_DEBUG) return;
    __android_log_write(level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO, "whisper", text);
}

// Little in-order cores (Cortex-A53/A55/A510/A520, Kryo silver) are roughly a
// third as fast as the big ones. ggml splits each op evenly across threads and
// waits for the slowest, so a pass takes about (1 / threads) / slowest_speed:
//   big cores only: 1 / n_big        all cores: 3 / n_all
// Use all cores only when that wins, i.e. when there are few big cores (e.g.
// 2 big + 6 little); otherwise the big ones. On a Pixel 8 (1+4 big, 4 little),
// 5 threads ran 1.4x faster than 9 and 1.75x faster than 6.
static int pick_n_threads() {
    if (g_tuning.n_threads > 0) return g_tuning.n_threads;
    int n_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cores < 1) n_cores = 4;

    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f == nullptr) return n_cores;

    static const std::set<unsigned> little_parts = {
        0xd03, 0xd04, 0xd05, 0xd46, 0xd80,  // Arm Cortex-A53, A35, A55, A510, A520
        0x801, 0x803, 0x805,                // Qualcomm Kryo silver
    };
    int n_big = 0, n_little = 0;
    char line[256];
    unsigned part;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "CPU part : 0x%x", &part) == 1) {
            (little_parts.count(part) ? n_little : n_big)++;
        }
    }
    fclose(f);
    // No per-core info, or all cores alike (e.g. 8x Cortex-A53): use them all.
    if (n_big == 0 || n_little == 0) return n_cores;
    return n_big + n_little > 3 * n_big ? n_big + n_little : n_big;
}

// Audio to transcribe, with the VAD speech probability for every
// samples_per_prob samples of it (empty if VAD is unavailable).
struct speech_audio {
    std::vector<float> pcm;
    std::vector<float> probs;
    int samples_per_prob = 0;
};

// Keep only the speech VAD finds, joined by short silent gaps. If VAD finds no
// speech, keep the full audio rather than risk dropping quiet speech.
// Silero VAD misses very quiet speech (around -60 dBFS), which whisper itself
// transcribes fine, so VAD would silently drop it. Give VAD a copy with the level
// evened out: each 100 ms block is scaled so the loudest block within +-1.5 s
// reaches about -20 dBFS, with the gain capped at +40 dB. Whisper still gets the
// original audio. If this errs, it errs towards keeping noise as speech.
static std::vector<float> normalize_for_vad(const float *pcm, int n_samples) {
    constexpr int kBlock = WHISPER_SAMPLE_RATE / 10;
    constexpr int kReach = 15;  // blocks either side
    constexpr float kTargetRms = 0.1f, kMaxGain = 100.0f;
    const int n_blocks = (n_samples + kBlock - 1) / kBlock;
    std::vector<float> rms(n_blocks);
    for (int b = 0; b < n_blocks; b++) {
        const int s0 = b * kBlock, s1 = std::min(n_samples, s0 + kBlock);
        double sum = 0;
        for (int i = s0; i < s1; i++) sum += (double) pcm[i] * pcm[i];
        rms[b] = (float) std::sqrt(sum / std::max(1, s1 - s0));
    }
    std::vector<float> out(pcm, pcm + n_samples);
    for (int b = 0; b < n_blocks; b++) {
        float env = 0;
        for (int k = std::max(0, b - kReach); k <= std::min(n_blocks - 1, b + kReach); k++) env = std::max(env, rms[k]);
        const float gain = std::clamp(kTargetRms / std::max(env, 1e-6f), 1.0f, kMaxGain);
        const int s0 = b * kBlock, s1 = std::min(n_samples, s0 + kBlock);
        for (int i = s0; i < s1; i++) out[i] = std::clamp(out[i] * gain, -1.0f, 1.0f);
    }
    return out;
}

static bool apply_vad(whisper_vad_context *vad, const float *pcm, int n_samples, speech_audio &out) {
    const std::vector<float> leveled = g_tuning.vad_agc ? normalize_for_vad(pcm, n_samples) : std::vector<float>();
    if (!whisper_vad_detect_speech(vad, g_tuning.vad_agc ? leveled.data() : pcm, n_samples)) return false;
    const float *probs = whisper_vad_probs(vad);
    const int n_probs = whisper_vad_n_probs(vad);
    if (n_probs <= 0) return false;
    const int spp = (n_samples + n_probs - 1) / n_probs;
    auto prob_at = [&](int sample) { return probs[std::min(sample / spp, n_probs - 1)]; };

    whisper_vad_params vparams = whisper_vad_default_params();
    // Only cut real pauses, keep the natural gaps between words.
    vparams.threshold = g_tuning.vad_threshold;
    vparams.min_silence_duration_ms = 500;
    vparams.speech_pad_ms = g_tuning.vad_pad_ms;
    whisper_vad_segments *segments = whisper_vad_segments_from_probs(vad, vparams);
    if (segments == nullptr) return false;

    // Kept ranges as (start in out.pcm, start in pcm, length).
    struct range { int out, in, len; };
    std::vector<range> ranges;
    const int n = g_tuning.vad_drop ? whisper_vad_segments_n_segments(segments) : 0;
    for (int i = 0; i < n; i++) {
        // Segment times are in centiseconds.
        int t0 = (int) (whisper_vad_segments_get_segment_t0(segments, i) / 100.0f * WHISPER_SAMPLE_RATE);
        int t1 = (int) (whisper_vad_segments_get_segment_t1(segments, i) / 100.0f * WHISPER_SAMPLE_RATE);
        t0 = std::clamp(t0, 0, n_samples);
        t1 = std::clamp(t1, t0, n_samples);
        if (t1 == t0) continue;
        LOGI("VAD speech %.2f-%.2fs", t0 / (float) WHISPER_SAMPLE_RATE, t1 / (float) WHISPER_SAMPLE_RATE);
        if (!out.pcm.empty()) out.pcm.insert(out.pcm.end(), kVadGapSamples, 0.0f);
        ranges.push_back({(int) out.pcm.size(), t0, t1 - t0});
        out.pcm.insert(out.pcm.end(), pcm + t0, pcm + t1);
    }
    whisper_vad_free_segments(segments);

    if (out.pcm.empty()) {
        out.pcm.assign(pcm, pcm + n_samples);
        ranges = {{0, 0, n_samples}};
    }
    // whisper_full skips input under 1s, so pad a single short word.
    if (out.pcm.size() < kMinSamples) out.pcm.resize(kMinSamples, 0.0f);

    // Map probabilities onto the joined audio; the inserted gaps are silence.
    out.samples_per_prob = spp;
    out.probs.assign((out.pcm.size() + spp - 1) / spp, 0.0f);
    size_t r = 0;
    for (size_t f = 0; f < out.probs.size(); f++) {
        const int pos = (int) f * spp;
        while (r < ranges.size() && pos >= ranges[r].out + ranges[r].len) r++;
        if (r < ranges.size() && pos >= ranges[r].out) {
            out.probs[f] = prob_at(ranges[r].in + pos - ranges[r].out);
        }
    }
    return true;
}

// Split audio into [start, end) chunks of chunk_min_samples..chunk_max_samples,
// each cut at the frame with the lowest speech probability. Without
// probabilities there is no safe place to cut, so it stays one piece.
static std::vector<std::pair<int, int>> plan_chunks(const speech_audio &audio) {
    const int n = (int) audio.pcm.size();
    std::vector<std::pair<int, int>> chunks;
    int start = 0;
    const int chunk_min = g_tuning.chunk_min_samples, chunk_max = g_tuning.chunk_max_samples;
    while (!audio.probs.empty() && n - start > chunk_max) {
        const int spp = audio.samples_per_prob;
        const int f_lo = (start + chunk_min) / spp;
        const int f_hi = std::min(start + chunk_max, n - chunk_min) / spp;
        int best = f_lo;
        for (int f = f_lo + 1; f <= f_hi; f++) {
            if (audio.probs[f] < audio.probs[best]) best = f;
        }
        chunks.emplace_back(start, best * spp);
        start = best * spp;
    }
    chunks.emplace_back(start, n);
    return chunks;
}

extern "C" {

// ggml's CPU backend is built as multiple dlopen-able variants (one per ARM
// feature level). Load the best one the device supports, exactly once, from the
// app's native library directory. Without this no CPU backend is registered and
// whisper_init fails.
static std::once_flag g_backends_once;

static void load_backends_once(JNIEnv *env, jstring libDir) {
    std::call_once(g_backends_once, [&] {
        whisper_log_set(log_callback, nullptr);
        const char *dir = env->GetStringUTFChars(libDir, nullptr);
        LOGI("Loading ggml CPU backend variants from: %s", dir);
        auto load = [&](const std::string &variant) {
            const std::string lib = std::string(dir) + "/libggml-cpu-" + variant + ".so";
            return ggml_backend_load(lib.c_str()) != nullptr;
        };
        if (g_tuning.cpu_variant == "auto") {
            ggml_backend_load_all_from_path(dir);
        } else if (!g_tuning.cpu_variant.empty()) {
            LOGI("Forcing CPU backend %s", g_tuning.cpu_variant.c_str());
            load(g_tuning.cpu_variant);
        } else {
            for (const char *variant : kPreferredCpuVariants) {
                if (load(variant)) {
                    LOGI("Using CPU backend %s", variant);
                    break;
                }
            }
        }
        env->ReleaseStringUTFChars(libDir, dir);
    });
}

JNIEXPORT jlong JNICALL
Java_codes_felix_whisp_whisper_WhisperEngine_initContext(
    JNIEnv *env, jobject, jstring modelPath, jstring vadModelPath, jstring libDir) {

    load_backends_once(env, libDir);

    const char *path = env->GetStringUTFChars(modelPath, nullptr);
    LOGI("Loading model from: %s", path);

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    cparams.flash_attn = true;
    struct whisper_context *ctx = whisper_init_from_file_with_params(path, cparams);

    env->ReleaseStringUTFChars(modelPath, path);

    if (ctx == nullptr) {
        LOGE("Failed to load whisper model");
        return 0;
    }

    const char *vad_path = env->GetStringUTFChars(vadModelPath, nullptr);
    struct whisper_vad_context_params vparams = whisper_vad_default_context_params();
    vparams.n_threads = pick_n_threads();
    struct whisper_vad_context *vad = whisper_vad_init_from_file_with_params(vad_path, vparams);
    env->ReleaseStringUTFChars(vadModelPath, vad_path);
    if (vad == nullptr) {
        LOGE("Failed to load VAD model, transcribing without it");
    }

    LOGI("Model loaded successfully");
    return reinterpret_cast<jlong>(new engine{ctx, vad});
}

JNIEXPORT jstring JNICALL
Java_codes_felix_whisp_whisper_WhisperEngine_transcribeNative(
    JNIEnv *env, jobject, jlong ctxPtr, jfloatArray samples, jobject listener) {

    auto *eng = reinterpret_cast<engine *>(ctxPtr);
    if (eng == nullptr) {
        return env->NewStringUTF("{\"text\":\"\",\"language\":\"en\"}");
    }
    whisper_context *ctx = eng->ctx;

    jsize n_samples = env->GetArrayLength(samples);
    jfloat *pcm = env->GetFloatArrayElements(samples, nullptr);

    const auto t_start = std::chrono::steady_clock::now();
    speech_audio audio;
    if (!g_tuning.use_vad || eng->vad == nullptr || !apply_vad(eng->vad, pcm, n_samples, audio)) {
        audio = {};
        audio.pcm.assign(pcm, pcm + n_samples);
    }
    env->ReleaseFloatArrayElements(samples, pcm, JNI_ABORT);
    const auto chunks = plan_chunks(audio);
    const auto vad_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_start).count();

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.single_segment = false;
    params.language = nullptr;  // auto-detect
    params.no_context = true;
    params.n_threads = pick_n_threads();

    callback_data cb = {};
    cb.t_start = t_start;
    if (listener != nullptr) {
        jclass cls = env->GetObjectClass(listener);
        cb.env = env;
        cb.listener = listener;
        cb.onPartial = env->GetMethodID(cls, "onPartial", "([B)V");
        cb.onSegment = env->GetMethodID(cls, "onSegment", "([B)V");
        cb.tid = std::this_thread::get_id();
        cb.eot = whisper_token_eot(ctx);
        cb.last_n_tokens = -1;
        cb.partial_enabled = true;
        if (cb.onPartial != nullptr && cb.onSegment != nullptr) {
            params.encoder_begin_callback = encoder_begin_callback;
            params.encoder_begin_callback_user_data = &cb;
            params.logits_filter_callback = logits_filter_callback;
            params.logits_filter_callback_user_data = &cb;
            params.new_segment_callback = new_segment_callback;
            params.new_segment_callback_user_data = &cb;
        }
    }

    LOGI("Starting transcription: %.1fs audio, %.1fs after VAD (%lld ms), %zu chunks, %d threads, streaming=%s",
         n_samples / (float) WHISPER_SAMPLE_RATE, audio.pcm.size() / (float) WHISPER_SAMPLE_RATE, (long long) vad_ms,
         chunks.size(), params.n_threads, listener ? "true" : "false");

    whisper_reset_timings(ctx);
    std::string text;
    int n_segments = 0;
    for (const auto &[start, end] : chunks) {
        const int len = end - start;
        // A chunk that fits in one window needs neither timestamps (they only
        // steer where the next window starts) nor the encoder's full 30s context.
        params.no_timestamps = len <= kWindowSamples && !g_tuning.chunk_timestamps;
        params.audio_ctx = 0;
        if (len <= kWindowSamples) {
            const int frames = len / (WHISPER_SAMPLE_RATE / 50) + g_tuning.audio_ctx_pad_frames;
            params.audio_ctx = std::min(std::max(frames, g_tuning.audio_ctx_min_frames), whisper_n_audio_ctx(ctx));
        }

        cb.t_chunk_start = ms_since(t_start);
        int ret = whisper_full(ctx, params, audio.pcm.data() + start, len);
        if (ret != 0) {
            LOGE("whisper_full failed with code %d", ret);
            return env->NewStringUTF("{\"text\":\"Transcription failed\",\"language\":\"en\"}");
        }

        // Detect the language on the first chunk only. Short chunks give a
        // less reliable guess, and a voice note rarely switches language.
        if (params.language == nullptr) {
            params.language = whisper_lang_str(whisper_full_lang_id(ctx));
        }

        const int n = whisper_full_n_segments(ctx);
        std::string chunk_text;
        for (int i = 0; i < n; i++) {
            const char *seg_text = whisper_full_get_segment_text(ctx, i);
            if (seg_text) {
                chunk_text += seg_text;
            }
        }
        // Never log the text itself: transcripts must not end up in logcat.
        LOGI("chunk %.2f-%.2fs audio_ctx=%d segments=%d chars=%zu", start / (float) WHISPER_SAMPLE_RATE,
             end / (float) WHISPER_SAMPLE_RATE, params.audio_ctx, n, chunk_text.size());
        text += chunk_text;
        n_segments += n;
    }
    whisper_print_timings(ctx);

    const char *lang = params.language;

    // Build JSON response
    const double total_ms = ms_since(t_start);
    std::string json = "{\"text\":\"" + escape_json(text) + "\",\"language\":\"";
    json += (lang ? lang : "en");
    char stats[512];
    snprintf(stats, sizeof(stats),
             "\",\"stats\":{\"audio_s\":%.2f,\"speech_s\":%.2f,\"chunks\":%zu,\"threads\":%d,"
             "\"vad_ms\":%lld,\"prefill_ms\":%.0f,\"first_text_ms\":%.0f,\"total_ms\":%.0f}}",
             n_samples / (double) WHISPER_SAMPLE_RATE, audio.pcm.size() / (double) WHISPER_SAMPLE_RATE,
             chunks.size(), params.n_threads, (long long) vad_ms, cb.prefill_ms, cb.first_text_ms, total_ms);
    json += stats;

    LOGI("Transcription done in %lld ms: %d segments, language=%s",
         (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - t_start).count(),
         n_segments, lang);

    return env->NewStringUTF(json.c_str());
}

JNIEXPORT void JNICALL
Java_codes_felix_whisp_whisper_WhisperEngine_setTuning(
    JNIEnv *env, jobject, jint threads, jfloat chunkMinS, jfloat chunkMaxS,
    jint ctxMinFrames, jint ctxPadFrames, jboolean useVad, jboolean chunkTimestamps,
    jfloat vadThreshold, jint vadPadMs, jboolean vadDrop, jboolean vadAgc, jstring cpuVariant) {

    tuning t;
    if (threads > 0) t.n_threads = threads;
    if (chunkMinS > 0) t.chunk_min_samples = (int) (chunkMinS * WHISPER_SAMPLE_RATE);
    if (chunkMaxS > 0) t.chunk_max_samples = (int) (chunkMaxS * WHISPER_SAMPLE_RATE);
    if (ctxMinFrames > 0) t.audio_ctx_min_frames = ctxMinFrames;
    if (ctxPadFrames >= 0) t.audio_ctx_pad_frames = ctxPadFrames;
    t.use_vad = useVad;
    t.chunk_timestamps = chunkTimestamps;
    if (vadThreshold > 0) t.vad_threshold = vadThreshold;
    if (vadPadMs >= 0) t.vad_pad_ms = vadPadMs;
    t.vad_drop = vadDrop;
    t.vad_agc = vadAgc;
    if (cpuVariant != nullptr) {
        const char *v = env->GetStringUTFChars(cpuVariant, nullptr);
        t.cpu_variant = v;
        env->ReleaseStringUTFChars(cpuVariant, v);
    }
    g_tuning = t;
}

// Decodes demuxed Opus packets (concatenated, with their sizes) straight to
// 16 kHz mono floats. libopus resamples and downmixes internally. preSkip is in
// 48 kHz samples, as stored in the OpusHead header.
JNIEXPORT jfloatArray JNICALL
Java_codes_felix_whisp_audio_OpusDecoder_decodeOpusNative(
    JNIEnv *env, jobject, jbyteArray packets, jintArray sizes, jint preSkip) {
    int err = 0;
    OpusDecoder *dec = opus_decoder_create(WHISPER_SAMPLE_RATE, 1, &err);
    if (dec == nullptr) {
        LOGE("opus_decoder_create failed: %d", err);
        return nullptr;
    }
    const jsize n_packets = env->GetArrayLength(sizes);
    jbyte *data = env->GetByteArrayElements(packets, nullptr);
    jint *len = env->GetIntArrayElements(sizes, nullptr);

    constexpr int kMaxFrame = WHISPER_SAMPLE_RATE * 120 / 1000;  // longest Opus frame, 120 ms
    std::vector<float> out;
    out.reserve((size_t) n_packets * WHISPER_SAMPLE_RATE / 50);
    float frame[kMaxFrame];
    size_t offset = 0;
    for (jsize i = 0; i < n_packets; i++) {
        const int n = opus_decode_float(dec, reinterpret_cast<const unsigned char *>(data) + offset, len[i], frame, kMaxFrame, 0);
        offset += len[i];
        if (n < 0) {
            LOGE("opus_decode_float failed on packet %d: %d", (int) i, n);
            continue;
        }
        out.insert(out.end(), frame, frame + n);
    }
    env->ReleaseIntArrayElements(sizes, len, JNI_ABORT);
    env->ReleaseByteArrayElements(packets, data, JNI_ABORT);
    opus_decoder_destroy(dec);

    const size_t skip = std::min(out.size(), (size_t) (preSkip / 3));  // 48 kHz -> 16 kHz
    jfloatArray result = env->NewFloatArray((jsize) (out.size() - skip));
    env->SetFloatArrayRegion(result, 0, (jsize) (out.size() - skip), out.data() + skip);
    return result;
}

JNIEXPORT void JNICALL
Java_codes_felix_whisp_whisper_WhisperEngine_freeContext(
    JNIEnv *, jobject, jlong ctxPtr) {

    auto *eng = reinterpret_cast<engine *>(ctxPtr);
    if (eng != nullptr) {
        whisper_free(eng->ctx);
        if (eng->vad != nullptr) whisper_vad_free(eng->vad);
        delete eng;
        LOGI("Whisper context freed");
    }
}

} // extern "C"
