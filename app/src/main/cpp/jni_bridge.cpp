#include <jni.h>
#include <string>
#include <mutex>
#include <unistd.h>
#include <android/log.h>
#include "whisper.h"
#include "ggml-backend.h"

#define TAG "WhisperJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct callback_data {
    JNIEnv *env;
    jobject listener;
    jmethodID onSegment;
    int last_sent_tokens;  // track how many tokens we've already sent
};

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

// Fires before each token is sampled - we use it to send the last decoded token
static void logits_filter_callback(
    struct whisper_context *ctx,
    struct whisper_state *state,
    const whisper_token_data *tokens,
    int n_tokens,
    float *logits,
    void *user_data) {

    auto *cb = static_cast<callback_data *>(user_data);
    if (!cb || !cb->listener || n_tokens <= 0) return;

    // Send the most recently sampled token (the last one in the tokens array)
    // Skip if it's a special token (timestamp, EOT, etc.)
    whisper_token eot = whisper_token_eot(ctx);
    const whisper_token_data &last = tokens[n_tokens - 1];

    if (last.id >= eot) return;  // skip special tokens

    const char *text = whisper_token_to_str(ctx, last.id);
    if (text && text[0] != '\0') {
        jstring jtext = cb->env->NewStringUTF(text);
        cb->env->CallVoidMethod(cb->listener, cb->onSegment, jtext);
        cb->env->DeleteLocalRef(jtext);
    }
}

extern "C" {

// ggml's CPU backend is built as multiple dlopen-able variants (one per ARM
// feature level). Load the best one the device supports, exactly once, from the
// app's native library directory. Without this no CPU backend is registered and
// whisper_init fails.
static std::once_flag g_backends_once;

static void load_backends_once(JNIEnv *env, jstring libDir) {
    std::call_once(g_backends_once, [&] {
        const char *dir = env->GetStringUTFChars(libDir, nullptr);
        LOGI("Loading ggml CPU backend variants from: %s", dir);
        ggml_backend_load_all_from_path(dir);
        env->ReleaseStringUTFChars(libDir, dir);
    });
}

JNIEXPORT jlong JNICALL
Java_codes_felix_whisp_whisper_WhisperEngine_initContext(
    JNIEnv *env, jobject, jstring modelPath, jstring libDir) {

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

    LOGI("Model loaded successfully");
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT jstring JNICALL
Java_codes_felix_whisp_whisper_WhisperEngine_transcribeNative(
    JNIEnv *env, jobject, jlong ctxPtr, jfloatArray samples, jobject listener) {

    auto *ctx = reinterpret_cast<struct whisper_context *>(ctxPtr);
    if (ctx == nullptr) {
        return env->NewStringUTF("{\"text\":\"\",\"language\":\"en\"}");
    }

    jsize n_samples = env->GetArrayLength(samples);
    jfloat *pcm = env->GetFloatArrayElements(samples, nullptr);

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.single_segment = false;
    params.language = nullptr;  // auto-detect
    params.no_context = true;
    int n_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cores < 1) n_cores = 4;
    params.n_threads = n_cores;

    // Set up per-token callback via logits_filter
    callback_data cb = {};
    if (listener != nullptr) {
        jclass cls = env->GetObjectClass(listener);
        cb.env = env;
        cb.listener = listener;
        cb.onSegment = env->GetMethodID(cls, "onNewSegment", "(Ljava/lang/String;)V");
        cb.last_sent_tokens = 0;
        if (cb.onSegment != nullptr) {
            params.logits_filter_callback = logits_filter_callback;
            params.logits_filter_callback_user_data = &cb;
        }
    }

    LOGI("Starting transcription with %d samples, %d threads, streaming=%s",
         n_samples, params.n_threads, listener ? "true" : "false");

    int ret = whisper_full(ctx, params, pcm, n_samples);

    env->ReleaseFloatArrayElements(samples, pcm, JNI_ABORT);

    if (ret != 0) {
        LOGE("whisper_full failed with code %d", ret);
        return env->NewStringUTF("{\"text\":\"Transcription failed\",\"language\":\"en\"}");
    }

    // Collect all segments for the final result
    std::string text;
    int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; i++) {
        const char *seg_text = whisper_full_get_segment_text(ctx, i);
        if (seg_text) {
            text += seg_text;
        }
    }

    // Get detected language
    int lang_id = whisper_full_lang_id(ctx);
    const char *lang = whisper_lang_str(lang_id);

    // Build JSON response
    std::string json = "{\"text\":\"" + escape_json(text) + "\",\"language\":\"";
    json += (lang ? lang : "en");
    json += "\"}";

    LOGI("Transcription done: %d segments, language=%s", n_segments, lang);

    return env->NewStringUTF(json.c_str());
}

JNIEXPORT void JNICALL
Java_codes_felix_whisp_whisper_WhisperEngine_freeContext(
    JNIEnv *, jobject, jlong ctxPtr) {

    auto *ctx = reinterpret_cast<struct whisper_context *>(ctxPtr);
    if (ctx != nullptr) {
        whisper_free(ctx);
        LOGI("Whisper context freed");
    }
}

} // extern "C"
