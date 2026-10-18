package codes.felix.whisp.whisper

import org.json.JSONObject

data class WhisperResult(
    val text: String,
    val detectedLanguage: String,
    val durationMs: Long,
    /** Native timing breakdown (see jni_bridge.cpp). */
    val stats: JSONObject? = null
)
