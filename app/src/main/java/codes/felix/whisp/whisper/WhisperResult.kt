package codes.felix.whisp.whisper

data class WhisperResult(
    val text: String,
    val detectedLanguage: String,
    val durationMs: Long
)
