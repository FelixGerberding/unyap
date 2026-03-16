package codes.felix.whisp.whisper

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import org.json.JSONObject
import java.io.File
import kotlin.concurrent.thread

class WhisperEngine private constructor(private val context: Context) {

    private var contextPtr: Long = 0L
    private val modelFile: File = File(context.filesDir, MODEL_FILENAME)

    var modelReady by mutableStateOf(false)
        private set

    init {
        modelReady = modelFile.exists() && modelFile.length() > 0
    }

    fun refreshModelState() {
        modelReady = modelFile.exists() && modelFile.length() > 0
        if (modelReady && contextPtr == 0L) {
            loadModel()
        }
    }

    private fun loadModel() {
        if (contextPtr != 0L) return
        contextPtr = initContext(modelFile.absolutePath, context.applicationInfo.nativeLibraryDir)
        if (contextPtr == 0L) {
            throw IllegalStateException("Failed to load whisper model - file may be corrupted")
        }
    }

    fun transcribeStreaming(pcm: FloatArray): Flow<StreamEvent> = callbackFlow<StreamEvent> {
        thread {
            try {
                if (contextPtr == 0L) {
                    loadModel()
                }
                val startTime = System.currentTimeMillis()

                val listener = object : SegmentListener {
                    override fun onNewSegment(text: String) {
                        trySend(StreamEvent.Segment(text))
                    }
                }

                val jsonStr = transcribeNative(contextPtr, pcm, listener)
                val elapsed = System.currentTimeMillis() - startTime

                val json = JSONObject(jsonStr)
                val result = WhisperResult(
                    text = json.getString("text").trim(),
                    detectedLanguage = json.optString("language", "en"),
                    durationMs = elapsed
                )
                trySend(StreamEvent.Done(result))
            } catch (e: Exception) {
                close(e)
            }
            close()
        }
        awaitClose()
    }.buffer(Channel.UNLIMITED)

    fun deleteModel() {
        if (contextPtr != 0L) {
            freeContext(contextPtr)
            contextPtr = 0L
        }
        modelFile.delete()
        modelReady = false
    }

    interface SegmentListener {
        fun onNewSegment(text: String)
    }

    sealed class StreamEvent {
        data class Segment(val text: String) : StreamEvent()
        data class Done(val result: WhisperResult) : StreamEvent()
    }

    private external fun initContext(modelPath: String, libDir: String): Long
    private external fun transcribeNative(ctxPtr: Long, samples: FloatArray, listener: SegmentListener?): String
    private external fun freeContext(ctxPtr: Long)

    companion object {
        const val MODEL_FILENAME = "ggml-small-q8_0.bin"
        const val MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q8_0.bin"

        @Volatile
        private var instance: WhisperEngine? = null

        init {
            System.loadLibrary("whisperjni")
        }

        fun getInstance(context: Context): WhisperEngine {
            return instance ?: synchronized(this) {
                instance ?: WhisperEngine(context.applicationContext).also { instance = it }
            }
        }
    }
}
