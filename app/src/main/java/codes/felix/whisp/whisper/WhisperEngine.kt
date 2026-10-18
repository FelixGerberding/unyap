package codes.felix.whisp.whisper

import android.content.Context
import android.util.Log
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

class WhisperEngine private constructor(
    private val context: Context,
    private val modelFile: File = File(context.filesDir, MODEL_FILENAME)
) {

    @Volatile
    private var contextPtr: Long = 0L
    private val vadModelFile: File = File(context.filesDir, VAD_MODEL_FILENAME)

    var modelReady by mutableStateOf(false)
        private set

    init {
        modelReady = modelFile.exists() && modelFile.length() > 0
    }

    fun refreshModelState() {
        modelReady = modelFile.exists() && modelFile.length() > 0
        if (modelReady) preload()
    }

    /**
     * Loads the model on a background thread, so it is ready (or at least well on
     * its way) by the time the audio has been decoded. Failures are left for
     * [transcribeStreaming] to retry and report.
     */
    fun preload() {
        if (!modelReady || contextPtr != 0L) return
        thread(name = "whisper-preload") {
            try {
                loadModel()
            } catch (e: Exception) {
                Log.w(TAG, "Model preload failed", e)
            }
        }
    }

    /** Loads the model on the calling thread, if it isn't loaded yet. */
    fun loadBlocking() = loadModel()

    @Synchronized
    private fun loadModel() {
        if (contextPtr != 0L) return
        // The VAD model ships in the APK; the native side needs it as a file.
        if (vadModelFile.length() == 0L) {
            context.assets.open(VAD_MODEL_FILENAME).use { input ->
                vadModelFile.outputStream().use { input.copyTo(it) }
            }
        }
        contextPtr = initContext(
            modelFile.absolutePath,
            vadModelFile.absolutePath,
            context.applicationInfo.nativeLibraryDir
        )
        if (contextPtr == 0L) {
            throw IllegalStateException("Failed to load whisper model - file may be corrupted")
        }
    }

    fun transcribeStreaming(pcm: FloatArray): Flow<StreamEvent> = callbackFlow<StreamEvent> {
        thread {
            try {
                loadModel()
                val startTime = System.currentTimeMillis()

                val listener = object : SegmentListener {
                    override fun onPartial(text: ByteArray) {
                        trySend(StreamEvent.Partial(String(text, Charsets.UTF_8)))
                    }

                    override fun onSegment(text: ByteArray) {
                        trySend(StreamEvent.Segment(String(text, Charsets.UTF_8)))
                    }
                }

                val jsonStr = transcribeNative(contextPtr, pcm, listener)
                val elapsed = System.currentTimeMillis() - startTime

                val json = JSONObject(jsonStr)
                val result = WhisperResult(
                    text = json.getString("text").trim(),
                    detectedLanguage = json.optString("language", "en"),
                    durationMs = elapsed,
                    stats = json.optJSONObject("stats")
                )
                trySend(StreamEvent.Done(result))
            } catch (e: Exception) {
                close(e)
            }
            close()
        }
        awaitClose()
    }.buffer(Channel.UNLIMITED)

    @Synchronized
    fun deleteModel() {
        if (contextPtr != 0L) {
            freeContext(contextPtr)
            contextPtr = 0L
        }
        modelFile.delete()
        modelReady = false
    }

    // Called from native code with raw UTF-8 bytes.
    interface SegmentListener {
        fun onPartial(text: ByteArray)
        fun onSegment(text: ByteArray)
    }

    sealed class StreamEvent {
        /** Provisional text of the window being decoded; replaces the previous partial. */
        data class Partial(val text: String) : StreamEvent()

        /** Final text of a finished window; clears the partial. */
        data class Segment(val text: String) : StreamEvent()
        data class Done(val result: WhisperResult) : StreamEvent()
    }

    private external fun initContext(modelPath: String, vadModelPath: String, libDir: String): Long
    private external fun transcribeNative(ctxPtr: Long, samples: FloatArray, listener: SegmentListener?): String
    private external fun freeContext(ctxPtr: Long)

    /**
     * Overrides native tuning defaults (0 or negative keeps a default; ctxPadFrames
     * keeps it only when negative). Benchmark use only: call before the first model
     * load, since the CPU backend is chosen once per process.
     */
    external fun setTuning(
        threads: Int, chunkMinS: Float, chunkMaxS: Float,
        ctxMinFrames: Int, ctxPadFrames: Int, useVad: Boolean, chunkTimestamps: Boolean,
        vadThreshold: Float, vadPadMs: Int, vadDrop: Boolean, vadAgc: Boolean, cpuVariant: String?
    )

    companion object {
        const val MODEL_FILENAME = "ggml-small-q8_0.bin"
        const val MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q8_0.bin"
        private const val VAD_MODEL_FILENAME = "ggml-silero-v6.2.0.bin"
        private const val TAG = "WhisperEngine"

        @Volatile
        private var instance: WhisperEngine? = null

        init {
            System.loadLibrary("whisperjni")
        }

        /** A separate engine for an arbitrary model file, for benchmarking. */
        fun forModel(context: Context, modelFile: File): WhisperEngine =
            WhisperEngine(context.applicationContext, modelFile)

        fun getInstance(context: Context): WhisperEngine {
            return instance ?: synchronized(this) {
                instance ?: WhisperEngine(context.applicationContext).also { instance = it }
            }
        }
    }
}
