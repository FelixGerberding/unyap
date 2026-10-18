package codes.felix.whisp

import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.util.Log
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.Text
import androidx.lifecycle.lifecycleScope
import codes.felix.whisp.audio.OpusDecoder
import codes.felix.whisp.whisper.WhisperEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.net.URL

/**
 * Benchmark entry point, only present in the `benchmark` build type. Transcribes
 * one file with the given model and tuning, then appends a JSON line with timings
 * and the transcript to bench_results.jsonl in the external files dir. Meant to
 * run in a fresh process per invocation (force-stop between runs), so model load
 * and CPU backend selection are measured cold.
 *
 *   adb shell am start -W -n codes.felix.whisp.bench/codes.felix.whisp.BenchActivity \
 *       --es audio de_60s.ogg --es model ggml-small-q8_0.bin --ei threads 5 \
 *       --ef chunkMax 12 --es tag mytest
 *
 * Audio (notes/<name>) and models (models/<name>) are fetched once from an HTTP
 * server on the host, reached through `adb reverse tcp:8765 tcp:8765`, and cached
 * in private storage. (Files adb pushes into the app's external dir are not
 * visible to the app.) Results go to the external files dir, which adb can read.
 * Optional: --ef chunkMin, --ei ctxMin, --ei ctxPad, --ez vad, --ez timestamps,
 * --ef vadThreshold, --ei vadPad, --ez vadDrop, --ez vadAgc, --es variant.
 */
class BenchActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Run as the top app even when the phone is locked: behind the lock
        // screen Android moves the process to the background CPU group, which
        // throttles it heavily and makes the numbers meaningless.
        setShowWhenLocked(true)
        setTurnScreenOn(true)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContent { Text("Benchmarking…") }

        val extras = intent.extras ?: Bundle()
        val base = getExternalFilesDir(null)!!
        val audio = extras.getString("audio")!!
        val model = extras.getString("model", WhisperEngine.MODEL_FILENAME)
        val chunkMax = extras.getFloat("chunkMax", 0f)
        val chunkMin = extras.getFloat("chunkMin", if (chunkMax > 0) chunkMax / 2 else 0f)

        val record = JSONObject()
            .put("tag", extras.getString("tag", ""))
            .put("audio", audio)
            .put("model", model)
            .put("threads_req", extras.getInt("threads", 0))
            .put("chunk_min", chunkMin.toDouble())
            .put("chunk_max", chunkMax.toDouble())
            .put("ctx_min", extras.getInt("ctxMin", 0))
            .put("ctx_pad", extras.getInt("ctxPad", -1))
            .put("vad", extras.getBoolean("vad", true))
            .put("variant", extras.getString("variant", ""))
            .put("timestamps", extras.getBoolean("timestamps", false))
            .put("thermal_before", thermalStatus())
            .put("headroom_before", thermalHeadroom())

        lifecycleScope.launch {
            try {
                val modelFile = fetch("models/$model")
                val audioFile = fetch("notes/$audio")
                val engine = WhisperEngine.forModel(this@BenchActivity, modelFile)
                engine.setTuning(
                    extras.getInt("threads", 0), chunkMin, chunkMax,
                    extras.getInt("ctxMin", 0), extras.getInt("ctxPad", -1),
                    extras.getBoolean("vad", true), extras.getBoolean("timestamps", false),
                    extras.getFloat("vadThreshold", 0f), extras.getInt("vadPad", -1),
                    extras.getBoolean("vadDrop", true), extras.getBoolean("vadAgc", true),
                    extras.getString("variant")
                )

                var t = System.nanoTime()
                record.put("extract_only_ms", extractOnlyMs(audioFile))
                t = System.nanoTime()
                val pcm = OpusDecoder.decodeToFloatArray(this@BenchActivity, Uri.fromFile(audioFile))
                record.put("audio_decode_ms", (System.nanoTime() - t) / 1_000_000)

                t = System.nanoTime()
                engine.loadBlocking()
                record.put("load_ms", (System.nanoTime() - t) / 1_000_000)

                engine.transcribeStreaming(pcm).collect { event ->
                    if (event is WhisperEngine.StreamEvent.Done) {
                        record.put("text", event.result.text)
                            .put("language", event.result.detectedLanguage)
                            .put("stats", event.result.stats)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Benchmark failed", e)
                record.put("error", e.toString())
            }
            record.put("thermal_after", thermalStatus())
            File(base, "bench_results.jsonl").appendText(record.toString() + "\n")
            Log.i(TAG, "BENCH_DONE ${record.optString("tag")}")
            finish()
        }
    }

    /** Time to demux every packet without decoding, to separate demuxer from codec cost. */
    private fun extractOnlyMs(file: File): Long {
        val t = System.nanoTime()
        val extractor = android.media.MediaExtractor()
        extractor.setDataSource(file.path)
        extractor.selectTrack(0)
        val buf = java.nio.ByteBuffer.allocate(1 shl 16)
        var packets = 0
        while (extractor.readSampleData(buf, 0) >= 0) {
            packets++
            extractor.advance()
        }
        extractor.release()
        Log.i(TAG, "extract-only: $packets packets")
        return (System.nanoTime() - t) / 1_000_000
    }

    private suspend fun fetch(path: String): File = withContext(Dispatchers.IO) {
        val file = File(filesDir, "bench/$path")
        if (!file.exists()) {
            file.parentFile!!.mkdirs()
            val tmp = File(file.path + ".part")
            URL("http://127.0.0.1:8765/$path").openStream().use { input ->
                tmp.outputStream().use { input.copyTo(it) }
            }
            tmp.renameTo(file)
        }
        file
    }

    private fun thermalStatus(): Int =
        getSystemService(PowerManager::class.java).currentThermalStatus

    private fun thermalHeadroom(): Double =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            getSystemService(PowerManager::class.java).getThermalHeadroom(0).toDouble()
        } else {
            -1.0
        }

    companion object {
        private const val TAG = "Bench"
    }
}
